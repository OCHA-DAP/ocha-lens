# ===========================================================================
# Ported verbatim — DO NOT MODIFY (this is the refactor source-of-truth)
# Source repo:   OCHA-DAP/ds-storm-impact-harmonisation
# Source ref:    e692da12b17b29926a5d704df94fc31f2d7541fa (origin/gdacs-adam-data)
# Source path:   pipelines/join_historical_exposure.py
# Pulled:        2026-05-07
# Provenance:    https://github.com/OCHA-DAP/ds-storm-impact-harmonisation/pull/2
# Refactored functions live in src/ocha_lens/datasources/ — see PR description.
# ===========================================================================

"""
Join ADAM + GDACS + CHD — Historical ADM0 and ADM1 Exposure

Loads ADM0-level population exposure estimates from the ADAM, GDACS, and CHD
historical pipelines and joins them into a single dataset where each row is one
storm x one ADM0 region. Also joins ADM1-level data from ADAM and GDACS, and
exports both combined datasets as JSON.

Sources:
    adam_historical_adm0_exposure.csv  — WFP ADAM wind exposure at 34/50/64 kt
                                         (cumulative, applied in adam pipeline)
    gdacs_historical_adm0_exposure.csv — GDACS wind exposure at 34 kt / 64 kt
    adm0_ibtracs_exp_all.parquet       — CHD wind exposure at 34/50/64 kt
    adam_historical_adm1_exposure.csv  — WFP ADAM ADM1 wind exposure at 34/50/64 kt
    gdacs_historical_adm1_exposure.csv — GDACS ADM1 wind exposure at 34 kt / 64 kt

Join key: all shared metadata columns (everything except population exposure columns)
CHD join key: sid + iso3 only (CHD does not carry the full metadata set)
ADM1 join key: JOIN_COLS + adm1_name (note: ADAM and GDACS may use different names)
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import ocha_stratus as stratus
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from constants import CHD_ZERO_FILL_ISO3, PROJECT_PREFIX

load_dotenv()

ADAM_ADM0_BLOB = f"{PROJECT_PREFIX}/processed/adam_historical_adm0_exposure.csv"
GDACS_ADM0_BLOB = f"{PROJECT_PREFIX}/processed/gdacs_historical_adm0_exposure.csv"
CHD_BLOB = f"{PROJECT_PREFIX}/processed/adm0_ibtracs_exp_all.parquet"
OUTPUT_ADM0_CSV = f"{PROJECT_PREFIX}/processed/combined_historical_adm0_exposure.csv"
OUTPUT_JSON = Path(__file__).resolve().parents[1] / "assets" / "exposure_data.json"

ADAM_ADM1_BLOB = f"{PROJECT_PREFIX}/processed/adam_historical_adm1_exposure.csv"
GDACS_ADM1_BLOB = f"{PROJECT_PREFIX}/processed/gdacs_historical_adm1_exposure.csv"
OUTPUT_ADM1_JSON = (
    Path(__file__).resolve().parents[1] / "assets" / "adm1_exposure_data.json"
)


JOIN_COLS = [
    "sid",
    "iso3",
    "from_date",
    "season",
    "name",
    "atcf_id",
    "number",
    "genesis_basin",
    "event_id",
    "episode_id",
    "provisional",
    "storm_id",
]

JOIN_COLS_ADM1 = JOIN_COLS + ["adm1_name"]


def pivot_chd(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot CHD data from long format to wide format with source-labelled columns.

    Input has one row per storm x country x wind-speed threshold (34/50/64 kt).
    Output has one row per storm x country with columns pop_34kt_chd, pop_50kt_chd,
    pop_64kt_chd.
    """
    df_wide = (
        df.rename(columns={"ADM0_A3": "iso3"})
        .pivot_table(index=["sid", "iso3"], columns="speed", values="pop_exposed")
        .reset_index()
    )
    df_wide.columns.name = None
    df_wide = df_wide.rename(
        columns={34: "pop_34kt_chd", 50: "pop_50kt_chd", 64: "pop_64kt_chd"}
    )
    return df_wide


def merge_adam_gdacs(df_adam, df_gdacs, join_cols):
    """Outer-join ADAM and GDACS on join_cols and consolidate duplicate columns."""
    df_merged = df_gdacs.merge(
        df_adam,
        on=join_cols,
        how="outer",
        suffixes=("_gdacs", "_adam"),
    )
    df_merged["alert_level"] = df_merged["alert_level_gdacs"].fillna(
        df_merged["alert_level_adam"]
    )
    df_merged["source"] = df_merged["source_gdacs"].fillna(df_merged["source_adam"])
    drop_cols = [
        "alert_level_gdacs",
        "alert_level_adam",
        "source_gdacs",
        "source_adam",
    ]
    has_cn = (
        "country_name_gdacs" in df_merged.columns
        and "country_name_adam" in df_merged.columns
    )
    if has_cn:
        df_merged["country_name"] = df_merged["country_name_gdacs"].fillna(
            df_merged["country_name_adam"]
        )
        drop_cols += ["country_name_gdacs", "country_name_adam"]
    drop_cols += [c for c in ("index_gdacs", "index_adam") if c in df_merged.columns]
    df_merged = df_merged.drop(columns=drop_cols)

    # pop_50kt comes only from ADAM; label it explicitly for consistency
    if "pop_50kt" in df_merged.columns:
        df_merged = df_merged.rename(columns={"pop_50kt": "pop_50kt_adam"})

    return df_merged


def main():
    # -----------------------------------------------------------------------
    # 1. Load all datasets from blob storage
    # -----------------------------------------------------------------------
    print("Loading datasets from blob storage...")
    df_adam = stratus.load_csv_from_blob(ADAM_ADM0_BLOB)
    df_gdacs = stratus.load_csv_from_blob(GDACS_ADM0_BLOB)
    df_chd = stratus.load_parquet_from_blob(CHD_BLOB)
    print(f"ADAM rows:  {len(df_adam)},  columns: {list(df_adam.columns)}")
    print(f"GDACS rows: {len(df_gdacs)}, columns: {list(df_gdacs.columns)}")
    print(f"CHD rows:   {len(df_chd)},  columns: {list(df_chd.columns)}")

    # -----------------------------------------------------------------------
    # 2. Replace -1 sentinel values in GDACS data with null
    # -----------------------------------------------------------------------
    df_gdacs = df_gdacs.replace(-1, pd.NA)
    print("Replaced -1 values in GDACS data with null.")

    # -----------------------------------------------------------------------
    # 3. Outer join ADAM + GDACS on shared metadata columns
    # -----------------------------------------------------------------------
    df_merged = merge_adam_gdacs(df_adam, df_gdacs, JOIN_COLS)
    print(f"Combined columns after ADAM+GDACS merge: {df_merged.shape}")

    # -----------------------------------------------------------------------
    # 5. Join CHD data
    # -----------------------------------------------------------------------
    df_chd_wide = pivot_chd(df_chd)
    print(f"CHD wide rows: {len(df_chd_wide)}, columns: {list(df_chd_wide.columns)}")

    df_merged = df_merged.merge(df_chd_wide, on=["sid", "iso3"], how="outer")
    print(f"Combined columns after CHD join, final shape: {df_merged.shape}")

    # -----------------------------------------------------------------------
    # 5b. Zero-fill missing CHD estimates
    #
    # Fill CHD pop columns with 0 where all three conditions hold:
    #   1. The SID exists in storms.ibtracs_storms
    #   2. ADAM or GDACS already has an estimate for that storm x country row
    #   3. The ISO3 is in the CHD_ZERO_FILL_ISO3 allowlist
    # -----------------------------------------------------------------------
    engine = stratus.get_engine("prod")
    with engine.connect() as conn:
        ibtracs_sids = set(
            pd.read_sql("SELECT DISTINCT sid FROM storms.ibtracs_storms", conn)[
                "sid"
            ].dropna()
        )
    print(f"ibtracs SIDs loaded: {len(ibtracs_sids)}")

    adam_gdacs_cols = [
        "pop_34kt_adam",
        "pop_50kt_adam",
        "pop_64kt_adam",
        "pop_34kt_gdacs",
        "pop_64kt_gdacs",
    ]
    chd_pop_cols = ["pop_34kt_chd", "pop_50kt_chd", "pop_64kt_chd"]

    fill_mask = (
        df_merged["sid"].isin(ibtracs_sids)
        & df_merged["iso3"].isin(CHD_ZERO_FILL_ISO3)
        & df_merged[adam_gdacs_cols].notna().any(axis=1)
    )
    for col in chd_pop_cols:
        df_merged.loc[fill_mask & df_merged[col].isna(), col] = 0

    filled = fill_mask.sum()
    print(f"Zero-filled CHD estimates for {filled} storm x country rows")

    # -----------------------------------------------------------------------
    # 6. Save combined dataset to Azure blob storage
    # -----------------------------------------------------------------------
    print(f"\nSaving {len(df_merged)} rows to {OUTPUT_ADM0_CSV}...")
    stratus.upload_csv_to_blob(df_merged, OUTPUT_ADM0_CSV)

    # -----------------------------------------------------------------------
    # 7. Export to JSON for dashboard (only storms with a GDACS event_id)
    # -----------------------------------------------------------------------
    df_json = df_merged[df_merged["event_id"].notna()]
    print(f"Dropped {len(df_merged) - len(df_json)} rows without GDACS event_id")
    df_json = df_json.replace({pd.NA: None, float("nan"): None})
    df_json = df_json.where(pd.notna(df_json), None)
    data_records = df_json.to_dict(orient="records")

    output = {
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data": data_records,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Exported {len(data_records)} records to {OUTPUT_JSON}")

    # -----------------------------------------------------------------------
    # 8. Join ADM1 data (ADAM + GDACS only) and export to JSON
    # -----------------------------------------------------------------------
    print("\nLoading ADM1 datasets from blob storage...")
    df_adam_adm1 = stratus.load_csv_from_blob(ADAM_ADM1_BLOB)
    df_gdacs_adm1 = stratus.load_csv_from_blob(GDACS_ADM1_BLOB)
    df_gdacs_adm1 = df_gdacs_adm1.replace(-1, pd.NA)
    print(f"ADAM ADM1 rows: {len(df_adam_adm1)}")
    print(f"GDACS ADM1 rows: {len(df_gdacs_adm1)}")

    df_adm1 = merge_adam_gdacs(df_adam_adm1, df_gdacs_adm1, JOIN_COLS_ADM1)
    print(f"Combined ADM1 shape after merge: {df_adm1.shape}")

    df_adm1_json = df_adm1[df_adm1["event_id"].notna()]
    df_adm1_json = df_adm1_json.replace({pd.NA: None, float("nan"): None})
    df_adm1_json = df_adm1_json.where(pd.notna(df_adm1_json), None)
    adm1_records = df_adm1_json.to_dict(orient="records")
    with open(OUTPUT_ADM1_JSON, "w") as f:
        json.dump({"data": adm1_records}, f, indent=2)
    print(f"Exported {len(adm1_records)} ADM1 records to {OUTPUT_ADM1_JSON}")

    print("Done.")


if __name__ == "__main__":
    main()
