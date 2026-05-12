# ===========================================================================
# Ported verbatim — DO NOT MODIFY (this is the refactor source-of-truth)
# Source repo:   OCHA-DAP/ds-storm-impact-harmonisation
# Source ref:    e692da12b17b29926a5d704df94fc31f2d7541fa (origin/gdacs-adam-data)
# Source path:   pipelines/adam_historical_exposure.py
# Pulled:        2026-05-07
# Provenance:    https://github.com/OCHA-DAP/ds-storm-impact-harmonisation/pull/2
# Refactored functions live in src/ocha_lens/datasources/ — see PR description.
# ===========================================================================

"""
WFP ADAM — Historical ADM0 Exposure

Retrieves ADM0-level population exposure for all tropical storms, using the
WFP ADAM OGC API Features endpoint for full historical coverage. For each storm,
retrieves only the latest available episode. Saves ADM0 and ADM1 level exposure
estimates to blob storage.

API path:
    collections/adam.adam_ts_events/items  (paginated)
      -> deduplicate to latest episode_id per event_id
        -> population_csv_url -> aggregate ADM2 rows by ADM0_NAME

Wind speed thresholds in output columns:
    pop_60kmh  — population exposed to winds >= 60 km/h (~32 kt, tropical storm force)
    pop_90kmh  — population exposed to winds >= 90 km/h (~49 kt)
    pop_120kmh — population exposed to winds >= 120 km/h (~65 kt, hurricane force)
"""

import base64
import json
import sys
from pathlib import Path

import ocha_stratus as stratus
import pandas as pd
import pycountry
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from constants import PROJECT_PREFIX

load_dotenv()

ADAM_BASE = "https://api.adam.geospatial.wfp.org/api"
COLLECTION = "adam.adam_ts_events"
PAGE_SIZE = 100
OUTPUT_ADM0_CSV = f"{PROJECT_PREFIX}/processed/adam_historical_adm0_exposure.csv"
OUTPUT_ADM1_CSV = f"{PROJECT_PREFIX}/processed/adam_historical_adm1_exposure.csv"
SOURCE = "NOAA"


# ---------------------------------------------------------------------------
# 1. Fetch all TC event records
# ---------------------------------------------------------------------------


def fetch_all_events():
    all_items = []
    offset = 0

    while True:
        resp = requests.get(
            f"{ADAM_BASE}/collections/{COLLECTION}/items",
            params={"limit": PAGE_SIZE, "offset": offset, "source": SOURCE},
            timeout=30,
        )
        resp.raise_for_status()
        if not resp.content:
            break
        data = json.loads(base64.b64decode(resp.content))
        features = data.get("features", [])
        if not features:
            break

        for f in features:
            p = f.get("properties", f)
            all_items.append(
                {
                    "event_id": p.get("event_id"),
                    "episode_id": p.get("episode_id"),
                    "uid": p.get("uid", ""),
                    "storm_name": p.get("name", ""),
                    "source": p.get("source", ""),
                    "from_date": p.get("from_date", ""),
                    "alert_level": p.get("alert_level", ""),
                    "population_csv_url": p.get("population_csv_url", ""),
                }
            )

        print(f"Offset {offset:>5}: {len(features)} fetched  (total: {len(all_items)})")
        offset += PAGE_SIZE
        if len(features) < PAGE_SIZE:
            break

    return pd.DataFrame(all_items)


# ---------------------------------------------------------------------------
# 2. Deduplicate to latest episode per event
# ---------------------------------------------------------------------------


def deduplicate(df_all):
    return (
        df_all.sort_values("episode_id", ascending=False)
        .drop_duplicates(subset="event_id", keep="first")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 3. Fetch and aggregate population exposure
# ---------------------------------------------------------------------------

_iso3_cache = {}

# Hard-coded overrides for names that pycountry cannot resolve or resolves incorrectly.
# None means the ISO3 is genuinely ambiguous and should be left blank.
_ISO3_OVERRIDES = {
    "Puerto Rico (USA)": "PRI",
    "United States Virgin Islands (USA)": "VIR",
    "Saint Pierre et Miquelon": "SPM",
    "Azores Islands": "PRT",
    "Clipperton Island": "CPT",
}


def name_to_iso3(country_name):
    if country_name in _iso3_cache:
        return _iso3_cache[country_name]
    if country_name in _ISO3_OVERRIDES:
        code = _ISO3_OVERRIDES[country_name]
    else:
        try:
            code = pycountry.countries.search_fuzzy(country_name)[0].alpha_3
        except LookupError:
            code = None
    _iso3_cache[country_name] = code
    return code


def fetch_adm0_exposure(row):
    csv_url = row["population_csv_url"]
    if not csv_url:
        return [], []

    try:
        resp = requests.get(csv_url, timeout=30)
        resp.raise_for_status()
        df_csv = pd.read_csv(pd.io.common.StringIO(resp.text))
    except Exception as e:
        print(f"  [{row['event_id']}] CSV fetch failed: {e}")
        return [], []

    df_csv.columns = df_csv.columns.str.strip().str.upper()

    pop_cols = ["POP_60_KMH", "POP_90_KMH", "POP_120_KMH"]
    kt_names = {
        "POP_60_KMH": "pop_34kt",
        "POP_90_KMH": "pop_50kt",
        "POP_120_KMH": "pop_64kt",
    }
    present = [c for c in pop_cols if c in df_csv.columns]
    if "ADM0_NAME" not in df_csv.columns or not present:
        return [], []

    meta = {
        "event_id": row["event_id"],
        "episode_id": row["episode_id"],
        "storm_name": row["storm_name"],
        "source": row["source"],
        "from_date": row["from_date"],
        "alert_level": row["alert_level"],
    }

    # ADM0
    adm0_out = []
    for _, r in (
        df_csv.groupby("ADM0_NAME", dropna=False)[present]
        .sum()
        .reset_index()
        .iterrows()
    ):
        adm0 = r["ADM0_NAME"]
        rec = {**meta, "iso3": name_to_iso3(str(adm0)), "country_name": adm0}
        for col in pop_cols:
            rec[kt_names[col]] = int(r[col]) if col in r and pd.notna(r[col]) else None
        adm0_out.append(rec)

    # ADM1
    adm1_out = []
    if "ADM1_NAME" in df_csv.columns:
        for _, r in (
            df_csv.groupby(["ADM0_NAME", "ADM1_NAME"], dropna=False)[present]
            .sum()
            .reset_index()
            .iterrows()
        ):
            adm0 = r["ADM0_NAME"]
            rec = {
                **meta,
                "iso3": name_to_iso3(str(adm0)),
                "country_name": adm0,
                "adm1_name": r["ADM1_NAME"],
            }
            for col in pop_cols:
                rec[kt_names[col]] = (
                    int(r[col]) if col in r and pd.notna(r[col]) else None
                )
            adm1_out.append(rec)

    return adm0_out, adm1_out


def build_exposure_df(df_latest):
    all_adm0 = []
    all_adm1 = []
    for _, ev in df_latest.iterrows():
        print(
            f"Fetching {ev['storm_name']} ({ev['event_id']}, ep {ev['episode_id']}) …",
            end=" ",
        )
        adm0_rows, adm1_rows = fetch_adm0_exposure(ev)
        all_adm0.extend(adm0_rows)
        all_adm1.extend(adm1_rows)
        print(f"{len(adm0_rows)} adm0 regions, {len(adm1_rows)} adm1 regions")

    def _to_df(rows, leading):
        df = (
            pd.DataFrame(rows)
            .sort_values(["from_date", "storm_name"], ascending=False)
            .reset_index(drop=True)
        )
        pop_cols = sorted([c for c in df.columns if c.startswith("pop_")])
        return df[leading + pop_cols]

    df_adm0 = _to_df(
        all_adm0,
        [
            "storm_name",
            "event_id",
            "episode_id",
            "source",
            "from_date",
            "alert_level",
            "iso3",
            "country_name",
        ],
    )
    df_adm1 = _to_df(
        all_adm1,
        [
            "storm_name",
            "event_id",
            "episode_id",
            "source",
            "from_date",
            "alert_level",
            "iso3",
            "country_name",
            "adm1_name",
        ],
    )
    return df_adm0, df_adm1


# ---------------------------------------------------------------------------
# 4. Make exposure values cumulative
# ---------------------------------------------------------------------------


def make_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    """Convert per-band exposure values to cumulative (≥ threshold).

    ADAM stores population counts per wind-speed band (e.g. pop_34kt is only
    the count *between* 34 kt and 50 kt). This converts to cumulative so that
    pop_34kt counts everyone exposed to *at least* 34 kt.
    """
    df = df.copy()
    v34 = df["pop_34kt"]
    v50 = df["pop_50kt"].fillna(0)
    v64 = df["pop_64kt"].fillna(0)
    df["pop_34kt"] = v34.where(v34.isna(), v34 + v50 + v64)
    df["pop_50kt"] = df["pop_50kt"].where(
        df["pop_50kt"].isna(), df["pop_50kt"].fillna(0) + v64
    )
    return df


# ---------------------------------------------------------------------------
# 5. Clean results and join with IBTrACS
# ---------------------------------------------------------------------------


def parse_name(input_name):
    season_str = input_name.split("-")[1]
    season = int("20" + season_str)
    storm_name = input_name.split("-")[0].split(" ")[-1]
    return season, storm_name


def join_ibtracs(df, df_ibtracs, geo_cols):
    df_sel = df.copy()
    df_sel[["season", "name"]] = df_sel["storm_name"].apply(
        lambda x: pd.Series(parse_name(x))
    )

    drop = list(geo_cols) + [c for c in df_sel.columns if c.startswith("pop_")]
    df_storms = (
        df_sel.drop_duplicates(subset=["storm_name"]).drop(columns=drop).reset_index()
    )
    print(f"Dataset has {len(df_storms)} unique storms.")

    df_merged = df_storms.merge(df_ibtracs, on=["season", "name"], how="left")
    df_merged = df_merged.sort_values("provisional").drop_duplicates(
        subset="storm_name", keep="first"
    )
    assert len(df_merged) == len(df_storms)

    df_final = df_sel.merge(df_merged)
    assert len(df_final) == len(df_sel)

    return df_final.drop(columns=["storm_name"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=== Fetching all TC event records ===")
    df_all = fetch_all_events()
    print(f"\nTotal records fetched: {len(df_all)}")

    print("\n=== Deduplicating to latest episode per event ===")
    df_latest = deduplicate(df_all)
    print(f"Unique events (latest episode only): {len(df_latest)}")

    print("\n=== Fetching population exposure per event ===")
    df_adm0, df_adm1 = build_exposure_df(df_latest)

    print("\n=== Making exposure values cumulative ===")
    df_adm0 = make_cumulative(df_adm0)
    df_adm1 = make_cumulative(df_adm1)

    print("\n=== Joining with IBTrACS ===")
    engine = stratus.get_engine("prod")
    with engine.connect() as conn:
        df_ibtracs = pd.read_sql("SELECT * FROM storms.ibtracs_storms", conn)

    df_adm0_final = join_ibtracs(df_adm0, df_ibtracs, geo_cols=["iso3", "country_name"])
    df_adm1_final = join_ibtracs(
        df_adm1, df_ibtracs, geo_cols=["iso3", "country_name", "adm1_name"]
    )

    print(f"\n=== Saving {len(df_adm0_final)} rows to {OUTPUT_ADM0_CSV} ===")
    stratus.upload_csv_to_blob(df_adm0_final, OUTPUT_ADM0_CSV)

    print(f"\n=== Saving {len(df_adm1_final)} rows to {OUTPUT_ADM1_CSV} ===")
    stratus.upload_csv_to_blob(df_adm1_final, OUTPUT_ADM1_CSV)

    print("Done.")


if __name__ == "__main__":
    main()
