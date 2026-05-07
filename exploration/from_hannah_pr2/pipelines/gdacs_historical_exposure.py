# ===========================================================================
# Ported verbatim — DO NOT MODIFY (this is the refactor source-of-truth)
# Source repo:   OCHA-DAP/ds-storm-impact-harmonisation
# Source ref:    e692da12b17b29926a5d704df94fc31f2d7541fa (origin/gdacs-adam-data)
# Source path:   pipelines/gdacs_historical_exposure.py
# Pulled:        2026-05-07
# Provenance:    https://github.com/OCHA-DAP/ds-storm-impact-harmonisation/pull/2
# Refactored functions live in src/ocha_lens/datasources/ — see PR description.
# ===========================================================================

"""
GDACS — Historical ADM0 Exposure

Retrieves ADM0-level population exposure for all tropical cyclones within a
date range, using the GDACS search endpoint for full historical coverage. Saves
ADM0 and ADM1 level exposure estimates to blob storage for all available storms.

API path:
    geteventlist/search  (paginated, filterable by date / source)
      -> getepisodedata  (latest episode per event)
        -> getimpact -> datums[alias='country'] -> ISO_3DIGIT, CNTRY_NAME, POP_AFFECTED
                     -> datums[alias='alert']   -> FIPS_ADMIN, GMI_ADMIN, ADMIN_NAME,
                                                   POP_AFFECTED
"""

import sys
from pathlib import Path

import ocha_stratus as stratus
import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from constants import PROJECT_PREFIX

load_dotenv()

GDACS_BASE = "https://www.gdacs.org/gdacsapi/api"
FROM_DATE = "2010-01-01"
TO_DATE = "2026-12-31"
SOURCE = "NOAA"  # 'NOAA' = Atlantic/E.Pacific | 'JTWC' = W.Pacific/Indian Ocean
OUTPUT_ADM0_CSV = f"{PROJECT_PREFIX}/processed/gdacs_historical_adm0_exposure.csv"
OUTPUT_ADM1_CSV = f"{PROJECT_PREFIX}/processed/gdacs_historical_adm1_exposure.csv"

# Wind speed (kt) implied by each buffer key
BUFFER_KT = {
    "buffer39": 34,
    "buffer74": 64,
}


# ---------------------------------------------------------------------------
# 1. Fetch all TC events in the date range
# ---------------------------------------------------------------------------


def fetch_all_events():
    all_events = []
    alert_levels = ["Red", "Orange", "Green"]

    for alert_level in alert_levels:
        print(f"\nSearching for {alert_level} alert storms...")
        page = 1

        while True:
            params = {
                "eventlist": "TC",
                "fromDate": FROM_DATE,
                "toDate": TO_DATE,
                "alertlevel": alert_level,
                "pageSize": 100,
                "pageNumber": page,
            }
            resp = requests.get(
                f"{GDACS_BASE}/events/geteventlist/search",
                params=params,
                timeout=30,
            )
            if not resp.text.strip():
                break
            features = resp.json().get("features", [])
            if not features:
                break

            for f in features:
                p = f["properties"]
                if SOURCE and p.get("source") != SOURCE:
                    continue
                event_id = str(p["eventid"])
                if any(e["event_id"] == event_id for e in all_events):
                    continue
                all_events.append(
                    {
                        "event_id": event_id,
                        "storm_name": p.get("name", ""),
                        "source": p.get("source", ""),
                        "from_date": p.get("fromdate", ""),
                        "to_date": p.get("todate", ""),
                        "alert_level": p.get("alertlevel", ""),
                    }
                )

            print(f"  Page {page}: {len(features)} events fetched")
            page += 1

    df_events = pd.DataFrame(all_events)
    print(f"\nTotal TCs found ({SOURCE or 'all basins'}): {len(df_events)}")
    return df_events


# ---------------------------------------------------------------------------
# 2. Fetch exposure data per event
# ---------------------------------------------------------------------------


def fetch_adm0_exposure(event_id):
    """Return ADM0 and ADM1 row dicts for the latest episode. Empty lists if no data."""
    try:
        props = requests.get(
            f"{GDACS_BASE}/events/getepisodedata",
            params={"eventtype": "TC", "eventid": event_id},
            timeout=30,
        ).json()["properties"]
    except Exception as e:
        print(f"  [{event_id}] failed: {e}")
        return [], []

    last_ep_url = props.get("episodes", [{}])[-1].get("details", "")
    episode_id = (
        last_ep_url.split("episodeid=")[-1].split("&")[0] if last_ep_url else "?"
    )
    buffers = {
        k: v
        for k, v in props.get("impacts", [{}])[0].get("resource", {}).items()
        if k.startswith("buffer")
    }

    adm0_data = {}
    adm1_data = {}
    for buf, url in buffers.items():
        col = f"pop_{BUFFER_KT.get(buf, buf)}kt"
        try:
            datums = requests.get(url, timeout=30).json().get("datums", [])
        except Exception:
            continue

        adm0_datum = next((d for d in datums if d["alias"] == "country"), None)
        if adm0_datum:
            for row in adm0_datum.get("datum", []):
                sc = {s["name"]: s["value"] for s in row["scalars"]["scalar"]}
                iso3 = sc.get("ISO_3DIGIT")
                if not iso3:
                    continue
                pop_affected = sc.get("POP_AFFECTED")
                if pop_affected is not None and pop_affected != "":
                    try:
                        pop_value = int(float(pop_affected))
                    except (ValueError, TypeError):
                        pop_value = None
                else:
                    pop_value = None
                adm0_data.setdefault(
                    iso3,
                    {
                        "event_id": event_id,
                        "episode_id": episode_id,
                        "iso3": iso3,
                        "country_name": sc.get("CNTRY_NAME"),
                    },
                )[col] = pop_value

        alert_datum = next((d for d in datums if d["alias"] == "alert"), None)
        if alert_datum:
            for row in alert_datum.get("datum", []):
                sc = {s["name"]: s["value"] for s in row["scalars"]["scalar"]}
                fips_admin = sc.get("FIPS_ADMIN")
                if not fips_admin:
                    continue
                gmi_admin = sc.get("GMI_ADMIN", "")
                iso3 = gmi_admin.split("-")[0] if gmi_admin else None
                pop_affected = sc.get("POP_AFFECTED")
                if pop_affected is not None and pop_affected != "":
                    try:
                        pop_value = int(float(pop_affected))
                    except (ValueError, TypeError):
                        pop_value = None
                else:
                    pop_value = None
                adm1_data.setdefault(
                    fips_admin,
                    {
                        "event_id": event_id,
                        "episode_id": episode_id,
                        "iso3": iso3,
                        "fips_admin": fips_admin,
                        "gmi_admin": gmi_admin,
                        "adm1_name": sc.get("ADMIN_NAME"),
                        "adm1_type": sc.get("TYPE_ENG"),
                    },
                )[col] = pop_value

    return list(adm0_data.values()), list(adm1_data.values())


def build_exposure_df(df_events):
    all_adm0 = []
    all_adm1 = []
    for _, ev in df_events.iterrows():
        eid = ev["event_id"]
        name = ev["storm_name"]
        print(f"Fetching {name} ({eid}) …", end=" ")
        adm0_rows, adm1_rows = fetch_adm0_exposure(eid)
        meta = {
            "storm_name": name,
            "source": ev["source"],
            "from_date": ev["from_date"],
            "alert_level": ev["alert_level"],
        }
        for r in adm0_rows:
            r.update(meta)
        for r in adm1_rows:
            r.update(meta)
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
            "fips_admin",
            "gmi_admin",
            "adm1_name",
            "adm1_type",
        ],
    )
    return df_adm0, df_adm1


# ---------------------------------------------------------------------------
# 3. Clean results and join with IBTrACS
# ---------------------------------------------------------------------------


def parse_name(input_name):
    season_str = input_name.split("-")[-1]
    season = int("20" + season_str)
    storm_name = input_name.split("-")[0].split(" ")[-1]
    return season, storm_name


def join_ibtracs(df, df_ibtracs, geo_cols):
    df_cleaned = df.copy()
    df_cleaned[["season", "name"]] = df_cleaned["storm_name"].apply(
        lambda x: pd.Series(parse_name(x))
    )

    drop = list(geo_cols) + [c for c in df_cleaned.columns if c.startswith("pop_")]
    df_storms = (
        df_cleaned.drop_duplicates(subset=["storm_name"])
        .drop(columns=drop)
        .reset_index()
    )
    print(f"Dataset has {len(df_storms)} unique storms.")

    df_merged = df_storms.merge(df_ibtracs, on=["season", "name"], how="left")
    df_merged = df_merged.sort_values("provisional").drop_duplicates(
        subset="storm_name", keep="first"
    )
    assert len(df_merged) == len(df_storms)

    df_final = df_cleaned.merge(df_merged)
    assert len(df_final) == len(df_cleaned)

    return df_final.drop(columns=["storm_name"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=== Fetching all TC events ===")
    df_events = fetch_all_events()

    print("\n=== Fetching population exposure per event ===")
    df_adm0, df_adm1 = build_exposure_df(df_events)

    print("\n=== Joining with IBTrACS ===")
    engine = stratus.get_engine("prod")
    with engine.connect() as conn:
        df_ibtracs = pd.read_sql("SELECT * FROM storms.ibtracs_storms", conn)

    df_adm0_final = join_ibtracs(df_adm0, df_ibtracs, geo_cols=["iso3", "country_name"])
    df_adm1_final = join_ibtracs(
        df_adm1,
        df_ibtracs,
        geo_cols=["iso3", "fips_admin", "gmi_admin", "adm1_name", "adm1_type"],
    )

    print(f"\n=== Saving {len(df_adm0_final)} rows to {OUTPUT_ADM0_CSV} ===")
    stratus.upload_csv_to_blob(df_adm0_final, OUTPUT_ADM0_CSV)

    print(f"\n=== Saving {len(df_adm1_final)} rows to {OUTPUT_ADM1_CSV} ===")
    stratus.upload_csv_to_blob(df_adm1_final, OUTPUT_ADM1_CSV)

    print("Done.")


if __name__ == "__main__":
    main()
