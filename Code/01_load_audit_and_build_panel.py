#!/usr/bin/env python3
"""
STEP 1: Load, check, and combine the 10-county wildfire datasets
================================================================

This beginner-friendly script performs only the first processing stage:

1. Defines the ten county names.
2. Builds the weather and wildfire CSV paths automatically.
3. Loads each county's two CSV files.
4. Converts the date columns to daily dates.
5. Checks duplicate weather dates and missing weather values.
6. Converts wildfire incidents into one binary ignition label per county-day.
7. Merges weather and wildfire information.
8. Creates the next-day prediction target.
9. Combines all ten counties into one panel dataset.
10. Saves the processed panel and a simple audit table.

No feature engineering, temporal splitting, scaling, feature selection,
or machine-learning model is performed in this script.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# 1. USER SETTINGS
# ============================================================

DATA_DIR: Path = Path(
    r"C:\Users\Dhiman Das\Documents\10 JUNE Final paper"
)

OUTPUT_DIR: Path = DATA_DIR / "wildfire_outputs-final"

COUNTIES: list[str] = [
    "Butte",
    "Fresno",
    "Kern",
    "Los_Angeles",
    "Riverside",
    "San_Bernardino",
    "San_Diego",
    "San_Luis_Obispo",
    "Shasta",
    "Siskiyou",
]

WEATHER_COLUMNS: list[str] = [
    "time",
    "t2m_mean_C",
    "t2m_max_C",
    "t2m_min_C",
    "rh_mean",
    "rh_max",
    "rh_min",
    "wind_mean",
    "wind_max",
    "wind_min",
]

FIRE_COLUMNS: list[str] = [
    "Incident",
    "Counties",
    "Started",
    "Acres",
    "Containment",
]


# ============================================================
# 2. CREATE FILE PATHS
# ============================================================

def get_county_file_paths(county: str) -> tuple[Path, Path]:
    """
    Create the two input file paths for one county.

    Example for San Diego:
    San_Diego_final_stats_2013_2023.csv
    San_Diego_Wildfires_2013_2023.csv
    """

    temp_rh_wind_csv: Path = DATA_DIR / (
        f"{county}_final_stats_2013_2023.csv"
    )

    fire_csv: Path = DATA_DIR / (
        f"{county}_Wildfires_2013_2023.csv"
    )

    return temp_rh_wind_csv, fire_csv


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

def check_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    file_name: str,
) -> None:
    """Stop the program when an expected column is missing."""

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {file_name}: {missing_columns}"
        )


# ============================================================
# 4. LOAD AND PROCESS ONE COUNTY
# ============================================================

def process_one_county(
    county: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Load weather and wildfire files for one county and create
    a daily county-level panel.
    """

    temp_rh_wind_csv, fire_csv = get_county_file_paths(county)

    if not temp_rh_wind_csv.exists():
        raise FileNotFoundError(
            f"Weather file not found:\n{temp_rh_wind_csv}"
        )

    if not fire_csv.exists():
        raise FileNotFoundError(
            f"Wildfire file not found:\n{fire_csv}"
        )

    print(f"\nProcessing county: {county}")
    print(f"Weather file : {temp_rh_wind_csv.name}")
    print(f"Wildfire file: {fire_csv.name}")

    # --------------------------------------------------------
    # Load CSV files
    # --------------------------------------------------------

    weather = pd.read_csv(temp_rh_wind_csv)
    fire = pd.read_csv(fire_csv)

    check_required_columns(
        dataframe=weather,
        required_columns=WEATHER_COLUMNS,
        file_name=temp_rh_wind_csv.name,
    )

    check_required_columns(
        dataframe=fire,
        required_columns=FIRE_COLUMNS,
        file_name=fire_csv.name,
    )

    # Keep only the columns required by the algorithm.
    weather = weather[WEATHER_COLUMNS].copy()
    fire = fire[FIRE_COLUMNS].copy()

    # --------------------------------------------------------
    # Process weather dates
    # --------------------------------------------------------

    weather["date"] = pd.to_datetime(
        weather["time"],
        errors="raise",
    ).dt.normalize()

    weather["county"] = county

    weather = weather.sort_values("date").reset_index(drop=True)

    duplicate_weather_dates = int(
        weather["date"].duplicated().sum()
    )

    weather_missing_values = int(
        weather[WEATHER_COLUMNS[1:]].isna().sum().sum()
    )

    # Require one weather row per date.
    if duplicate_weather_dates > 0:
        raise ValueError(
            f"{county}: duplicate weather dates found = "
            f"{duplicate_weather_dates}"
        )

    # --------------------------------------------------------
    # Process wildfire ignition dates
    # --------------------------------------------------------

    fire["date"] = pd.to_datetime(
        fire["Started"],
        errors="raise",
    ).dt.normalize()

    fire["county"] = county

    # Multiple incidents on the same date are collapsed into
    # one positive county-day. incident_count is retained.
    daily_fire = (
        fire.groupby(["county", "date"])
        .size()
        .reset_index(name="incident_count")
    )

    daily_fire["fire_ignition"] = 1

    # --------------------------------------------------------
    # Merge weather and wildfire information
    # --------------------------------------------------------

    county_panel = weather.merge(
        daily_fire,
        on=["county", "date"],
        how="left",
    )

    county_panel["incident_count"] = (
        county_panel["incident_count"]
        .fillna(0)
        .astype(int)
    )

    county_panel["fire_ignition"] = (
        county_panel["fire_ignition"]
        .fillna(0)
        .astype(int)
    )

    county_panel = county_panel.sort_values(
        ["county", "date"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Create next-day target
    # --------------------------------------------------------

    # The row dated t contains weather observed through day t.
    # The target indicates whether ignition occurs on day t+1.
    county_panel["target_next_day"] = (
        county_panel["fire_ignition"].shift(-1)
    )

    county_panel["target_date"] = (
        county_panel["date"] + pd.Timedelta(days=1)
    )

    # The final row has no following-day label.
    county_panel = county_panel.dropna(
        subset=["target_next_day"]
    ).copy()

    county_panel["target_next_day"] = (
        county_panel["target_next_day"].astype(int)
    )

    # --------------------------------------------------------
    # Prepare audit information
    # --------------------------------------------------------

    model_period = county_panel[
        county_panel["target_date"].between(
            "2014-01-01",
            "2023-12-31",
        )
    ]

    audit = {
        "county": county,
        "weather_file": temp_rh_wind_csv.name,
        "fire_file": fire_csv.name,
        "weather_start": weather["date"].min(),
        "weather_end": weather["date"].max(),
        "weather_rows": len(weather),
        "duplicate_weather_dates": duplicate_weather_dates,
        "weather_missing_values": weather_missing_values,
        "incident_records": len(fire),
        "positive_ignition_days_2014_2023": int(
            model_period["target_next_day"].sum()
        ),
        "processed_rows": len(county_panel),
    }

    return county_panel, audit


# ============================================================
# 5. PROCESS ALL TEN COUNTIES
# ============================================================

def main() -> None:
    """Run Step 1 for all ten counties."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    county_panels: list[pd.DataFrame] = []
    audit_rows: list[dict] = []

    for county in COUNTIES:
        county_panel, audit = process_one_county(county)

        county_panels.append(county_panel)
        audit_rows.append(audit)

    # --------------------------------------------------------
    # Combine all counties
    # --------------------------------------------------------

    full_panel = pd.concat(
        county_panels,
        ignore_index=True,
    )

    full_panel = full_panel.sort_values(
        ["date", "county"]
    ).reset_index(drop=True)

    audit_table = pd.DataFrame(audit_rows)

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    panel_output: Path = (
        OUTPUT_DIR / "01_combined_county_day_panel.csv"
    )

    audit_output: Path = (
        OUTPUT_DIR / "01_data_audit.csv"
    )

    full_panel.to_csv(
        panel_output,
        index=False,
    )

    audit_table.to_csv(
        audit_output,
        index=False,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETED")
    print("=" * 60)
    print(f"Number of counties       : {full_panel['county'].nunique()}")
    print(f"Total processed rows     : {len(full_panel)}")
    print(f"Total next-day positives : {full_panel['target_next_day'].sum()}")
    print(f"Panel saved to           : {panel_output}")
    print(f"Audit saved to           : {audit_output}")


if __name__ == "__main__":
    main()
