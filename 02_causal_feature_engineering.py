#!/usr/bin/env python3
"""
CODE 02: Causal feature engineering
===================================

This script reads the output produced by Code 01:

    01_combined_county_day_panel.csv

It then creates only causal predictors. Every feature associated with issue
date t is calculated using weather observations from date t or earlier.

The script creates:

1. daily_VPD_proxy
2. Lags 1-7 days
3. Rolling summaries over 3, 7, and 30 days
4. Consecutive dry-day and hot-day indicators
5. Annual seasonality variables doy_sin and doy_cos
6. A final feature table for target dates from 2014 through 2023

Output:

    02_causal_feature_table.csv
    02_feature_audit.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. USER SETTINGS
# ============================================================

DATA_DIR: Path = Path(
    r"C:\Users\Dhiman Das\Documents\10 JUNE Final paper"
)

OUTPUT_DIR: Path = DATA_DIR / "wildfire_outputs-final"

INPUT_FILE: Path = (
    OUTPUT_DIR / "01_combined_county_day_panel.csv"
)

OUTPUT_FILE: Path = (
    OUTPUT_DIR / "02_causal_feature_table.csv"
)

AUDIT_FILE: Path = (
    OUTPUT_DIR / "02_feature_audit.csv"
)


# ============================================================
# 2. FEATURE SETTINGS
# ============================================================

RAW_WEATHER_COLUMNS: list[str] = [
    "t2m_max_C",
    "rh_min",
    "wind_max",
]

LAG_DAYS: list[int] = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
]

ROLLING_WINDOWS: list[int] = [
    3,
    7,
    30,
]

DRY_DAY_RH_THRESHOLD: float = 20.0

HOT_DAY_TEMPERATURE_THRESHOLD: float = 35.0


# ============================================================
# 3. REQUIRED INPUT COLUMNS
# ============================================================

REQUIRED_COLUMNS: list[str] = [
    "county",
    "date",
    "target_date",
    "t2m_mean_C",
    "t2m_max_C",
    "t2m_min_C",
    "rh_mean",
    "rh_max",
    "rh_min",
    "wind_mean",
    "wind_max",
    "wind_min",
    "incident_count",
    "fire_ignition",
    "target_next_day",
]


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

def check_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """Stop when an expected input column is missing."""

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The Code 01 output is missing these columns:\n"
            f"{missing_columns}"
        )


# ============================================================
# 5. CHECK COUNTY-DATE ORDER AND CONTINUITY
# ============================================================

def check_county_dates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Check duplicate dates and missing daily dates separately
    for each county.
    """

    audit_rows: list[dict] = []

    for county, county_data in dataframe.groupby("county"):

        county_data = county_data.sort_values("date").copy()

        duplicate_dates = int(
            county_data["date"].duplicated().sum()
        )

        expected_dates = pd.date_range(
            start=county_data["date"].min(),
            end=county_data["date"].max(),
            freq="D",
        )

        available_dates = pd.DatetimeIndex(
            county_data["date"]
        )

        missing_dates = expected_dates.difference(
            available_dates
        )

        audit_rows.append(
            {
                "county": county,
                "first_issue_date": county_data["date"].min(),
                "last_issue_date": county_data["date"].max(),
                "input_rows": len(county_data),
                "duplicate_dates": duplicate_dates,
                "missing_dates": len(missing_dates),
            }
        )

        if duplicate_dates > 0:
            raise ValueError(
                f"{county} contains {duplicate_dates} "
                "duplicate issue dates."
            )

        if len(missing_dates) > 0:
            raise ValueError(
                f"{county} contains {len(missing_dates)} "
                "missing daily dates. Lag and rolling features "
                "would not represent exact calendar-day histories."
            )

    return pd.DataFrame(audit_rows)


# ============================================================
# 6. CALCULATE DAILY VPD PROXY
# ============================================================

def add_daily_vpd_proxy(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate a daily vapour-pressure-deficit proxy.

    Saturation vapour pressure:

        e_s(T) = 0.6108 * exp[17.27*T / (T + 237.3)]

    Daily VPD proxy:

        daily_VPD_proxy
        = e_s(t2m_max_C) * (1 - rh_min/100)

    This is called a proxy because daily maximum temperature and
    daily minimum relative humidity may not occur at the same hour.
    """

    dataframe = dataframe.copy()

    saturation_vapour_pressure = (
        0.6108
        * np.exp(
            17.27
            * dataframe["t2m_max_C"]
            / (dataframe["t2m_max_C"] + 237.3)
        )
    )

    dataframe["daily_VPD_proxy"] = (
        saturation_vapour_pressure
        * (1.0 - dataframe["rh_min"] / 100.0)
    )

    return dataframe


# ============================================================
# 7. CREATE LAGS 1-7
# ============================================================

def add_lag_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create lagged features separately within each county.

    Example:

        t2m_max_C_lag1 at date t
        = t2m_max_C observed at date t-1
    """

    dataframe = dataframe.copy()

    lag_columns = [
        "t2m_max_C",
        "rh_min",
        "wind_max",
        "daily_VPD_proxy",
    ]

    for column in lag_columns:

        for lag in LAG_DAYS:

            new_column = f"{column}_lag{lag}"

            dataframe[new_column] = (
                dataframe
                .groupby("county")[column]
                .shift(lag)
            )

    return dataframe


# ============================================================
# 8. CREATE ROLLING SUMMARIES
# ============================================================

def add_rolling_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create backward-looking rolling summaries.

    Every W-day feature at issue date t uses:

        [t-W+1, ..., t]

    Therefore, the rolling features are causal for predicting t+1.
    """

    dataframe = dataframe.copy()

    for window in ROLLING_WINDOWS:

        # Mean maximum temperature over the previous W days,
        # including the current issue day.
        dataframe[f"t2m_max_C_mean{window}d"] = (
            dataframe
            .groupby("county")["t2m_max_C"]
            .transform(
                lambda series: series.rolling(
                    window=window,
                    min_periods=window,
                ).mean()
            )
        )

        # Mean minimum relative humidity.
        dataframe[f"rh_min_mean{window}d"] = (
            dataframe
            .groupby("county")["rh_min"]
            .transform(
                lambda series: series.rolling(
                    window=window,
                    min_periods=window,
                ).mean()
            )
        )

        # Maximum wind speed during the W-day period.
        dataframe[f"wind_max_max{window}d"] = (
            dataframe
            .groupby("county")["wind_max"]
            .transform(
                lambda series: series.rolling(
                    window=window,
                    min_periods=window,
                ).max()
            )
        )

        # Mean daily VPD proxy.
        dataframe[f"daily_VPD_proxy_mean{window}d"] = (
            dataframe
            .groupby("county")["daily_VPD_proxy"]
            .transform(
                lambda series: series.rolling(
                    window=window,
                    min_periods=window,
                ).mean()
            )
        )

    return dataframe


# ============================================================
# 9. CONSECUTIVE-DAY FUNCTION
# ============================================================

def calculate_consecutive_days(
    binary_series: pd.Series,
) -> pd.Series:
    """
    Calculate the length of the current consecutive run of ones.

    Example:

        input : 0, 1, 1, 1, 0, 1
        output: 0, 1, 2, 3, 0, 1
    """

    groups = (
        binary_series
        != binary_series.shift()
    ).cumsum()

    consecutive_values = (
        binary_series
        .groupby(groups)
        .cumsum()
    )

    return consecutive_values.astype(int)


# ============================================================
# 10. CREATE HOT- AND DRY-DAY FEATURES
# ============================================================

def add_persistence_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create dry-day, hot-day, and consecutive-day indicators.

    Dry day:
        rh_min < 20 percent

    Hot day:
        t2m_max_C > 35 degrees Celsius
    """

    dataframe = dataframe.copy()

    dataframe["dry_day"] = (
        dataframe["rh_min"]
        < DRY_DAY_RH_THRESHOLD
    ).astype(int)

    dataframe["hot_day"] = (
        dataframe["t2m_max_C"]
        > HOT_DAY_TEMPERATURE_THRESHOLD
    ).astype(int)

    dataframe["consecutive_dry_days"] = (
        dataframe
        .groupby(
            "county",
            group_keys=False,
        )["dry_day"]
        .apply(calculate_consecutive_days)
    )

    dataframe["consecutive_hot_days"] = (
        dataframe
        .groupby(
            "county",
            group_keys=False,
        )["hot_day"]
        .apply(calculate_consecutive_days)
    )

    return dataframe


# ============================================================
# 11. CREATE SEASONALITY FEATURES
# ============================================================

def add_seasonality_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Encode day of year using sine and cosine functions.

    Two variables are required because the annual calendar is circular:
    31 December and 1 January should remain close to each other.
    """

    dataframe = dataframe.copy()

    day_of_year = (
        dataframe["date"]
        .dt.dayofyear
        .astype(float)
    )

    dataframe["doy_sin"] = np.sin(
        2.0
        * np.pi
        * day_of_year
        / 365.25
    )

    dataframe["doy_cos"] = np.cos(
        2.0
        * np.pi
        * day_of_year
        / 365.25
    )

    return dataframe


# ============================================================
# 12. CHECK FEATURE VALUES
# ============================================================

def check_feature_table(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Check for infinite and missing values in the final table."""

    numeric_values = dataframe[
        feature_columns
    ].to_numpy(dtype=float)

    number_of_infinite_values = int(
        np.isinf(numeric_values).sum()
    )

    number_of_missing_values = int(
        dataframe[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    if number_of_infinite_values > 0:
        raise ValueError(
            "The final feature table contains "
            f"{number_of_infinite_values} infinite values."
        )

    if number_of_missing_values > 0:
        raise ValueError(
            "The final feature table contains "
            f"{number_of_missing_values} missing feature values."
        )


# ============================================================
# 13. MAIN PROGRAM
# ============================================================

def main() -> None:
    """Create and save the causal feature table."""

    print("\n" + "=" * 70)
    print("CODE 02: CAUSAL FEATURE ENGINEERING")
    print("=" * 70)

    # --------------------------------------------------------
    # 13.1 Check and load Code 01 output
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "The Code 01 output was not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run Code 01 before running Code 02."
        )

    print(f"\nLoading:\n{INPUT_FILE}")

    data = pd.read_csv(
        INPUT_FILE
    )

    check_required_columns(
        dataframe=data,
        required_columns=REQUIRED_COLUMNS,
    )

    # --------------------------------------------------------
    # 13.2 Parse saved date columns again
    # --------------------------------------------------------

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    ).dt.normalize()

    data["target_date"] = pd.to_datetime(
        data["target_date"],
        errors="raise",
    ).dt.normalize()

    data = data.sort_values(
        ["county", "date"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 13.3 Audit date continuity
    # --------------------------------------------------------

    date_audit = check_county_dates(
        dataframe=data,
    )

    # --------------------------------------------------------
    # 13.4 Create features sequentially
    # --------------------------------------------------------

    print("\n1. Calculating daily_VPD_proxy...")

    data = add_daily_vpd_proxy(
        dataframe=data,
    )

    print("2. Creating lags 1-7...")

    data = add_lag_features(
        dataframe=data,
    )

    print("3. Creating 3-, 7-, and 30-day rolling summaries...")

    data = add_rolling_features(
        dataframe=data,
    )

    print("4. Creating consecutive hot and dry days...")

    data = add_persistence_features(
        dataframe=data,
    )

    print("5. Creating doy_sin and doy_cos...")

    data = add_seasonality_features(
        dataframe=data,
    )

    # --------------------------------------------------------
    # 13.5 Define the generated feature columns
    # --------------------------------------------------------

    generated_feature_columns = [
        "daily_VPD_proxy",
    ]

    for column in [
        "t2m_max_C",
        "rh_min",
        "wind_max",
        "daily_VPD_proxy",
    ]:

        for lag in LAG_DAYS:

            generated_feature_columns.append(
                f"{column}_lag{lag}"
            )

    for window in ROLLING_WINDOWS:

        generated_feature_columns.extend(
            [
                f"t2m_max_C_mean{window}d",
                f"rh_min_mean{window}d",
                f"wind_max_max{window}d",
                f"daily_VPD_proxy_mean{window}d",
            ]
        )

    generated_feature_columns.extend(
        [
            "dry_day",
            "hot_day",
            "consecutive_dry_days",
            "consecutive_hot_days",
            "doy_sin",
            "doy_cos",
        ]
    )

    # --------------------------------------------------------
    # 13.6 Keep labelled target dates from 2014-2023
    # --------------------------------------------------------

    # Data from 2013 were used as weather-history warm-up.
    # The modelling period begins with target date 2014-01-01.
    feature_table = data[
        data["target_date"].between(
            "2014-01-01",
            "2023-12-31",
        )
    ].copy()

    # All required 30-day histories should be available because
    # 2013 was retained as a warm-up year.
    feature_table = feature_table.dropna(
        subset=generated_feature_columns
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 13.7 Validate causal feature table
    # --------------------------------------------------------

    check_feature_table(
        dataframe=feature_table,
        feature_columns=generated_feature_columns,
    )

    duplicate_county_dates = int(
        feature_table.duplicated(
            subset=["county", "date"]
        ).sum()
    )

    if duplicate_county_dates > 0:

        raise ValueError(
            "Duplicate county-date rows were found "
            "in the final feature table."
        )

    # --------------------------------------------------------
    # 13.8 Arrange output columns
    # --------------------------------------------------------

    identification_columns = [
        "county",
        "date",
        "target_date",
    ]

    original_weather_columns = [
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

    label_columns = [
        "incident_count",
        "fire_ignition",
        "target_next_day",
    ]

    output_columns = (
        identification_columns
        + original_weather_columns
        + generated_feature_columns
        + label_columns
    )

    feature_table = feature_table[
        output_columns
    ]

    # --------------------------------------------------------
    # 13.9 Add feature audit information
    # --------------------------------------------------------

    date_audit["output_rows"] = (
        date_audit["county"]
        .map(
            feature_table["county"]
            .value_counts()
        )
        .fillna(0)
        .astype(int)
    )

    date_audit["generated_feature_count"] = (
        len(generated_feature_columns)
    )

    date_audit["positive_next_day_targets"] = (
        date_audit["county"]
        .map(
            feature_table
            .groupby("county")["target_next_day"]
            .sum()
        )
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # 13.10 Save outputs
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_table.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    date_audit.to_csv(
        AUDIT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # 13.11 Print final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CODE 02 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Counties                  : "
        f"{feature_table['county'].nunique()}"
    )

    print(
        f"Feature-table rows        : "
        f"{len(feature_table)}"
    )

    print(
        f"Generated causal features : "
        f"{len(generated_feature_columns)}"
    )

    print(
        f"Positive next-day targets : "
        f"{feature_table['target_next_day'].sum()}"
    )

    print(
        f"\nFeature table saved at:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nFeature audit saved at:\n"
        f"{AUDIT_FILE}"
    )


# ============================================================
# 14. RUN THE PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
