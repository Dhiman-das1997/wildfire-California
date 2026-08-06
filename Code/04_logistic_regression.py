#!/usr/bin/env python3
"""
CODE 04: Leakage-free Logistic Regression
=========================================

This script must be run after Code 03.

Inputs
------
1. 02_causal_feature_table.csv
2. 03_outer_fold_registry.csv
3. 03_inner_fold_registry.csv

Main operations
---------------
1. Read the causal feature table and temporal-fold registries.
2. Use only outer-training rows for model development.
3. Use inner forward-chaining folds to select:
       - the Logistic Regression regularisation parameter C
       - the number of weather-derived predictors
4. Fit median imputation and standardisation on training rows only.
5. Fit a class-weighted Logistic Regression model.
6. Refit the selected configuration on the complete outer-training period.
7. Predict raw next-day ignition probabilities for the untouched outer-test year.
8. Save 2019-2023 out-of-fold predictions, selected features,
   hyperparameters, coefficients, and fold-level audit information.

Important
---------
This code saves RAW probabilities only. Probability calibration and warning-
threshold selection should be performed in a later code using inner out-of-fold
predictions, without accessing outer-test labels during model development.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. USER SETTINGS
# ============================================================

DATA_DIR: Path = Path(
    r"wildfire-California\Data"
)

OUTPUT_DIR: Path = DATA_DIR / "wildfire_outputs-final"

FEATURE_FILE: Path = (
    OUTPUT_DIR / "02_causal_feature_table.csv"
)

OUTER_REGISTRY_FILE: Path = (
    OUTPUT_DIR / "03_outer_fold_registry.csv"
)

INNER_REGISTRY_FILE: Path = (
    OUTPUT_DIR / "03_inner_fold_registry.csv"
)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

OOF_PREDICTION_FILE: Path = (
    OUTPUT_DIR / "04_logistic_oof_raw_predictions.csv"
)

SELECTED_FEATURE_FILE: Path = (
    OUTPUT_DIR / "04_logistic_selected_features.csv"
)

HYPERPARAMETER_FILE: Path = (
    OUTPUT_DIR / "04_logistic_hyperparameters.csv"
)

COEFFICIENT_FILE: Path = (
    OUTPUT_DIR / "04_logistic_coefficients.csv"
)

INNER_RESULT_FILE: Path = (
    OUTPUT_DIR / "04_logistic_inner_validation_results.csv"
)


INNER_OOF_PREDICTION_FILE: Path = (
    OUTPUT_DIR / "04_logistic_inner_oof_predictions.csv"
)

AUDIT_FILE: Path = (
    OUTPUT_DIR / "04_logistic_fold_audit.csv"
)


# ============================================================
# 3. MODEL SETTINGS
# ============================================================

RANDOM_SEED: int = 42

CORRELATION_CUTOFF: float = 0.90

AP_TIE_TOLERANCE: float = 0.005

C_VALUES: list[float] = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

WEATHER_SUBSET_SIZES: list[object] = [
    10,
    15,
    20,
    "all",
]

REFERENCE_COUNTY: str = "Butte"


# ============================================================
# 4. FEATURE POLICY
# ============================================================

STRUCTURAL_FEATURES: list[str] = [
    "doy_sin",
    "doy_cos",
]

NON_MODEL_COLUMNS: set[str] = {
    "county",
    "date",
    "target_date",
    "incident_count",
    "fire_ignition",
    "target_next_day",
    "t2m_mean_C",
    "t2m_min_C",
    "rh_mean",
    "rh_max",
    "wind_mean",
    "wind_min",
    "dry_day",
    "hot_day",
}

REQUIRED_COLUMNS: list[str] = [
    "county",
    "date",
    "target_date",
    "target_next_day",
    "doy_sin",
    "doy_cos",
]


# ============================================================
# 5. BASIC VALIDATION
# ============================================================

def check_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """Stop when a required column is missing."""

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{table_name} is missing columns:\n"
            f"{missing_columns}"
        )


def safe_average_precision(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> float:
    """Return AP only when both target classes are present."""

    if np.unique(y_true).size < 2:
        return np.nan

    return float(
        average_precision_score(
            y_true,
            probability,
        )
    )


# ============================================================
# 6. PREPARE COUNTY FIXED EFFECTS
# ============================================================

def add_county_fixed_effects(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create predetermined county dummy variables.

    Butte is used as the reference county, so its dummy column is omitted.
    """

    dataframe = dataframe.copy()

    counties = sorted(
        dataframe["county"].unique().tolist()
    )

    if REFERENCE_COUNTY not in counties:
        raise ValueError(
            f"Reference county {REFERENCE_COUNTY} was not found."
        )

    county_columns: list[str] = []

    for county in counties:

        if county == REFERENCE_COUNTY:
            continue

        column_name = f"county_{county}"

        dataframe[column_name] = (
            dataframe["county"] == county
        ).astype(int)

        county_columns.append(
            column_name
        )

    return dataframe, county_columns


# ============================================================
# 7. IDENTIFY WEATHER CANDIDATES
# ============================================================

def identify_weather_candidates(
    dataframe: pd.DataFrame,
    county_columns: list[str],
) -> list[str]:
    """
    Select compact weather-derived candidate predictors.

    Structural seasonality and county columns are always retained and are
    therefore not included in weather-feature screening.
    """

    excluded = (
        NON_MODEL_COLUMNS
        | set(STRUCTURAL_FEATURES)
        | set(county_columns)
    )

    candidates: list[str] = []

    for column in dataframe.columns:

        if column in excluded:
            continue

        if not pd.api.types.is_numeric_dtype(
            dataframe[column]
        ):
            continue

        if (
            column.startswith("t2m_max_C")
            or column.startswith("rh_min")
            or column.startswith("wind_max")
            or column.startswith("daily_VPD_proxy")
            or column in [
                "consecutive_dry_days",
                "consecutive_hot_days",
            ]
        ):
            candidates.append(column)

    if not candidates:
        raise ValueError(
            "No weather-derived predictors were identified."
        )

    return sorted(candidates)


# ============================================================
# 8. TRAINING-ONLY FEATURE SCREENING
# ============================================================

def feature_simplicity_key(
    feature: str,
) -> tuple:
    """
    Prefer simpler physical summaries when two candidates are effectively tied.
    """

    if feature in [
        "t2m_max_C",
        "rh_min",
        "wind_max",
        "daily_VPD_proxy",
    ]:
        return 0, feature

    if "_lag" in feature:
        return 1, feature

    if "mean" in feature or "max" in feature:
        return 2, feature

    if "consecutive" in feature:
        return 3, feature

    return 4, feature


def screen_weather_features(
    training_data: pd.DataFrame,
    candidate_features: list[str],
    subset_size: object,
) -> tuple[list[str], pd.DataFrame]:
    """
    Perform feature screening using training rows only.

    Steps
    -----
    1. Remove constant predictors.
    2. Median-impute using training data.
    3. Calculate mutual information on training data.
    4. Group highly correlated predictors.
    5. Keep one representative from each correlated group.
    6. Rank representatives by mutual information.
    7. Keep K predictors or all retained predictors.
    """

    X = training_data[
        candidate_features
    ].copy()

    y = training_data[
        "target_next_day"
    ].to_numpy(dtype=int)

    nonconstant_features = [
        column
        for column in candidate_features
        if X[column].nunique(dropna=True) > 1
    ]

    if not nonconstant_features:
        raise ValueError(
            "No nonconstant weather predictors survived screening."
        )

    imputer = SimpleImputer(
        strategy="median"
    )

    X_filled = pd.DataFrame(
        imputer.fit_transform(
            X[nonconstant_features]
        ),
        columns=nonconstant_features,
        index=X.index,
    )

    mutual_information_values = mutual_info_classif(
        X_filled,
        y,
        random_state=RANDOM_SEED,
    )

    mutual_information = dict(
        zip(
            nonconstant_features,
            mutual_information_values,
        )
    )

    correlation_matrix = (
        X_filled
        .corr(method="spearman")
        .abs()
    )

    unused_features = set(
        nonconstant_features
    )

    correlation_groups: list[list[str]] = []

    while unused_features:

        first_feature = sorted(
            unused_features
        )[0]

        current_group = [
            feature
            for feature in sorted(unused_features)
            if correlation_matrix.loc[
                first_feature,
                feature,
            ] > CORRELATION_CUTOFF
        ]

        correlation_groups.append(
            current_group
        )

        unused_features.difference_update(
            current_group
        )

    representatives: list[str] = []

    feature_group_lookup: dict[str, int] = {}

    for group_number, group in enumerate(
        correlation_groups,
        start=1,
    ):

        representative = sorted(
            group,
            key=lambda feature: (
                -mutual_information[feature],
                feature_simplicity_key(feature),
            ),
        )[0]

        representatives.append(
            representative
        )

        for feature in group:
            feature_group_lookup[feature] = group_number

    ranked_representatives = sorted(
        representatives,
        key=lambda feature: (
            -mutual_information[feature],
            feature_simplicity_key(feature),
        ),
    )

    if subset_size == "all":
        selected_features = ranked_representatives
    else:
        selected_features = ranked_representatives[
            : min(
                int(subset_size),
                len(ranked_representatives),
            )
        ]

    detail_rows: list[dict] = []

    for rank, feature in enumerate(
        selected_features,
        start=1,
    ):

        detail_rows.append(
            {
                "selected_feature": feature,
                "mutual_information": mutual_information[
                    feature
                ],
                "mi_rank": rank,
                "correlation_group": feature_group_lookup[
                    feature
                ],
                "subset_size_requested": subset_size,
            }
        )

    return selected_features, pd.DataFrame(detail_rows)


# ============================================================
# 9. FIT ONE LOGISTIC MODEL
# ============================================================

def fit_logistic_model(
    training_data: pd.DataFrame,
    predictor_columns: list[str],
    c_value: float,
) -> tuple[
    SimpleImputer,
    StandardScaler,
    LogisticRegression,
]:
    """
    Fit imputation, scaling, and Logistic Regression on training rows only.
    """

    X_train = training_data[
        predictor_columns
    ]

    y_train = training_data[
        "target_next_day"
    ].to_numpy(dtype=int)

    if np.unique(y_train).size < 2:
        raise ValueError(
            "Training data must contain both target classes."
        )

    imputer = SimpleImputer(
        strategy="median"
    )

    scaler = StandardScaler()

    X_train_imputed = imputer.fit_transform(
        X_train
    )

    X_train_scaled = scaler.fit_transform(
        X_train_imputed
    )

    model = LogisticRegression(
        C=c_value,
        penalty="l2",
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=RANDOM_SEED,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    return imputer, scaler, model


def predict_probability(
    dataframe: pd.DataFrame,
    predictor_columns: list[str],
    imputer: SimpleImputer,
    scaler: StandardScaler,
    model: LogisticRegression,
) -> np.ndarray:
    """Apply fitted preprocessing and return positive-class probabilities."""

    X = dataframe[
        predictor_columns
    ]

    X_imputed = imputer.transform(
        X
    )

    X_scaled = scaler.transform(
        X_imputed
    )

    return model.predict_proba(
        X_scaled
    )[:, 1]


# ============================================================
# 10. INNER TEMPORAL MODEL SELECTION
# ============================================================

def tune_logistic_regression(
    outer_training_data: pd.DataFrame,
    inner_registry_for_outer_fold: pd.DataFrame,
    weather_candidates: list[str],
    structural_columns: list[str],
) -> tuple[
    float,
    object,
    pd.DataFrame,
]:
    """
    Select C and weather subset size using mean inner-validation AP.

    Feature screening, imputation, scaling, and model fitting are repeated
    independently inside every inner fold.
    """

    result_rows: list[dict] = []

    for subset_size in WEATHER_SUBSET_SIZES:

        for c_value in C_VALUES:

            fold_scores: list[float] = []

            for inner_fold in (
                inner_registry_for_outer_fold
                .sort_values("inner_fold_id")
                .itertuples(index=False)
            ):

                inner_train_mask = (
                    outer_training_data["date"].between(
                        pd.Timestamp(
                            inner_fold.inner_train_issue_start
                        ),
                        pd.Timestamp(
                            inner_fold.inner_train_issue_end
                        ),
                    )
                )

                inner_validation_mask = (
                    outer_training_data["date"].between(
                        pd.Timestamp(
                            inner_fold.inner_validation_issue_start
                        ),
                        pd.Timestamp(
                            inner_fold.inner_validation_issue_end
                        ),
                    )
                )

                inner_train = (
                    outer_training_data
                    .loc[inner_train_mask]
                    .copy()
                )

                inner_validation = (
                    outer_training_data
                    .loc[inner_validation_mask]
                    .copy()
                )

                if (
                    len(inner_train) == 0
                    or len(inner_validation) == 0
                ):
                    continue

                if (
                    inner_train["target_next_day"].nunique() < 2
                    or inner_validation["target_next_day"].nunique() < 2
                ):
                    continue

                selected_weather, _ = screen_weather_features(
                    training_data=inner_train,
                    candidate_features=weather_candidates,
                    subset_size=subset_size,
                )

                predictors = (
                    selected_weather
                    + structural_columns
                )

                imputer, scaler, model = fit_logistic_model(
                    training_data=inner_train,
                    predictor_columns=predictors,
                    c_value=c_value,
                )

                validation_probability = predict_probability(
                    dataframe=inner_validation,
                    predictor_columns=predictors,
                    imputer=imputer,
                    scaler=scaler,
                    model=model,
                )

                validation_ap = safe_average_precision(
                    y_true=inner_validation[
                        "target_next_day"
                    ].to_numpy(dtype=int),
                    probability=validation_probability,
                )

                fold_scores.append(
                    validation_ap
                )

                result_rows.append(
                    {
                        "inner_fold_id": int(
                            inner_fold.inner_fold_id
                        ),
                        "validation_target_year": int(
                            inner_fold.validation_target_year
                        ),
                        "subset_size": subset_size,
                        "C": c_value,
                        "number_of_selected_weather_features": len(
                            selected_weather
                        ),
                        "validation_AP": validation_ap,
                    }
                )

            if not fold_scores:
                raise ValueError(
                    "No valid inner-validation scores were produced."
                )

    inner_results = pd.DataFrame(
        result_rows
    )

    summary = (
        inner_results
        .groupby(
            ["subset_size", "C"],
            as_index=False,
        )
        .agg(
            mean_inner_AP=(
                "validation_AP",
                "mean",
            ),
            standard_deviation_inner_AP=(
                "validation_AP",
                "std",
            ),
            number_of_inner_folds=(
                "validation_AP",
                "count",
            ),
        )
    )

    best_ap = summary[
        "mean_inner_AP"
    ].max()

    practically_tied = summary[
        summary["mean_inner_AP"]
        >= best_ap - AP_TIE_TOLERANCE
    ].copy()

    def subset_order(value: object) -> int:
        if str(value) == "all":
            return 10_000
        return int(value)

    practically_tied[
        "subset_order"
    ] = practically_tied[
        "subset_size"
    ].map(subset_order)

    # Prefer the smaller weather subset when AP is practically tied.
    # Then prefer stronger regularisation, represented here by smaller C.
    selected_row = (
        practically_tied
        .sort_values(
            [
                "subset_order",
                "C",
            ]
        )
        .iloc[0]
    )

    selected_c = float(
        selected_row["C"]
    )

    selected_subset_size: object = (
        selected_row["subset_size"]
    )

    if str(selected_subset_size) != "all":
        selected_subset_size = int(
            selected_subset_size
        )

    inner_results = inner_results.merge(
        summary,
        on=[
            "subset_size",
            "C",
        ],
        how="left",
    )

    return (
        selected_c,
        selected_subset_size,
        inner_results,
    )



# ============================================================
# 10B. GENERATE INNER OOF PREDICTIONS
# ============================================================

def generate_selected_inner_oof_predictions(
    outer_training_data: pd.DataFrame,
    inner_registry_for_outer_fold: pd.DataFrame,
    weather_candidates: list[str],
    structural_columns: list[str],
    selected_subset_size: object,
    selected_c: float,
    outer_fold_id: int,
    outer_target_year: int,
) -> pd.DataFrame:
    """Generate row-level inner-validation probabilities."""

    prediction_tables: list[pd.DataFrame] = []

    for inner_fold in (
        inner_registry_for_outer_fold
        .sort_values("inner_fold_id")
        .itertuples(index=False)
    ):
        inner_train = outer_training_data.loc[
            outer_training_data["date"].between(
                pd.Timestamp(inner_fold.inner_train_issue_start),
                pd.Timestamp(inner_fold.inner_train_issue_end),
            )
        ].copy()

        inner_validation = outer_training_data.loc[
            outer_training_data["date"].between(
                pd.Timestamp(inner_fold.inner_validation_issue_start),
                pd.Timestamp(inner_fold.inner_validation_issue_end),
            )
        ].copy()

        if len(inner_train) == 0 or len(inner_validation) == 0:
            continue

        if (
            inner_train["target_next_day"].nunique() < 2
            or inner_validation["target_next_day"].nunique() < 2
        ):
            continue

        selected_weather, _ = screen_weather_features(
            training_data=inner_train,
            candidate_features=weather_candidates,
            subset_size=selected_subset_size,
        )

        predictors = selected_weather + structural_columns

        imputer, scaler, model = fit_logistic_model(
            training_data=inner_train,
            predictor_columns=predictors,
            c_value=selected_c,
        )

        raw_probability = predict_probability(
            dataframe=inner_validation,
            predictor_columns=predictors,
            imputer=imputer,
            scaler=scaler,
            model=model,
        )

        fold_predictions = inner_validation[
            [
                "county",
                "date",
                "target_date",
                "target_next_day",
            ]
        ].copy()

        fold_predictions.insert(0, "model", "logistic_regression")
        fold_predictions.insert(1, "outer_fold_id", outer_fold_id)
        fold_predictions.insert(2, "outer_target_year", outer_target_year)
        fold_predictions.insert(
            3,
            "inner_fold_id",
            int(inner_fold.inner_fold_id),
        )
        fold_predictions.insert(
            4,
            "validation_target_year",
            int(inner_fold.validation_target_year),
        )
        fold_predictions["raw_probability"] = raw_probability

        prediction_tables.append(fold_predictions)

    if not prediction_tables:
        raise ValueError(
            f"No Logistic inner OOF predictions were generated for "
            f"outer fold {outer_fold_id}."
        )

    return pd.concat(prediction_tables, ignore_index=True)


# ============================================================
# 11. MAIN PROGRAM
# ============================================================

def main() -> None:
    """Train and evaluate Logistic Regression across five outer folds."""

    print("\n" + "=" * 75)
    print("CODE 04: LEAKAGE-FREE LOGISTIC REGRESSION")
    print("=" * 75)

    # --------------------------------------------------------
    # 11.1 Confirm that Code 02 and Code 03 outputs exist
    # --------------------------------------------------------

    required_files = [
        FEATURE_FILE,
        OUTER_REGISTRY_FILE,
        INNER_REGISTRY_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required input file was not found:\n"
                f"{file_path}"
            )

    # --------------------------------------------------------
    # 11.2 Load data
    # --------------------------------------------------------

    feature_data = pd.read_csv(
        FEATURE_FILE
    )

    outer_registry = pd.read_csv(
        OUTER_REGISTRY_FILE
    )

    inner_registry = pd.read_csv(
        INNER_REGISTRY_FILE
    )

    check_required_columns(
        dataframe=feature_data,
        required_columns=REQUIRED_COLUMNS,
        table_name="Feature table",
    )

    feature_data["date"] = pd.to_datetime(
        feature_data["date"],
        errors="raise",
    ).dt.normalize()

    feature_data["target_date"] = pd.to_datetime(
        feature_data["target_date"],
        errors="raise",
    ).dt.normalize()

    feature_data["target_next_day"] = (
        feature_data["target_next_day"]
        .astype(int)
    )

    for column in [
        "train_issue_start",
        "train_issue_end",
        "embargo_issue_start",
        "embargo_issue_end",
        "test_issue_start",
        "test_issue_end",
        "test_target_start",
        "test_target_end",
    ]:
        outer_registry[column] = pd.to_datetime(
            outer_registry[column],
            errors="raise",
        ).dt.normalize()

    inner_date_columns = [
        column
        for column in inner_registry.columns
        if (
            "start" in column
            or "end" in column
        )
    ]

    for column in inner_date_columns:
        inner_registry[column] = pd.to_datetime(
            inner_registry[column],
            errors="raise",
        ).dt.normalize()

    # --------------------------------------------------------
    # 11.3 Add predetermined county fixed effects
    # --------------------------------------------------------

    feature_data, county_columns = add_county_fixed_effects(
        feature_data
    )

    structural_columns = (
        STRUCTURAL_FEATURES
        + county_columns
    )

    weather_candidates = identify_weather_candidates(
        dataframe=feature_data,
        county_columns=county_columns,
    )

    print(
        f"\nWeather-derived candidates: "
        f"{len(weather_candidates)}"
    )

    print(
        f"Always-retained structural features: "
        f"{len(structural_columns)}"
    )

    # --------------------------------------------------------
    # 11.4 Storage for outputs
    # --------------------------------------------------------

    all_oof_predictions: list[pd.DataFrame] = []
    selected_feature_rows: list[dict] = []
    hyperparameter_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    all_inner_results: list[pd.DataFrame] = []
    all_inner_oof_predictions: list[pd.DataFrame] = []
    audit_rows: list[dict] = []

    # --------------------------------------------------------
    # 11.5 Process outer folds sequentially
    # --------------------------------------------------------

    for outer_fold in (
        outer_registry
        .sort_values("fold_id")
        .itertuples(index=False)
    ):

        fold_id = int(
            outer_fold.fold_id
        )

        target_year = int(
            outer_fold.target_year
        )

        print("\n" + "-" * 75)
        print(
            f"Outer fold {fold_id}: "
            f"target year {target_year}"
        )
        print("-" * 75)

        outer_train_mask = feature_data[
            "date"
        ].between(
            pd.Timestamp(
                outer_fold.train_issue_start
            ),
            pd.Timestamp(
                outer_fold.train_issue_end
            ),
        )

        outer_test_mask = feature_data[
            "date"
        ].between(
            pd.Timestamp(
                outer_fold.test_issue_start
            ),
            pd.Timestamp(
                outer_fold.test_issue_end
            ),
        )

        outer_train = (
            feature_data
            .loc[outer_train_mask]
            .copy()
        )

        outer_test = (
            feature_data
            .loc[outer_test_mask]
            .copy()
        )

        if (
            len(outer_train) == 0
            or len(outer_test) == 0
        ):
            raise ValueError(
                f"Outer fold {fold_id} has empty train or test data."
            )

        if outer_train["target_next_day"].nunique() < 2:
            raise ValueError(
                f"Outer fold {fold_id} training data "
                "does not contain both classes."
            )

        if outer_test["target_next_day"].nunique() < 2:
            raise ValueError(
                f"Outer fold {fold_id} test data "
                "does not contain both classes."
            )

        inner_for_outer_fold = inner_registry[
            inner_registry["outer_fold_id"]
            == fold_id
        ].copy()

        if inner_for_outer_fold.empty:
            raise ValueError(
                f"No inner folds were found for outer fold {fold_id}."
            )

        # ----------------------------------------------------
        # Tune Logistic Regression using inner folds only
        # ----------------------------------------------------

        (
            selected_c,
            selected_subset_size,
            inner_results,
        ) = tune_logistic_regression(
            outer_training_data=outer_train,
            inner_registry_for_outer_fold=inner_for_outer_fold,
            weather_candidates=weather_candidates,
            structural_columns=structural_columns,
        )

        inner_results.insert(
            0,
            "outer_fold_id",
            fold_id,
        )

        inner_results.insert(
            1,
            "outer_target_year",
            target_year,
        )

        all_inner_results.append(
            inner_results
        )


        selected_inner_oof = generate_selected_inner_oof_predictions(
            outer_training_data=outer_train,
            inner_registry_for_outer_fold=inner_for_outer_fold,
            weather_candidates=weather_candidates,
            structural_columns=structural_columns,
            selected_subset_size=selected_subset_size,
            selected_c=selected_c,
            outer_fold_id=fold_id,
            outer_target_year=target_year,
        )

        all_inner_oof_predictions.append(selected_inner_oof)

        # ----------------------------------------------------
        # Repeat feature screening on full outer training data
        # ----------------------------------------------------

        (
            selected_weather,
            feature_details,
        ) = screen_weather_features(
            training_data=outer_train,
            candidate_features=weather_candidates,
            subset_size=selected_subset_size,
        )

        final_predictors = (
            selected_weather
            + structural_columns
        )

        # ----------------------------------------------------
        # Fit final outer-fold model
        # ----------------------------------------------------

        (
            final_imputer,
            final_scaler,
            final_model,
        ) = fit_logistic_model(
            training_data=outer_train,
            predictor_columns=final_predictors,
            c_value=selected_c,
        )

        raw_test_probability = predict_probability(
            dataframe=outer_test,
            predictor_columns=final_predictors,
            imputer=final_imputer,
            scaler=final_scaler,
            model=final_model,
        )

        outer_test_ap = safe_average_precision(
            y_true=outer_test[
                "target_next_day"
            ].to_numpy(dtype=int),
            probability=raw_test_probability,
        )

        print(
            f"Selected C            : {selected_c}"
        )

        print(
            f"Selected subset size  : {selected_subset_size}"
        )

        print(
            f"Weather features used : {len(selected_weather)}"
        )

        print(
            f"Outer-test raw AP     : {outer_test_ap:.6f}"
        )

        # ----------------------------------------------------
        # Save untouched outer-test predictions
        # ----------------------------------------------------

        fold_predictions = outer_test[
            [
                "county",
                "date",
                "target_date",
                "target_next_day",
            ]
        ].copy()

        fold_predictions.insert(
            0,
            "model",
            "logistic_regression",
        )

        fold_predictions.insert(
            1,
            "outer_fold_id",
            fold_id,
        )

        fold_predictions.insert(
            2,
            "target_year",
            target_year,
        )

        fold_predictions["raw_probability"] = (
            raw_test_probability
        )

        all_oof_predictions.append(
            fold_predictions
        )

        # ----------------------------------------------------
        # Save selected feature information
        # ----------------------------------------------------

        for row in feature_details.to_dict(
            orient="records"
        ):

            row.update(
                {
                    "outer_fold_id": fold_id,
                    "target_year": target_year,
                    "feature_type": "weather_selected",
                }
            )

            selected_feature_rows.append(
                row
            )

        for feature in structural_columns:

            selected_feature_rows.append(
                {
                    "outer_fold_id": fold_id,
                    "target_year": target_year,
                    "selected_feature": feature,
                    "mutual_information": np.nan,
                    "mi_rank": np.nan,
                    "correlation_group": "always_retained",
                    "subset_size_requested": selected_subset_size,
                    "feature_type": "structural_always_retained",
                }
            )

        # ----------------------------------------------------
        # Save hyperparameters
        # ----------------------------------------------------

        hyperparameter_rows.append(
            {
                "outer_fold_id": fold_id,
                "target_year": target_year,
                "selected_C": selected_c,
                "selected_weather_subset_size": selected_subset_size,
                "number_of_selected_weather_features": len(
                    selected_weather
                ),
                "number_of_structural_features": len(
                    structural_columns
                ),
                "total_predictors": len(
                    final_predictors
                ),
                "class_weight": "balanced",
                "penalty": "l2",
                "solver": "liblinear",
                "outer_test_raw_AP": outer_test_ap,
            }
        )

        # ----------------------------------------------------
        # Save standardised coefficients
        # ----------------------------------------------------

        for predictor, coefficient in zip(
            final_predictors,
            final_model.coef_[0],
        ):

            coefficient_rows.append(
                {
                    "outer_fold_id": fold_id,
                    "target_year": target_year,
                    "predictor": predictor,
                    "standardized_coefficient": float(
                        coefficient
                    ),
                    "absolute_standardized_coefficient": abs(
                        float(coefficient)
                    ),
                }
            )

        coefficient_rows.append(
            {
                "outer_fold_id": fold_id,
                "target_year": target_year,
                "predictor": "intercept",
                "standardized_coefficient": float(
                    final_model.intercept_[0]
                ),
                "absolute_standardized_coefficient": abs(
                    float(final_model.intercept_[0])
                ),
            }
        )

        # ----------------------------------------------------
        # Fold audit
        # ----------------------------------------------------

        audit_rows.append(
            {
                "outer_fold_id": fold_id,
                "target_year": target_year,
                "training_rows": len(outer_train),
                "test_rows": len(outer_test),
                "training_positive_labels": int(
                    outer_train[
                        "target_next_day"
                    ].sum()
                ),
                "test_positive_labels": int(
                    outer_test[
                        "target_next_day"
                    ].sum()
                ),
                "training_start": outer_train[
                    "date"
                ].min(),
                "training_end": outer_train[
                    "date"
                ].max(),
                "test_issue_start": outer_test[
                    "date"
                ].min(),
                "test_issue_end": outer_test[
                    "date"
                ].max(),
                "test_target_start": outer_test[
                    "target_date"
                ].min(),
                "test_target_end": outer_test[
                    "target_date"
                ].max(),
                "selected_C": selected_c,
                "selected_subset_size": selected_subset_size,
                "outer_test_raw_AP": outer_test_ap,
            }
        )

    # --------------------------------------------------------
    # 11.6 Combine and save outputs
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    oof_predictions = pd.concat(
        all_oof_predictions,
        ignore_index=True,
    ).sort_values(
        [
            "target_date",
            "county",
            "outer_fold_id",
        ]
    )

    selected_features = pd.DataFrame(
        selected_feature_rows
    )

    hyperparameters = pd.DataFrame(
        hyperparameter_rows
    )

    coefficients = pd.DataFrame(
        coefficient_rows
    )

    inner_results_table = pd.concat(
        all_inner_results,
        ignore_index=True,
    )


    inner_oof_predictions = pd.concat(
        all_inner_oof_predictions,
        ignore_index=True,
    ).sort_values(
        [
            "outer_fold_id",
            "inner_fold_id",
            "target_date",
            "county",
        ]
    )

    audit_table = pd.DataFrame(
        audit_rows
    )

    oof_predictions.to_csv(
        OOF_PREDICTION_FILE,
        index=False,
    )

    selected_features.to_csv(
        SELECTED_FEATURE_FILE,
        index=False,
    )

    hyperparameters.to_csv(
        HYPERPARAMETER_FILE,
        index=False,
    )

    coefficients.to_csv(
        COEFFICIENT_FILE,
        index=False,
    )

    inner_results_table.to_csv(
        INNER_RESULT_FILE,
        index=False,
    )


    inner_oof_predictions.to_csv(
        INNER_OOF_PREDICTION_FILE,
        index=False,
    )

    audit_table.to_csv(
        AUDIT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # 11.7 Final summary
    # --------------------------------------------------------

    pooled_raw_ap = safe_average_precision(
        y_true=oof_predictions[
            "target_next_day"
        ].to_numpy(dtype=int),
        probability=oof_predictions[
            "raw_probability"
        ].to_numpy(dtype=float),
    )

    print("\n" + "=" * 75)
    print("CODE 04 COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print(
        f"Outer folds completed      : "
        f"{oof_predictions['outer_fold_id'].nunique()}"
    )

    print(
        f"OOF prediction rows        : "
        f"{len(oof_predictions)}"
    )

    print(
        f"OOF target years           : "
        f"{sorted(oof_predictions['target_year'].unique().tolist())}"
    )

    print(
        f"Pooled raw OOF AP          : "
        f"{pooled_raw_ap:.6f}"
    )

    print(
        f"\nRaw OOF predictions saved at:\n"
        f"{OOF_PREDICTION_FILE}"
    )

    print(
        f"\nSelected features saved at:\n"
        f"{SELECTED_FEATURE_FILE}"
    )

    print(
        f"\nHyperparameters saved at:\n"
        f"{HYPERPARAMETER_FILE}"
    )

    print(
        f"\nCoefficients saved at:\n"
        f"{COEFFICIENT_FILE}"
    )

    print(
        f"\nInner-validation results saved at:\n"
        f"{INNER_RESULT_FILE}"
    )

    print(
        f"\nFold audit saved at:\n"
        f"{AUDIT_FILE}"
    )


# ============================================================
# 12. RUN THE PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
