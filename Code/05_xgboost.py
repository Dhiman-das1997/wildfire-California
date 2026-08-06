#!/usr/bin/env python3
"""
CODE 05: Leakage-free XGBoost
=============================

Run this script after Codes 01-04.

Inputs
------
1. 02_causal_feature_table.csv
2. 03_outer_fold_registry.csv
3. 03_inner_fold_registry.csv

Important
---------
XGBoost does not use the output probabilities from Code 04. Logistic
Regression and XGBoost are trained independently using the same causal feature
table and the same temporal folds.

This script:

1. Loads the causal feature table and temporal fold registries.
2. Uses only outer-training rows for model development.
3. Uses inner forward-chaining folds to select XGBoost hyperparameters.
4. Performs weather-feature screening inside training data only.
5. Uses class imbalance weighting through scale_pos_weight.
6. Trains one final XGBoost model for each outer fold.
7. Predicts raw probabilities for untouched target years 2019-2023.
8. Saves OOF predictions, selected features, hyperparameters,
   feature importance, inner-validation results, and audit information.

Raw probabilities are saved here. Calibration and final warning-threshold
selection should be performed in a later code.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score

try:
    from xgboost import XGBClassifier
except ImportError as error:
    raise ImportError(
        "XGBoost is not installed.\n"
        "Install it using:\n"
        "    pip install xgboost"
    ) from error

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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
    OUTPUT_DIR / "05_xgboost_oof_raw_predictions.csv"
)

SELECTED_FEATURE_FILE: Path = (
    OUTPUT_DIR / "05_xgboost_selected_features.csv"
)

HYPERPARAMETER_FILE: Path = (
    OUTPUT_DIR / "05_xgboost_hyperparameters.csv"
)

FEATURE_IMPORTANCE_FILE: Path = (
    OUTPUT_DIR / "05_xgboost_feature_importance.csv"
)

INNER_RESULT_FILE: Path = (
    OUTPUT_DIR / "05_xgboost_inner_validation_results.csv"
)


INNER_OOF_PREDICTION_FILE: Path = (
    OUTPUT_DIR / "05_xgboost_inner_oof_predictions.csv"
)

AUDIT_FILE: Path = (
    OUTPUT_DIR / "05_xgboost_fold_audit.csv"
)


# ============================================================
# 3. MODEL SETTINGS
# ============================================================

RANDOM_SEED: int = 42

CORRELATION_CUTOFF: float = 0.90

AP_TIE_TOLERANCE: float = 0.005

REFERENCE_COUNTY: str = "Butte"

WEATHER_SUBSET_SIZES: list[object] = [
    10,
    15,
    20,
    "all",
]

# A compact beginner-friendly hyperparameter grid.
# A larger grid can be used later for final manuscript analysis.
XGBOOST_PARAMETER_GRID: list[dict] = [
    {
        "max_depth": 2,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_weight": 1,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_weight": 1,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "max_depth": 3,
        "learning_rate": 0.03,
        "n_estimators": 500,
        "min_child_weight": 3,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.10,
        "reg_lambda": 2.0,
    },
]


# ============================================================
# 4. FEATURE POLICY
# ============================================================

STRUCTURAL_FEATURES: list[str] = [
    "doy_sin",
    "doy_cos",
]

REQUIRED_COLUMNS: list[str] = [
    "county",
    "date",
    "target_date",
    "target_next_day",
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


# ============================================================
# 5. BASIC CHECKS
# ============================================================

def check_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """Stop when required columns are absent."""

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
    """Calculate AP only when both classes are present."""

    if np.unique(y_true).size < 2:
        return np.nan

    return float(
        average_precision_score(
            y_true,
            probability,
        )
    )


def calculate_scale_pos_weight(
    labels: np.ndarray,
) -> float:
    """
    Calculate class imbalance weight:

        scale_pos_weight = number of negatives / number of positives
    """

    labels = np.asarray(
        labels,
        dtype=int,
    )

    number_of_positive_labels = int(
        np.sum(labels == 1)
    )

    number_of_negative_labels = int(
        np.sum(labels == 0)
    )

    if number_of_positive_labels == 0:
        raise ValueError(
            "Training data contains no positive labels."
        )

    return (
        number_of_negative_labels
        / number_of_positive_labels
    )


# ============================================================
# 6. COUNTY FIXED EFFECTS
# ============================================================

def add_county_fixed_effects(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Add one-hot county indicators.

    Butte is used as the reference county and is not given a dummy column.
    """

    dataframe = dataframe.copy()

    counties = sorted(
        dataframe["county"]
        .dropna()
        .unique()
        .tolist()
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
# 7. IDENTIFY XGBOOST WEATHER CANDIDATES
# ============================================================

def identify_xgboost_weather_candidates(
    dataframe: pd.DataFrame,
    county_columns: list[str],
) -> list[str]:
    """
    Identify weather-derived XGBoost candidates.

    Primary XGBoost predictors use raw physical weather families:
        temperature, humidity, wind, and persistence.

    daily_VPD_proxy is excluded from the primary model because it is a
    deterministic combination of temperature and humidity. It can be tested
    later as a separate sensitivity experiment.
    """

    excluded_columns = (
        NON_MODEL_COLUMNS
        | set(STRUCTURAL_FEATURES)
        | set(county_columns)
    )

    candidate_features: list[str] = []

    for column in dataframe.columns:

        if column in excluded_columns:
            continue

        if not pd.api.types.is_numeric_dtype(
            dataframe[column]
        ):
            continue

        if column.startswith("daily_VPD_proxy"):
            continue

        if (
            column.startswith("t2m_max_C")
            or column.startswith("rh_min")
            or column.startswith("wind_max")
            or column in [
                "consecutive_dry_days",
                "consecutive_hot_days",
            ]
        ):
            candidate_features.append(
                column
            )

    if not candidate_features:
        raise ValueError(
            "No XGBoost weather candidates were identified."
        )

    return sorted(
        candidate_features
    )


# ============================================================
# 8. TRAINING-ONLY FEATURE SCREENING
# ============================================================

def feature_simplicity_key(
    feature: str,
) -> tuple:
    """Prefer simpler and more directly interpretable predictors."""

    if feature in [
        "t2m_max_C",
        "rh_min",
        "wind_max",
    ]:
        return 0, feature

    if "_lag" in feature:
        return 1, feature

    if "mean" in feature or "_max" in feature:
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
    Select weather predictors from training rows only.

    Steps
    -----
    1. Remove constant predictors.
    2. Median-impute temporarily for screening.
    3. Calculate mutual information.
    4. Group predictors with absolute Spearman correlation > 0.90.
    5. Retain one representative from each group.
    6. Rank retained representatives by mutual information.
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

    mi_values = mutual_info_classif(
        X_filled,
        y,
        random_state=RANDOM_SEED,
    )

    mutual_information = dict(
        zip(
            nonconstant_features,
            mi_values,
        )
    )

    correlation_matrix = (
        X_filled
        .corr(method="spearman")
        .abs()
    )

    unassigned_features = set(
        nonconstant_features
    )

    correlation_groups: list[list[str]] = []

    while unassigned_features:

        first_feature = sorted(
            unassigned_features
        )[0]

        group = [
            feature
            for feature in sorted(unassigned_features)
            if correlation_matrix.loc[
                first_feature,
                feature,
            ] > CORRELATION_CUTOFF
        ]

        correlation_groups.append(
            group
        )

        unassigned_features.difference_update(
            group
        )

    representatives: list[str] = []

    group_lookup: dict[str, int] = {}

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
            group_lookup[feature] = group_number

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
                "correlation_group": group_lookup[
                    feature
                ],
                "subset_size_requested": subset_size,
            }
        )

    return (
        selected_features,
        pd.DataFrame(detail_rows),
    )


# ============================================================
# 9. FIT AND PREDICT WITH XGBOOST
# ============================================================

def fit_xgboost_model(
    training_data: pd.DataFrame,
    predictor_columns: list[str],
    parameters: dict,
) -> tuple[
    SimpleImputer,
    XGBClassifier,
    float,
]:
    """
    Fit median imputation and XGBoost on training data only.

    XGBoost does not require standardisation.
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

    X_train_imputed = imputer.fit_transform(
        X_train
    )

    scale_pos_weight = calculate_scale_pos_weight(
        labels=y_train
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=parameters["max_depth"],
        learning_rate=parameters["learning_rate"],
        n_estimators=parameters["n_estimators"],
        min_child_weight=parameters["min_child_weight"],
        subsample=parameters["subsample"],
        colsample_bytree=parameters["colsample_bytree"],
        reg_alpha=parameters["reg_alpha"],
        reg_lambda=parameters["reg_lambda"],
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )

    model.fit(
        X_train_imputed,
        y_train,
    )

    return (
        imputer,
        model,
        scale_pos_weight,
    )


def predict_xgboost_probability(
    dataframe: pd.DataFrame,
    predictor_columns: list[str],
    imputer: SimpleImputer,
    model: XGBClassifier,
) -> np.ndarray:
    """Return positive-class probabilities."""

    X = dataframe[
        predictor_columns
    ]

    X_imputed = imputer.transform(
        X
    )

    probability = model.predict_proba(
        X_imputed
    )[:, 1]

    return probability


# ============================================================
# 10. INNER TEMPORAL TUNING
# ============================================================

def tune_xgboost(
    outer_training_data: pd.DataFrame,
    inner_registry_for_outer_fold: pd.DataFrame,
    weather_candidates: list[str],
    structural_columns: list[str],
) -> tuple[
    dict,
    object,
    pd.DataFrame,
]:
    """
    Select feature subset size and XGBoost parameters using mean inner AP.
    """

    result_rows: list[dict] = []

    for subset_size in WEATHER_SUBSET_SIZES:

        for parameter_id, parameters in enumerate(
            XGBOOST_PARAMETER_GRID,
            start=1,
        ):

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

                predictor_columns = (
                    selected_weather
                    + structural_columns
                )

                (
                    imputer,
                    model,
                    scale_pos_weight,
                ) = fit_xgboost_model(
                    training_data=inner_train,
                    predictor_columns=predictor_columns,
                    parameters=parameters,
                )

                validation_probability = (
                    predict_xgboost_probability(
                        dataframe=inner_validation,
                        predictor_columns=predictor_columns,
                        imputer=imputer,
                        model=model,
                    )
                )

                validation_ap = safe_average_precision(
                    y_true=inner_validation[
                        "target_next_day"
                    ].to_numpy(dtype=int),
                    probability=validation_probability,
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
                        "parameter_id": parameter_id,
                        "max_depth": parameters["max_depth"],
                        "learning_rate": parameters["learning_rate"],
                        "n_estimators": parameters["n_estimators"],
                        "min_child_weight": parameters[
                            "min_child_weight"
                        ],
                        "subsample": parameters["subsample"],
                        "colsample_bytree": parameters[
                            "colsample_bytree"
                        ],
                        "reg_alpha": parameters["reg_alpha"],
                        "reg_lambda": parameters["reg_lambda"],
                        "scale_pos_weight": scale_pos_weight,
                        "number_of_selected_weather_features": len(
                            selected_weather
                        ),
                        "validation_AP": validation_ap,
                    }
                )

    inner_results = pd.DataFrame(
        result_rows
    )

    if inner_results.empty:
        raise ValueError(
            "No valid inner-validation result was produced."
        )

    summary = (
        inner_results
        .groupby(
            [
                "subset_size",
                "parameter_id",
                "max_depth",
                "learning_rate",
                "n_estimators",
                "min_child_weight",
                "subsample",
                "colsample_bytree",
                "reg_alpha",
                "reg_lambda",
            ],
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

    # Prefer a smaller feature subset and shallower model when AP is tied.
    selected_row = (
        practically_tied
        .sort_values(
            [
                "subset_order",
                "max_depth",
                "n_estimators",
                "learning_rate",
                "parameter_id",
            ]
        )
        .iloc[0]
    )

    selected_subset_size: object = (
        selected_row["subset_size"]
    )

    if str(selected_subset_size) != "all":
        selected_subset_size = int(
            selected_subset_size
        )

    selected_parameter_id = int(
        selected_row["parameter_id"]
    )

    selected_parameters = (
        XGBOOST_PARAMETER_GRID[
            selected_parameter_id - 1
        ].copy()
    )

    inner_results = inner_results.merge(
        summary,
        on=[
            "subset_size",
            "parameter_id",
            "max_depth",
            "learning_rate",
            "n_estimators",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
        ],
        how="left",
    )

    return (
        selected_parameters,
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
    selected_parameters: dict,
    outer_fold_id: int,
    outer_target_year: int,
) -> pd.DataFrame:
    """Generate row-level XGBoost inner-validation probabilities."""

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

        imputer, model, _ = fit_xgboost_model(
            training_data=inner_train,
            predictor_columns=predictors,
            parameters=selected_parameters,
        )

        raw_probability = predict_xgboost_probability(
            dataframe=inner_validation,
            predictor_columns=predictors,
            imputer=imputer,
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

        fold_predictions.insert(0, "model", "xgboost")
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
            f"No XGBoost inner OOF predictions were generated for "
            f"outer fold {outer_fold_id}."
        )

    return pd.concat(prediction_tables, ignore_index=True)


# ============================================================
# 11. MAIN PROGRAM
# ============================================================

def main() -> None:
    """Train and evaluate XGBoost across the five outer folds."""

    print("\n" + "=" * 75)
    print("CODE 05: LEAKAGE-FREE XGBOOST")
    print("=" * 75)

    # --------------------------------------------------------
    # 11.1 Check required files
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
    # 11.2 Load tables
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

    outer_date_columns = [
        "train_issue_start",
        "train_issue_end",
        "embargo_issue_start",
        "embargo_issue_end",
        "test_issue_start",
        "test_issue_end",
        "test_target_start",
        "test_target_end",
    ]

    for column in outer_date_columns:

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
    # 11.3 Add county indicators and identify predictors
    # --------------------------------------------------------

    feature_data, county_columns = add_county_fixed_effects(
        dataframe=feature_data
    )

    structural_columns = (
        STRUCTURAL_FEATURES
        + county_columns
    )

    weather_candidates = (
        identify_xgboost_weather_candidates(
            dataframe=feature_data,
            county_columns=county_columns,
        )
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
    # 11.4 Output containers
    # --------------------------------------------------------

    all_oof_predictions: list[pd.DataFrame] = []
    selected_feature_rows: list[dict] = []
    hyperparameter_rows: list[dict] = []
    feature_importance_rows: list[dict] = []
    all_inner_results: list[pd.DataFrame] = []
    all_inner_oof_predictions: list[pd.DataFrame] = []
    audit_rows: list[dict] = []

    # --------------------------------------------------------
    # 11.5 Process outer folds
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
            f"Outer fold {fold_id}: target year {target_year}"
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
                f"Outer fold {fold_id} contains empty train or test data."
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
        # Tune model inside outer training data
        # ----------------------------------------------------

        (
            selected_parameters,
            selected_subset_size,
            inner_results,
        ) = tune_xgboost(
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
            selected_parameters=selected_parameters,
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
            final_model,
            scale_pos_weight,
        ) = fit_xgboost_model(
            training_data=outer_train,
            predictor_columns=final_predictors,
            parameters=selected_parameters,
        )

        raw_test_probability = (
            predict_xgboost_probability(
                dataframe=outer_test,
                predictor_columns=final_predictors,
                imputer=final_imputer,
                model=final_model,
            )
        )

        outer_test_ap = safe_average_precision(
            y_true=outer_test[
                "target_next_day"
            ].to_numpy(dtype=int),
            probability=raw_test_probability,
        )

        print(
            f"Selected subset size : {selected_subset_size}"
        )

        print(
            f"Selected parameters  : {selected_parameters}"
        )

        print(
            f"scale_pos_weight     : {scale_pos_weight:.4f}"
        )

        print(
            f"Outer-test raw AP    : {outer_test_ap:.6f}"
        )

        # ----------------------------------------------------
        # Save OOF predictions
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
            "xgboost",
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

        fold_predictions[
            "raw_probability"
        ] = raw_test_probability

        all_oof_predictions.append(
            fold_predictions
        )

        # ----------------------------------------------------
        # Save selected features
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
        # Save selected hyperparameters
        # ----------------------------------------------------

        hyperparameter_row = {
            "outer_fold_id": fold_id,
            "target_year": target_year,
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
            "scale_pos_weight": scale_pos_weight,
            "outer_test_raw_AP": outer_test_ap,
        }

        hyperparameter_row.update(
            selected_parameters
        )

        hyperparameter_rows.append(
            hyperparameter_row
        )

        # ----------------------------------------------------
        # Save built-in gain importance
        # ----------------------------------------------------

        booster = final_model.get_booster()

        gain_scores = booster.get_score(
            importance_type="gain"
        )

        # XGBoost may return f0, f1, ... because NumPy arrays are used.
        for feature_index, predictor in enumerate(
            final_predictors
        ):

            internal_name = f"f{feature_index}"

            gain_value = float(
                gain_scores.get(
                    internal_name,
                    0.0,
                )
            )

            feature_importance_rows.append(
                {
                    "outer_fold_id": fold_id,
                    "target_year": target_year,
                    "predictor": predictor,
                    "importance_type": "gain",
                    "importance_value": gain_value,
                }
            )

        # ----------------------------------------------------
        # Save fold audit
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
                "selected_subset_size": selected_subset_size,
                "scale_pos_weight": scale_pos_weight,
                "outer_test_raw_AP": outer_test_ap,
            }
        )

    # --------------------------------------------------------
    # 11.6 Combine outputs
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

    feature_importance = pd.DataFrame(
        feature_importance_rows
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

    # --------------------------------------------------------
    # 11.7 Save outputs
    # --------------------------------------------------------

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

    feature_importance.to_csv(
        FEATURE_IMPORTANCE_FILE,
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
    # 11.8 Final summary
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
    print("CODE 05 COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print(
        f"Outer folds completed : "
        f"{oof_predictions['outer_fold_id'].nunique()}"
    )

    print(
        f"OOF prediction rows   : "
        f"{len(oof_predictions)}"
    )

    print(
        f"OOF target years      : "
        f"{sorted(oof_predictions['target_year'].unique().tolist())}"
    )

    print(
        f"Pooled raw OOF AP     : "
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
        f"\nFeature importance saved at:\n"
        f"{FEATURE_IMPORTANCE_FILE}"
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
