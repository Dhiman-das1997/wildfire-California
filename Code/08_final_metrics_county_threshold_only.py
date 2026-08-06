#!/usr/bin/env python3
"""
CODE 08: Final OOF metrics for county-specific thresholds
=========================================================

Run after Code 07.

This script evaluates Logistic Regression, XGBoost, and LSTM using the frozen
county-specific thresholds selected from inner OOF predictions.

It also summarizes:

1. panel, county, year, and county-year OOF metrics;
2. confusion matrices and calibration coordinates;
3. threshold sensitivity from Code 07;
4. Logistic and XGBoost feature-subset sensitivity;
5. LSTM lookback sensitivity;
6. selected feature subset and lookback by outer target year.

No model is retrained.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 1. USER SETTINGS
# ============================================================

DATA_DIR: Path = Path(
    r"wildfire-California\Data"
)

OUTPUT_DIR: Path = DATA_DIR / "wildfire_outputs-final"

NUMBER_OF_CALIBRATION_BINS: int = 10

MODEL_ORDER: list[str] = [
    "logistic_regression",
    "xgboost",
    "lstm",
]


# ============================================================
# 2. INPUT FILES
# ============================================================

PREDICTION_FILE: Path = (
    OUTPUT_DIR
    / "07_calibrated_oof_predictions.csv"
)

COUNTY_THRESHOLD_FILE: Path = (
    OUTPUT_DIR
    / "07_county_thresholds.csv"
)

THRESHOLD_SEARCH_FILE: Path = (
    OUTPUT_DIR
    / "07_threshold_search.csv"
)

THRESHOLD_SENSITIVITY_INPUT_FILE: Path = (
    OUTPUT_DIR
    / "07_threshold_sensitivity.csv"
)

LOGISTIC_INNER_FILE: Path = (
    OUTPUT_DIR
    / "04_logistic_inner_validation_results.csv"
)

XGBOOST_INNER_FILE: Path = (
    OUTPUT_DIR
    / "05_xgboost_inner_validation_results.csv"
)

LSTM_INNER_FILE: Path = (
    OUTPUT_DIR
    / "06_lstm_inner_validation_results.csv"
)

LOGISTIC_HYPERPARAMETER_FILE: Path = (
    OUTPUT_DIR
    / "04_logistic_hyperparameters.csv"
)

XGBOOST_HYPERPARAMETER_FILE: Path = (
    OUTPUT_DIR
    / "05_xgboost_hyperparameters.csv"
)

LSTM_HYPERPARAMETER_FILE: Path = (
    OUTPUT_DIR
    / "06_lstm_hyperparameters.csv"
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

PANEL_METRICS_FILE: Path = (
    OUTPUT_DIR
    / "08_panel_metrics.csv"
)

COUNTY_METRICS_FILE: Path = (
    OUTPUT_DIR
    / "08_county_metrics.csv"
)

YEARLY_METRICS_FILE: Path = (
    OUTPUT_DIR
    / "08_yearly_metrics.csv"
)

COUNTY_YEAR_METRICS_FILE: Path = (
    OUTPUT_DIR
    / "08_county_year_metrics.csv"
)

CONFUSION_MATRIX_FILE: Path = (
    OUTPUT_DIR
    / "08_confusion_matrices.csv"
)

CALIBRATION_CURVE_FILE: Path = (
    OUTPUT_DIR
    / "08_calibration_curves.csv"
)

THRESHOLD_SENSITIVITY_FILE: Path = (
    OUTPUT_DIR
    / "08_threshold_sensitivity_summary.csv"
)

FEATURE_SENSITIVITY_FILE: Path = (
    OUTPUT_DIR
    / "08_feature_sensitivity.csv"
)

SELECTED_COMPLEXITY_FILE: Path = (
    OUTPUT_DIR
    / "08_selected_model_complexity.csv"
)

MODEL_RANKING_FILE: Path = (
    OUTPUT_DIR
    / "08_model_ranking.csv"
)

AUDIT_FILE: Path = (
    OUTPUT_DIR
    / "08_final_audit.csv"
)


# ============================================================
# 4. REQUIRED COLUMNS
# ============================================================

REQUIRED_PREDICTION_COLUMNS: list[str] = [
    "model",
    "outer_fold_id",
    "target_year",
    "county",
    "date",
    "target_date",
    "target_next_day",
    "raw_probability",
    "calibrated_probability",
    "selected_threshold",
    "threshold_policy",
    "predicted_label",
]


# ============================================================
# 5. HELPERS
# ============================================================

def require_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{table_name} is missing required columns:\n{missing}"
        )


def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


def normalize_subset_label(
    value,
) -> str:
    value_text = str(
        value
    ).strip().lower()

    if value_text in {
        "all",
        "all.0",
    }:
        return "all"

    try:
        return str(
            int(
                float(
                    value_text
                )
            )
        )
    except ValueError:
        return value_text


def safe_average_precision(
    labels: np.ndarray,
    probability: np.ndarray,
) -> float:
    if np.unique(labels).size < 2:
        return np.nan

    return float(
        average_precision_score(
            labels,
            probability,
        )
    )


def safe_roc_auc(
    labels: np.ndarray,
    probability: np.ndarray,
) -> float:
    if np.unique(labels).size < 2:
        return np.nan

    return float(
        roc_auc_score(
            labels,
            probability,
        )
    )


def trapezoidal_pr_auc(
    labels: np.ndarray,
    probability: np.ndarray,
) -> float:
    if np.unique(labels).size < 2:
        return np.nan

    precision, recall, _ = (
        precision_recall_curve(
            labels,
            probability,
        )
    )

    order = np.argsort(
        recall
    )

    sorted_recall = recall[
        order
    ]

    sorted_precision = precision[
        order
    ]

    if hasattr(
        np,
        "trapezoid",
    ):
        area = np.trapezoid(
            sorted_precision,
            sorted_recall,
        )
    elif hasattr(
        np,
        "trapz",
    ):
        area = np.trapz(
            sorted_precision,
            sorted_recall,
        )
    else:
        area = np.sum(
            np.diff(
                sorted_recall
            )
            * 0.5
            * (
                sorted_precision[:-1]
                + sorted_precision[1:]
            )
        )

    return float(
        area
    )


# ============================================================
# 6. LOAD FINAL PREDICTIONS
# ============================================================

def load_predictions() -> pd.DataFrame:
    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Code 07 output was not found:\n{PREDICTION_FILE}"
        )

    dataframe = pd.read_csv(
        PREDICTION_FILE
    )

    require_columns(
        dataframe,
        REQUIRED_PREDICTION_COLUMNS,
        PREDICTION_FILE.name,
    )

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="raise",
    ).dt.normalize()

    dataframe["target_date"] = pd.to_datetime(
        dataframe["target_date"],
        errors="raise",
    ).dt.normalize()

    dataframe["target_next_day"] = (
        dataframe["target_next_day"]
        .astype(int)
    )

    dataframe["predicted_label"] = (
        dataframe["predicted_label"]
        .astype(int)
    )

    dataframe["target_year"] = (
        dataframe["target_year"]
        .astype(int)
    )

    observed_models = set(
        dataframe["model"].unique()
    )

    unexpected_models = (
        observed_models
        - set(MODEL_ORDER)
    )

    if unexpected_models:
        raise ValueError(
            "Unexpected models were found in Code 07 output:\n"
            f"{sorted(unexpected_models)}"
        )

    return dataframe


# ============================================================
# 7. COMPLETE METRICS
# ============================================================

def calculate_metrics(
    dataframe: pd.DataFrame,
) -> dict:
    labels = dataframe[
        "target_next_day"
    ].to_numpy(dtype=int)

    probability = dataframe[
        "calibrated_probability"
    ].to_numpy(dtype=float)

    predicted = dataframe[
        "predicted_label"
    ].to_numpy(dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predicted,
        labels=[0, 1],
    ).ravel()

    prevalence = float(
        labels.mean()
    )

    average_precision = (
        safe_average_precision(
            labels,
            probability,
        )
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else np.nan
    )

    brier = float(
        brier_score_loss(
            labels,
            probability,
        )
    )

    climatology = np.full(
        len(labels),
        prevalence,
        dtype=float,
    )

    climatology_brier = float(
        brier_score_loss(
            labels,
            climatology,
        )
    )

    return {
        "n_rows": int(
            len(dataframe)
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "prevalence": prevalence,
        "accuracy": float(
            accuracy_score(
                labels,
                predicted,
            )
        ),
        "AP": average_precision,
        "PR_AUC_trapezoid": (
            trapezoidal_pr_auc(
                labels,
                probability,
            )
        ),
        "PR_lift": (
            average_precision
            / prevalence
            if prevalence > 0
            else np.nan
        ),
        "ROC_AUC": safe_roc_auc(
            labels,
            probability,
        ),
        "Brier": brier,
        "climatology_Brier": (
            climatology_brier
        ),
        "Brier_skill_score": (
            1.0
            - brier
            / climatology_brier
            if climatology_brier > 0
            else np.nan
        ),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "FPR": (
            fp / (fp + tn)
            if (fp + tn) > 0
            else np.nan
        ),
        "F1": float(
            f1_score(
                labels,
                predicted,
                zero_division=0,
            )
        ),
        "balanced_accuracy": (
            float(
                balanced_accuracy_score(
                    labels,
                    predicted,
                )
            )
            if np.unique(labels).size == 2
            else np.nan
        ),
        "custom_score": (
            recall * specificity
            if (
                np.isfinite(recall)
                and np.isfinite(
                    specificity
                )
            )
            else np.nan
        ),
        "warning_rate": float(
            predicted.mean()
        ),
        "false_alarms_per_detected_ignition": (
            fp / tp
            if tp > 0
            else np.nan
        ),
    }


def grouped_metrics(
    dataframe: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []

    for group_values, group_data in (
        dataframe.groupby(
            group_columns,
            dropna=False,
            sort=True,
        )
    ):
        if not isinstance(
            group_values,
            tuple,
        ):
            group_values = (
                group_values,
            )

        row = {
            column: value
            for column, value in zip(
                group_columns,
                group_values,
            )
        }

        row.update(
            calculate_metrics(
                group_data
            )
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 8. CONFUSION MATRICES AND CALIBRATION
# ============================================================

def confusion_matrix_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for model_name, model_data in (
        dataframe.groupby(
            "model",
            sort=True,
        )
    ):
        for county, county_data in (
            model_data.groupby(
                "county",
                sort=True,
            )
        ):
            matrix = confusion_matrix(
                county_data[
                    "target_next_day"
                ],
                county_data[
                    "predicted_label"
                ],
                labels=[0, 1],
            )

            for observed_class in [
                0,
                1,
            ]:
                row_total = matrix[
                    observed_class,
                    :
                ].sum()

                for predicted_class in [
                    0,
                    1,
                ]:
                    count = int(
                        matrix[
                            observed_class,
                            predicted_class,
                        ]
                    )

                    rows.append(
                        {
                            "model": model_name,
                            "county": county,
                            "observed_class": (
                                observed_class
                            ),
                            "predicted_class": (
                                predicted_class
                            ),
                            "count": count,
                            "row_normalized_fraction": (
                                count / row_total
                                if row_total > 0
                                else np.nan
                            ),
                        }
                    )

    return pd.DataFrame(rows)


def calibration_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for (
        model_name,
        county,
    ), group in dataframe.groupby(
        [
            "model",
            "county",
        ],
        sort=True,
    ):
        if group[
            "target_next_day"
        ].nunique() < 2:
            continue

        observed, predicted = (
            calibration_curve(
                group[
                    "target_next_day"
                ],
                group[
                    "calibrated_probability"
                ],
                n_bins=(
                    NUMBER_OF_CALIBRATION_BINS
                ),
                strategy="quantile",
            )
        )

        for bin_id, (
            predicted_probability,
            observed_fraction,
        ) in enumerate(
            zip(
                predicted,
                observed,
            ),
            start=1,
        ):
            rows.append(
                {
                    "model": model_name,
                    "county": county,
                    "bin_id": bin_id,
                    "mean_predicted_probability": (
                        float(
                            predicted_probability
                        )
                    ),
                    "observed_event_fraction": (
                        float(
                            observed_fraction
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 9. THRESHOLD SENSITIVITY SUMMARY
# ============================================================

def summarize_threshold_sensitivity() -> pd.DataFrame:
    if not THRESHOLD_SENSITIVITY_INPUT_FILE.exists():
        raise FileNotFoundError(
            "Code 07 threshold-sensitivity file was not found:\n"
            f"{THRESHOLD_SENSITIVITY_INPUT_FILE}"
        )

    sensitivity = pd.read_csv(
        THRESHOLD_SENSITIVITY_INPUT_FILE
    )

    required = [
        "model",
        "outer_fold_id",
        "target_year",
        "county",
        "threshold_delta",
        "applied_threshold",
        "recall",
        "specificity",
        "custom_score",
        "warning_rate",
    ]

    require_columns(
        sensitivity,
        required,
        THRESHOLD_SENSITIVITY_INPUT_FILE.name,
    )

    summary = (
        sensitivity
        .groupby(
            [
                "model",
                "county",
                "threshold_delta",
            ],
            as_index=False,
        )
        .agg(
            mean_applied_threshold=(
                "applied_threshold",
                "mean",
            ),
            mean_recall=(
                "recall",
                "mean",
            ),
            mean_specificity=(
                "specificity",
                "mean",
            ),
            mean_custom_score=(
                "custom_score",
                "mean",
            ),
            mean_warning_rate=(
                "warning_rate",
                "mean",
            ),
        )
    )

    return summary


# ============================================================
# 10. FEATURE SENSITIVITY
# ============================================================

def summarize_tabular_feature_sensitivity(
    file_path: Path,
    model_name: str,
) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame()

    data = pd.read_csv(
        file_path
    )

    subset_column = first_existing_column(
        data,
        [
            "subset_size",
            "weather_subset_size",
            "selected_weather_subset_size",
        ],
    )

    ap_column = first_existing_column(
        data,
        [
            "validation_AP",
            "average_precision",
            "inner_validation_AP",
            "mean_validation_AP",
            "AP",
        ],
    )

    fold_column = first_existing_column(
        data,
        [
            "outer_fold_id",
            "target_year",
            "outer_target_year",
        ],
    )

    if (
        subset_column is None
        or ap_column is None
        or fold_column is None
    ):
        return pd.DataFrame()

    output = data[
        [
            fold_column,
            subset_column,
            ap_column,
        ]
    ].copy()

    output = output.rename(
        columns={
            fold_column: (
                "outer_fold_or_year"
            ),
            subset_column: (
                "complexity_value"
            ),
            ap_column: "inner_AP",
        }
    )

    output[
        "complexity_value"
    ] = output[
        "complexity_value"
    ].map(
        normalize_subset_label
    )

    output["model"] = model_name
    output["sensitivity_type"] = (
        "weather_feature_subset"
    )

    return output[
        [
            "model",
            "sensitivity_type",
            "outer_fold_or_year",
            "complexity_value",
            "inner_AP",
        ]
    ]


def summarize_lstm_sensitivity() -> pd.DataFrame:
    if not LSTM_INNER_FILE.exists():
        return pd.DataFrame()

    data = pd.read_csv(
        LSTM_INNER_FILE
    )

    lookback_column = first_existing_column(
        data,
        [
            "lookback",
            "lookback_days",
            "sequence_length",
        ],
    )

    ap_column = first_existing_column(
        data,
        [
            "validation_AP",
            "average_precision",
            "inner_validation_AP",
            "mean_validation_AP",
            "AP",
        ],
    )

    fold_column = first_existing_column(
        data,
        [
            "outer_fold_id",
            "target_year",
            "outer_target_year",
        ],
    )

    if (
        lookback_column is None
        or ap_column is None
        or fold_column is None
    ):
        return pd.DataFrame()

    output = data[
        [
            fold_column,
            lookback_column,
            ap_column,
        ]
    ].copy()

    output = output.rename(
        columns={
            fold_column: (
                "outer_fold_or_year"
            ),
            lookback_column: (
                "complexity_value"
            ),
            ap_column: "inner_AP",
        }
    )

    output["model"] = "lstm"
    output["sensitivity_type"] = (
        "lookback_days"
    )

    return output[
        [
            "model",
            "sensitivity_type",
            "outer_fold_or_year",
            "complexity_value",
            "inner_AP",
        ]
    ]


def create_feature_sensitivity_table() -> pd.DataFrame:
    tables = [
        summarize_tabular_feature_sensitivity(
            LOGISTIC_INNER_FILE,
            "logistic_regression",
        ),
        summarize_tabular_feature_sensitivity(
            XGBOOST_INNER_FILE,
            "xgboost",
        ),
        summarize_lstm_sensitivity(),
    ]

    tables = [
        table
        for table in tables
        if not table.empty
    ]

    if not tables:
        return pd.DataFrame(
            columns=[
                "model",
                "sensitivity_type",
                "outer_fold_or_year",
                "complexity_value",
                "inner_AP",
            ]
        )

    return pd.concat(
        tables,
        ignore_index=True,
    )


def create_selected_complexity_table() -> pd.DataFrame:
    rows: list[dict] = []

    for (
        file_path,
        model_name,
        complexity_candidates,
        complexity_type,
    ) in [
        (
            LOGISTIC_HYPERPARAMETER_FILE,
            "logistic_regression",
            [
                "selected_weather_subset_size",
                "weather_subset_size",
                "subset_size",
            ],
            "weather_feature_subset",
        ),
        (
            XGBOOST_HYPERPARAMETER_FILE,
            "xgboost",
            [
                "selected_weather_subset_size",
                "weather_subset_size",
                "subset_size",
            ],
            "weather_feature_subset",
        ),
        (
            LSTM_HYPERPARAMETER_FILE,
            "lstm",
            [
                "lookback",
                "lookback_days",
                "sequence_length",
            ],
            "lookback_days",
        ),
    ]:
        if not file_path.exists():
            continue

        data = pd.read_csv(
            file_path
        )

        year_column = first_existing_column(
            data,
            [
                "target_year",
                "outer_target_year",
            ],
        )

        complexity_column = (
            first_existing_column(
                data,
                complexity_candidates,
            )
        )

        if (
            year_column is None
            or complexity_column is None
        ):
            continue

        for row in data.itertuples(
            index=False
        ):
            value = getattr(
                row,
                complexity_column,
            )

            if (
                complexity_type
                == "weather_feature_subset"
            ):
                value = normalize_subset_label(
                    value
                )

            rows.append(
                {
                    "model": model_name,
                    "target_year": int(
                        getattr(
                            row,
                            year_column,
                        )
                    ),
                    "complexity_type": (
                        complexity_type
                    ),
                    "selected_value": value,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 11. RANKING AND AUDIT
# ============================================================

def create_model_ranking(
    panel_metrics: pd.DataFrame,
) -> pd.DataFrame:
    ranking = panel_metrics.copy()

    ranking["AP_rank"] = (
        ranking["AP"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    ranking[
        "custom_score_rank"
    ] = (
        ranking[
            "custom_score"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    ranking["Brier_rank"] = (
        ranking["Brier"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    ranking = ranking.sort_values(
        [
            "AP_rank",
            "custom_score_rank",
            "Brier_rank",
        ]
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "overall_order",
        np.arange(
            1,
            len(ranking) + 1,
        ),
    )

    return ranking


def create_audit(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    duplicate_count = int(
        predictions.duplicated(
            subset=[
                "model",
                "outer_fold_id",
                "county",
                "target_date",
            ]
        ).sum()
    )

    rows = [
        {
            "check": (
                "duplicate_model_fold_county_target"
            ),
            "status": (
                "PASS"
                if duplicate_count == 0
                else "FAIL"
            ),
            "observed": duplicate_count,
            "required": 0,
        }
    ]

    for model_name, group in (
        predictions.groupby(
            "model",
            sort=True,
        )
    ):
        rows.append(
            {
                "check": (
                    f"county_count::{model_name}"
                ),
                "status": (
                    "PASS"
                    if group[
                        "county"
                    ].nunique() == 10
                    else "FAIL"
                ),
                "observed": int(
                    group[
                        "county"
                    ].nunique()
                ),
                "required": 10,
            }
        )

        rows.append(
            {
                "check": (
                    f"target_year_count::{model_name}"
                ),
                "status": (
                    "PASS"
                    if group[
                        "target_year"
                    ].nunique() == 5
                    else "FAIL"
                ),
                "observed": int(
                    group[
                        "target_year"
                    ].nunique()
                ),
                "required": 5,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 12. MAIN
# ============================================================

def main() -> None:
    print("\n" + "=" * 82)
    print(
        "CODE 08: FINAL METRICS FOR COUNTY-SPECIFIC THRESHOLDS"
    )
    print("=" * 82)

    predictions = load_predictions()

    panel_metrics = grouped_metrics(
        predictions,
        ["model"],
    )

    county_metrics = grouped_metrics(
        predictions,
        [
            "model",
            "county",
        ],
    )

    yearly_metrics = grouped_metrics(
        predictions,
        [
            "model",
            "target_year",
        ],
    )

    county_year_metrics = (
        grouped_metrics(
            predictions,
            [
                "model",
                "county",
                "target_year",
            ],
        )
    )

    confusion_matrices = (
        confusion_matrix_table(
            predictions
        )
    )

    calibration_curves = (
        calibration_table(
            predictions
        )
    )

    threshold_sensitivity = (
        summarize_threshold_sensitivity()
    )

    feature_sensitivity = (
        create_feature_sensitivity_table()
    )

    selected_complexity = (
        create_selected_complexity_table()
    )

    model_ranking = (
        create_model_ranking(
            panel_metrics
        )
    )

    audit = create_audit(
        predictions
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel_metrics.to_csv(
        PANEL_METRICS_FILE,
        index=False,
    )

    county_metrics.to_csv(
        COUNTY_METRICS_FILE,
        index=False,
    )

    yearly_metrics.to_csv(
        YEARLY_METRICS_FILE,
        index=False,
    )

    county_year_metrics.to_csv(
        COUNTY_YEAR_METRICS_FILE,
        index=False,
    )

    confusion_matrices.to_csv(
        CONFUSION_MATRIX_FILE,
        index=False,
    )

    calibration_curves.to_csv(
        CALIBRATION_CURVE_FILE,
        index=False,
    )

    threshold_sensitivity.to_csv(
        THRESHOLD_SENSITIVITY_FILE,
        index=False,
    )

    feature_sensitivity.to_csv(
        FEATURE_SENSITIVITY_FILE,
        index=False,
    )

    selected_complexity.to_csv(
        SELECTED_COMPLEXITY_FILE,
        index=False,
    )

    model_ranking.to_csv(
        MODEL_RANKING_FILE,
        index=False,
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
    )

    print("\n" + "=" * 82)
    print(
        "FINAL PANEL-POOLED RESULTS"
    )
    print("=" * 82)

    print(
        panel_metrics[
            [
                "model",
                "AP",
                "PR_lift",
                "ROC_AUC",
                "Brier",
                "precision",
                "recall",
                "specificity",
                "F1",
                "custom_score",
                "warning_rate",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 82)
    print(
        "CODE 08 COMPLETED SUCCESSFULLY"
    )
    print("=" * 82)

    print(
        f"\nThreshold sensitivity:\n"
        f"{THRESHOLD_SENSITIVITY_FILE}"
    )

    print(
        f"\nFeature sensitivity:\n"
        f"{FEATURE_SENSITIVITY_FILE}"
    )


if __name__ == "__main__":
    main()
