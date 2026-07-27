#!/usr/bin/env python3
"""
CODE 07: Leakage-free calibration and county-specific threshold selection
=========================================================================

Run after revised Codes 04, 05, and 06.

For every model and outer fold this script:

1. Fits one Platt calibrator using inner OOF predictions only.
2. Applies the frozen calibrator to untouched outer-test probabilities.
3. Selects one threshold separately for every county using only that county's
   calibrated inner OOF predictions.
4. Uses the pooled model-fold threshold only when a county does not contain
   both classes in the inner OOF data.
5. Applies the frozen county threshold to the corresponding outer-test rows.
6. Saves row-level calibrated OOF predictions, county thresholds, complete
   threshold-search curves, threshold-sensitivity tables, and audit outputs.

No ensemble is constructed. Outer-test labels are never used to fit a
calibrator or select a threshold.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 1. USER SETTINGS
# ============================================================

DATA_DIR: Path = Path(
    r"C:\Users\Dhiman Das\Documents\10 JUNE Final paper"
)

OUTPUT_DIR: Path = DATA_DIR / "wildfire_outputs-final"

RANDOM_SEED: int = 42
PROBABILITY_EPSILON: float = 1e-6

THRESHOLD_GRID: np.ndarray = np.linspace(
    0.005,
    0.50,
    100,
)

THRESHOLD_DELTAS: list[float] = [
    -0.02,
    -0.01,
    0.00,
    0.01,
    0.02,
    0.05,
]


# ============================================================
# 2. INPUT FILES
# ============================================================

MODEL_FILES: dict[str, dict[str, Path]] = {
    "logistic_regression": {
        "outer": (
            OUTPUT_DIR
            / "04_logistic_oof_raw_predictions.csv"
        ),
        "inner": (
            OUTPUT_DIR
            / "04_logistic_inner_oof_predictions.csv"
        ),
    },
    "xgboost": {
        "outer": (
            OUTPUT_DIR
            / "05_xgboost_oof_raw_predictions.csv"
        ),
        "inner": (
            OUTPUT_DIR
            / "05_xgboost_inner_oof_predictions.csv"
        ),
    },
    "lstm": {
        "outer": (
            OUTPUT_DIR
            / "06_lstm_oof_raw_predictions.csv"
        ),
        "inner": (
            OUTPUT_DIR
            / "06_lstm_inner_oof_predictions.csv"
        ),
    },
}


# ============================================================
# 3. OUTPUT FILES
# ============================================================

CALIBRATED_PREDICTION_FILE: Path = (
    OUTPUT_DIR
    / "07_calibrated_oof_predictions.csv"
)

CALIBRATION_PARAMETER_FILE: Path = (
    OUTPUT_DIR
    / "07_calibration_parameters.csv"
)

COUNTY_THRESHOLD_FILE: Path = (
    OUTPUT_DIR
    / "07_county_thresholds.csv"
)

THRESHOLD_SEARCH_FILE: Path = (
    OUTPUT_DIR
    / "07_threshold_search.csv"
)

THRESHOLD_SENSITIVITY_FILE: Path = (
    OUTPUT_DIR
    / "07_threshold_sensitivity.csv"
)

CALIBRATION_AUDIT_FILE: Path = (
    OUTPUT_DIR
    / "07_calibration_audit.csv"
)


# ============================================================
# 4. REQUIRED COLUMNS
# ============================================================

OUTER_REQUIRED_COLUMNS: list[str] = [
    "outer_fold_id",
    "target_year",
    "county",
    "date",
    "target_date",
    "target_next_day",
    "raw_probability",
]

INNER_REQUIRED_COLUMNS: list[str] = [
    "outer_fold_id",
    "inner_fold_id",
    "validation_target_year",
    "county",
    "date",
    "target_date",
    "target_next_day",
    "raw_probability",
]


# ============================================================
# 5. BASIC HELPERS
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


def validate_probability(
    values: np.ndarray,
    table_name: str,
) -> None:
    values = np.asarray(
        values,
        dtype=float,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{table_name} contains non-finite probabilities."
        )

    if (
        np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError(
            f"{table_name} contains probabilities outside [0, 1]."
        )


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


def safe_brier(
    labels: np.ndarray,
    probability: np.ndarray,
) -> float:
    if len(labels) == 0:
        return np.nan

    return float(
        brier_score_loss(
            labels,
            probability,
        )
    )


def probability_to_logit(
    probability: np.ndarray,
) -> np.ndarray:
    probability = np.clip(
        np.asarray(
            probability,
            dtype=float,
        ),
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )

    return np.log(
        probability
        / (1.0 - probability)
    )


# ============================================================
# 6. LOAD PREDICTIONS
# ============================================================

def load_predictions() -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    outer_tables: dict[str, pd.DataFrame] = {}
    inner_tables: dict[str, pd.DataFrame] = {}

    for model_name, paths in MODEL_FILES.items():
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(
                    f"Required prediction file was not found:\n{path}"
                )

        outer = pd.read_csv(
            paths["outer"]
        )

        inner = pd.read_csv(
            paths["inner"]
        )

        require_columns(
            outer,
            OUTER_REQUIRED_COLUMNS,
            paths["outer"].name,
        )

        require_columns(
            inner,
            INNER_REQUIRED_COLUMNS,
            paths["inner"].name,
        )

        outer["model"] = model_name
        inner["model"] = model_name

        for dataframe in [
            outer,
            inner,
        ]:
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

            dataframe["outer_fold_id"] = (
                dataframe["outer_fold_id"]
                .astype(int)
            )

            dataframe["raw_probability"] = (
                dataframe["raw_probability"]
                .astype(float)
            )

        outer["target_year"] = (
            outer["target_year"]
            .astype(int)
        )

        inner["inner_fold_id"] = (
            inner["inner_fold_id"]
            .astype(int)
        )

        inner["validation_target_year"] = (
            inner["validation_target_year"]
            .astype(int)
        )

        validate_probability(
            outer["raw_probability"],
            paths["outer"].name,
        )

        validate_probability(
            inner["raw_probability"],
            paths["inner"].name,
        )

        outer_tables[
            model_name
        ] = outer

        inner_tables[
            model_name
        ] = inner

    return outer_tables, inner_tables


# ============================================================
# 7. PLATT CALIBRATION
# ============================================================

def fit_platt_calibrator(
    raw_probability: np.ndarray,
    labels: np.ndarray,
) -> LogisticRegression:
    labels = np.asarray(
        labels,
        dtype=int,
    )

    if np.unique(labels).size < 2:
        raise ValueError(
            "Calibration data must contain both classes."
        )

    calibrator = LogisticRegression(
        penalty="l2",
        C=1.0e6,
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_SEED,
    )

    calibrator.fit(
        probability_to_logit(
            raw_probability
        ).reshape(-1, 1),
        labels,
    )

    return calibrator


def apply_platt_calibrator(
    calibrator: LogisticRegression,
    raw_probability: np.ndarray,
) -> np.ndarray:
    return calibrator.predict_proba(
        probability_to_logit(
            raw_probability
        ).reshape(-1, 1)
    )[:, 1]


# ============================================================
# 8. THRESHOLD METRICS
# ============================================================

def threshold_metrics(
    labels: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict:
    labels = np.asarray(
        labels,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    predicted = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predicted,
        labels=[0, 1],
    ).ravel()

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

    custom_score = (
        recall * specificity
        if (
            np.isfinite(recall)
            and np.isfinite(specificity)
        )
        else np.nan
    )

    return {
        "threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "FPR": (
            fp / (fp + tn)
            if (fp + tn) > 0
            else np.nan
        ),
        "custom_score": custom_score,
        "warning_rate": float(
            predicted.mean()
        ),
    }


def select_threshold(
    labels: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    table = pd.DataFrame(
        [
            threshold_metrics(
                labels,
                probability,
                float(threshold),
            )
            for threshold in THRESHOLD_GRID
        ]
    )

    valid = table.dropna(
        subset=[
            "custom_score",
            "recall",
            "specificity",
        ]
    )

    if valid.empty:
        raise ValueError(
            "No valid threshold was found."
        )

    best = (
        valid
        .sort_values(
            [
                "custom_score",
                "recall",
                "specificity",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .iloc[0]
    )

    selected = float(
        best["threshold"]
    )

    table[
        "selected_threshold"
    ] = (
        np.isclose(
            table["threshold"],
            selected,
        )
    )

    return selected, table


# ============================================================
# 9. COUNTY-SPECIFIC THRESHOLDS
# ============================================================

def build_county_threshold_table(
    calibrated_inner: pd.DataFrame,
    model_name: str,
    outer_fold_id: int,
    target_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_threshold, pooled_search = (
        select_threshold(
            calibrated_inner[
                "target_next_day"
            ],
            calibrated_inner[
                "calibrated_probability"
            ],
        )
    )

    pooled_search.insert(
        0,
        "model",
        model_name,
    )
    pooled_search.insert(
        1,
        "outer_fold_id",
        outer_fold_id,
    )
    pooled_search.insert(
        2,
        "target_year",
        target_year,
    )
    pooled_search.insert(
        3,
        "county",
        "ALL",
    )
    pooled_search.insert(
        4,
        "search_scope",
        "pooled",
    )

    threshold_rows: list[dict] = []
    search_tables: list[pd.DataFrame] = [
        pooled_search
    ]

    for county, county_data in (
        calibrated_inner.groupby(
            "county",
            sort=True,
        )
    ):
        labels = county_data[
            "target_next_day"
        ].to_numpy(dtype=int)

        probability = county_data[
            "calibrated_probability"
        ].to_numpy(dtype=float)

        positive_count = int(
            np.sum(labels == 1)
        )

        negative_count = int(
            np.sum(labels == 0)
        )

        if (
            positive_count > 0
            and negative_count > 0
        ):
            (
                county_threshold,
                county_search,
            ) = select_threshold(
                labels,
                probability,
            )

            threshold_source = (
                "county_specific"
            )

            county_search.insert(
                0,
                "model",
                model_name,
            )
            county_search.insert(
                1,
                "outer_fold_id",
                outer_fold_id,
            )
            county_search.insert(
                2,
                "target_year",
                target_year,
            )
            county_search.insert(
                3,
                "county",
                county,
            )
            county_search.insert(
                4,
                "search_scope",
                "county",
            )

            search_tables.append(
                county_search
            )
        else:
            county_threshold = (
                pooled_threshold
            )

            threshold_source = (
                "pooled_fallback"
            )

        county_metrics = threshold_metrics(
            labels,
            probability,
            county_threshold,
        )

        threshold_rows.append(
            {
                "model": model_name,
                "outer_fold_id": (
                    outer_fold_id
                ),
                "target_year": (
                    target_year
                ),
                "county": county,
                "pooled_threshold": (
                    pooled_threshold
                ),
                "county_threshold": (
                    county_threshold
                ),
                "selected_threshold": (
                    county_threshold
                ),
                "threshold_policy": (
                    threshold_source
                ),
                "inner_rows": len(
                    county_data
                ),
                "inner_positive_count": (
                    positive_count
                ),
                "inner_negative_count": (
                    negative_count
                ),
                "inner_recall_at_selected": (
                    county_metrics[
                        "recall"
                    ]
                ),
                "inner_specificity_at_selected": (
                    county_metrics[
                        "specificity"
                    ]
                ),
                "inner_custom_score_at_selected": (
                    county_metrics[
                        "custom_score"
                    ]
                ),
                "inner_warning_rate_at_selected": (
                    county_metrics[
                        "warning_rate"
                    ]
                ),
            }
        )

    return (
        pd.DataFrame(
            threshold_rows
        ),
        pd.concat(
            search_tables,
            ignore_index=True,
        ),
    )


def apply_county_thresholds(
    calibrated_outer: pd.DataFrame,
    threshold_table: pd.DataFrame,
) -> pd.DataFrame:
    output = calibrated_outer.merge(
        threshold_table,
        on=[
            "model",
            "outer_fold_id",
            "target_year",
            "county",
        ],
        how="left",
        validate="many_to_one",
    )

    if output[
        "selected_threshold"
    ].isna().any():
        raise ValueError(
            "Some outer-test rows did not receive a county threshold."
        )

    output[
        "predicted_label"
    ] = (
        output[
            "calibrated_probability"
        ].to_numpy(dtype=float)
        >= output[
            "selected_threshold"
        ].to_numpy(dtype=float)
    ).astype(int)

    return output


# ============================================================
# 10. THRESHOLD SENSITIVITY
# ============================================================

def create_threshold_sensitivity(
    calibrated_inner: pd.DataFrame,
    threshold_table: pd.DataFrame,
) -> pd.DataFrame:
    merged = calibrated_inner.merge(
        threshold_table[
            [
                "model",
                "outer_fold_id",
                "target_year",
                "county",
                "selected_threshold",
            ]
        ],
        on=[
            "model",
            "outer_fold_id",
            "county",
        ],
        how="left",
        suffixes=(
            "",
            "_threshold",
        ),
        validate="many_to_one",
    )

    rows: list[dict] = []

    for (
        model_name,
        outer_fold_id,
        county,
    ), group in merged.groupby(
        [
            "model",
            "outer_fold_id",
            "county",
        ],
        sort=True,
    ):
        selected_threshold = float(
            group[
                "selected_threshold"
            ].iloc[0]
        )

        for delta in THRESHOLD_DELTAS:
            applied_threshold = float(
                np.clip(
                    selected_threshold
                    + delta,
                    THRESHOLD_GRID.min(),
                    THRESHOLD_GRID.max(),
                )
            )

            metrics = threshold_metrics(
                group[
                    "target_next_day"
                ],
                group[
                    "calibrated_probability"
                ],
                applied_threshold,
            )

            rows.append(
                {
                    "model": model_name,
                    "outer_fold_id": int(
                        outer_fold_id
                    ),
                    "target_year": int(
                        threshold_table.loc[
                            (
                                threshold_table[
                                    "model"
                                ]
                                == model_name
                            )
                            & (
                                threshold_table[
                                    "outer_fold_id"
                                ]
                                == outer_fold_id
                            )
                            & (
                                threshold_table[
                                    "county"
                                ]
                                == county
                            ),
                            "target_year",
                        ].iloc[0]
                    ),
                    "county": county,
                    "selected_threshold": (
                        selected_threshold
                    ),
                    "threshold_delta": (
                        delta
                    ),
                    "applied_threshold": (
                        applied_threshold
                    ),
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key != "threshold"
                    },
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 11. MAIN
# ============================================================

def main() -> None:
    print("\n" + "=" * 82)
    print(
        "CODE 07: COUNTY-SPECIFIC THRESHOLD SELECTION"
    )
    print("=" * 82)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        outer_tables,
        inner_tables,
    ) = load_predictions()

    prediction_tables: list[pd.DataFrame] = []
    parameter_rows: list[dict] = []
    threshold_tables: list[pd.DataFrame] = []
    search_tables: list[pd.DataFrame] = []
    sensitivity_tables: list[pd.DataFrame] = []
    audit_rows: list[dict] = []

    for model_name in MODEL_FILES:
        outer_model = outer_tables[
            model_name
        ]

        inner_model = inner_tables[
            model_name
        ]

        for outer_fold_id in sorted(
            outer_model[
                "outer_fold_id"
            ].unique()
        ):
            outer = outer_model[
                outer_model[
                    "outer_fold_id"
                ]
                == outer_fold_id
            ].copy()

            inner = inner_model[
                inner_model[
                    "outer_fold_id"
                ]
                == outer_fold_id
            ].copy()

            target_year = int(
                outer[
                    "target_year"
                ].iloc[0]
            )

            print(
                f"\nModel={model_name} | "
                f"Outer fold={outer_fold_id} | "
                f"Target year={target_year}"
            )

            calibrator = fit_platt_calibrator(
                inner[
                    "raw_probability"
                ],
                inner[
                    "target_next_day"
                ],
            )

            inner[
                "calibrated_probability"
            ] = apply_platt_calibrator(
                calibrator,
                inner[
                    "raw_probability"
                ],
            )

            outer[
                "calibrated_probability"
            ] = apply_platt_calibrator(
                calibrator,
                outer[
                    "raw_probability"
                ],
            )

            (
                county_thresholds,
                threshold_search,
            ) = build_county_threshold_table(
                inner,
                model_name,
                int(
                    outer_fold_id
                ),
                target_year,
            )

            final_predictions = (
                apply_county_thresholds(
                    outer,
                    county_thresholds,
                )
            )

            threshold_sensitivity = (
                create_threshold_sensitivity(
                    inner,
                    county_thresholds,
                )
            )

            prediction_tables.append(
                final_predictions
            )

            threshold_tables.append(
                county_thresholds
            )

            search_tables.append(
                threshold_search
            )

            sensitivity_tables.append(
                threshold_sensitivity
            )

            parameter_rows.append(
                {
                    "model": model_name,
                    "outer_fold_id": int(
                        outer_fold_id
                    ),
                    "target_year": (
                        target_year
                    ),
                    "calibration_method": (
                        "sigmoid_platt"
                    ),
                    "calibrator_intercept": float(
                        calibrator.intercept_[
                            0
                        ]
                    ),
                    "calibrator_coefficient": float(
                        calibrator.coef_[
                            0,
                            0,
                        ]
                    ),
                    "inner_rows": len(
                        inner
                    ),
                    "inner_raw_AP": (
                        safe_average_precision(
                            inner[
                                "target_next_day"
                            ],
                            inner[
                                "raw_probability"
                            ],
                        )
                    ),
                    "inner_calibrated_AP": (
                        safe_average_precision(
                            inner[
                                "target_next_day"
                            ],
                            inner[
                                "calibrated_probability"
                            ],
                        )
                    ),
                    "inner_raw_Brier": (
                        safe_brier(
                            inner[
                                "target_next_day"
                            ],
                            inner[
                                "raw_probability"
                            ],
                        )
                    ),
                    "inner_calibrated_Brier": (
                        safe_brier(
                            inner[
                                "target_next_day"
                            ],
                            inner[
                                "calibrated_probability"
                            ],
                        )
                    ),
                }
            )

            audit_rows.append(
                {
                    "model": model_name,
                    "outer_fold_id": int(
                        outer_fold_id
                    ),
                    "target_year": (
                        target_year
                    ),
                    "number_of_counties": int(
                        county_thresholds[
                            "county"
                        ].nunique()
                    ),
                    "minimum_threshold": float(
                        county_thresholds[
                            "selected_threshold"
                        ].min()
                    ),
                    "maximum_threshold": float(
                        county_thresholds[
                            "selected_threshold"
                        ].max()
                    ),
                    "mean_threshold": float(
                        county_thresholds[
                            "selected_threshold"
                        ].mean()
                    ),
                    "pooled_fallback_count": int(
                        (
                            county_thresholds[
                                "threshold_policy"
                            ]
                            == "pooled_fallback"
                        ).sum()
                    ),
                }
            )

    predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    ).sort_values(
        [
            "model",
            "target_date",
            "county",
            "outer_fold_id",
        ]
    )

    parameters = pd.DataFrame(
        parameter_rows
    )

    county_thresholds = pd.concat(
        threshold_tables,
        ignore_index=True,
    ).sort_values(
        [
            "model",
            "outer_fold_id",
            "county",
        ]
    )

    threshold_search = pd.concat(
        search_tables,
        ignore_index=True,
    )

    threshold_sensitivity = pd.concat(
        sensitivity_tables,
        ignore_index=True,
    )

    audit = pd.DataFrame(
        audit_rows
    )

    predictions.to_csv(
        CALIBRATED_PREDICTION_FILE,
        index=False,
    )

    parameters.to_csv(
        CALIBRATION_PARAMETER_FILE,
        index=False,
    )

    county_thresholds.to_csv(
        COUNTY_THRESHOLD_FILE,
        index=False,
    )

    threshold_search.to_csv(
        THRESHOLD_SEARCH_FILE,
        index=False,
    )

    threshold_sensitivity.to_csv(
        THRESHOLD_SENSITIVITY_FILE,
        index=False,
    )

    audit.to_csv(
        CALIBRATION_AUDIT_FILE,
        index=False,
    )

    print("\n" + "=" * 82)
    print(
        "CODE 07 COMPLETED SUCCESSFULLY"
    )
    print("=" * 82)

    print(
        f"\nCalibrated predictions:\n"
        f"{CALIBRATED_PREDICTION_FILE}"
    )

    print(
        f"\nCounty thresholds:\n"
        f"{COUNTY_THRESHOLD_FILE}"
    )

    print(
        f"\nThreshold search curves:\n"
        f"{THRESHOLD_SEARCH_FILE}"
    )

    print(
        f"\nThreshold sensitivity:\n"
        f"{THRESHOLD_SENSITIVITY_FILE}"
    )


if __name__ == "__main__":
    main()
