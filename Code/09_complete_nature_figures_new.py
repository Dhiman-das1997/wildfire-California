#!/usr/bin/env python3
"""
CODE 09: Complete Nature-style OOF figures (county-threshold version)
=====================================================================

Run after:
    Code 07: 07_county_specific_threshold_selection.py
    Code 08: 08_final_metrics_county_threshold_only.py

This figure script includes the full set of visual outputs for the three base models:
1. Logistic Regression
2. XGBoost
3. LSTM
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
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

FIGURE_DIR: Path = (
    OUTPUT_DIR / "nature_figures_complete_county_threshold_only"
)

DPI: int = 600
SAVE_PDF: bool = True
SHOW_FIGURES: bool = False
CALIBRATION_BINS: int = 8

MODEL_ORDER: list[str] = [
    "logistic_regression",
    "xgboost",
    "lstm",
]

MODEL_LABELS: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
}

COUNTY_ORDER: list[str] = [
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

COUNTY_LABELS: dict[str, str] = {
    county: county.replace("_", " ")
    for county in COUNTY_ORDER
}

TARGET_YEARS: list[int] = [
    2019,
    2020,
    2021,
    2022,
    2023,
]


# ============================================================
# 2. INPUT FILES
# ============================================================

PREDICTION_FILE: Path = (
    OUTPUT_DIR / "07_calibrated_oof_predictions.csv"
)

COUNTY_THRESHOLD_FILE: Path = (
    OUTPUT_DIR / "07_county_thresholds.csv"
)

THRESHOLD_SEARCH_FILE: Path = (
    OUTPUT_DIR / "07_threshold_search.csv"
)

PANEL_METRICS_FILE: Path = (
    OUTPUT_DIR / "08_panel_metrics.csv"
)

COUNTY_METRICS_FILE: Path = (
    OUTPUT_DIR / "08_county_metrics.csv"
)

YEARLY_METRICS_FILE: Path = (
    OUTPUT_DIR / "08_yearly_metrics.csv"
)

CALIBRATION_CURVE_FILE: Path = (
    OUTPUT_DIR / "08_calibration_curves.csv"
)

THRESHOLD_SENSITIVITY_FILE: Path = (
    OUTPUT_DIR / "08_threshold_sensitivity_summary.csv"
)

FEATURE_SENSITIVITY_FILE: Path = (
    OUTPUT_DIR / "08_feature_sensitivity.csv"
)

SELECTED_COMPLEXITY_FILE: Path = (
    OUTPUT_DIR / "08_selected_model_complexity.csv"
)

LOGISTIC_COEFFICIENTS_FILE: Path = (
    OUTPUT_DIR / "04_logistic_coefficients.csv"
)

XGBOOST_IMPORTANCE_FILE: Path = (
    OUTPUT_DIR / "05_xgboost_feature_importance.csv"
)

CAUSAL_FEATURE_FILE: Path = (
    OUTPUT_DIR / "02_causal_feature_table.csv"
)


# ============================================================
# 3. STYLE AND FILE HELPERS
# ============================================================

def set_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 11.5,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.titlesize": 13.0,
            "axes.linewidth": 1.2,        
            "lines.linewidth": 2.5,       
            "lines.markersize": 6.5,      
            "axes.spines.top": False,     
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n{path}"
        )


def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = FIGURE_DIR / f"{filename}.png"

    figure.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    if SAVE_PDF:
        figure.savefig(
            FIGURE_DIR / f"{filename}.pdf",
            bbox_inches="tight",
            facecolor="white",
        )

    if SHOW_FIGURES:
        plt.show()

    plt.close(figure)
    print(f"Saved: {png_path}")


def load_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path)


def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for candidate in dataframe.columns:
        if candidate in dataframe.columns:
            return candidate
    return None


def normalize_subset_label(value) -> str:
    value_text = str(value).strip().lower()
    if value_text in {"all", "all.0"}:
        return "all"
    try:
        return str(int(float(value_text)))
    except ValueError:
        return value_text


# ============================================================
# 4. INPUT LOADING HELPER
# ============================================================

def load_inputs() -> dict[str, pd.DataFrame]:
    """Loads all required CSV evaluation outputs into a centralized dictionary."""
    print("Loading input data components...")
    return {
        "predictions": load_csv(PREDICTION_FILE),
        "thresholds": load_csv(COUNTY_THRESHOLD_FILE),
        "threshold_search": load_csv(THRESHOLD_SEARCH_FILE),
        "panel_metrics": load_csv(PANEL_METRICS_FILE),
        "county_metrics": load_csv(COUNTY_METRICS_FILE),
        "yearly_metrics": load_csv(YEARLY_METRICS_FILE),
        "calibration_curves": load_csv(CALIBRATION_CURVE_FILE),
        "threshold_sensitivity": load_csv(THRESHOLD_SENSITIVITY_FILE),
        "feature_sensitivity": load_csv(FEATURE_SENSITIVITY_FILE),
        "selected_complexity": load_csv(SELECTED_COMPLEXITY_FILE),
        "logistic_coefficients": load_csv(LOGISTIC_COEFFICIENTS_FILE),
        "xgboost_importance": load_csv(XGBOOST_IMPORTANCE_FILE),
        "causal_features": load_csv(CAUSAL_FEATURE_FILE),
    }


# ============================================================
# 5. GENERAL PERFORMANCE FIGURES
# ============================================================

def plot_panel_model_comparison(
    panel_metrics: pd.DataFrame,
) -> None:
    metric_specs = [
        ("AP", "Average Precision"),
        ("ROC_AUC", "ROC-AUC"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("specificity", "Specificity"),
        ("F1", "F1 Score"),
        ("custom_score", "Recall × Specificity"),
        ("PR_lift", "PR Lift"),
    ]

    panel = (
        panel_metrics
        .set_index("model")
        .reindex(MODEL_ORDER)
    )

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(15.0, 7.5),
    )

    for axis, (metric, label) in zip(
        axes.flatten(),
        metric_specs,
    ):
        values = panel[metric].to_numpy(dtype=float)

        bars = axis.bar(
            np.arange(len(MODEL_ORDER)),
            values,
            edgecolor="black",
            linewidth=1.0,
            color=["#4C72B0", "#DD8452", "#55A868"]
        )

        axis.set_xticks(np.arange(len(MODEL_ORDER)))
        axis.set_xticklabels(
            ["Logistic", "XGBoost", "LSTM"],
            rotation=30,
            ha="right",
            weight="bold",
        )

        axis.set_title("", weight="bold")
        axis.grid(False)

        for bar, value in zip(bars, values):
            if np.isfinite(value):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    weight="bold",
                )

    figure.tight_layout()
    save_figure(figure, "Figure_09A_panel_model_comparison")


def plot_county_performance_heatmap(
    county_metrics: pd.DataFrame,
) -> None:
    metrics = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("F1", "F1"),
        ("specificity", "Specificity"),
        ("PR_AUC_trapezoid", "PR-AUC"),
        ("ROC_AUC", "ROC-AUC"),
        ("custom_score", "R×S"),
    ]

    row_specs = [
        (
            county,
            model_name,
            f"{COUNTY_LABELS[county]} — {MODEL_LABELS[model_name]}",
        )
        for county in COUNTY_ORDER
        for model_name in MODEL_ORDER
    ]

    matrix = np.full(
        (
            len(row_specs),
            len(metrics),
        ),
        np.nan,
    )

    for row_index, (county, model_name, _) in enumerate(row_specs):
        selected = county_metrics[
            (county_metrics["county"] == county)
            & (county_metrics["model"] == model_name)
        ]

        if selected.empty:
            continue

        for column_index, (metric, _) in enumerate(metrics):
            matrix[row_index, column_index] = float(
                selected.iloc[0][metric]
            )

    figure, axis = plt.subplots(
        figsize=(12.0, 16.0)
    )

    image = axis.imshow(
        matrix,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="YlGnBu",
    )

    axis.set_xticks(np.arange(len(metrics)))
    axis.set_xticklabels(
        [label for _, label in metrics],
        rotation=30,
        ha="right",
        weight="bold",
    )

    axis.set_yticks(np.arange(len(row_specs)))
    axis.set_yticklabels(
        [label for _, _, label in row_specs],
        fontsize=8.0,
        weight="bold",
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                axis.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    weight="bold",
                    color="white" if value > 0.62 else "black",
                )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label("Metric value", weight="bold")
    axis.grid(False)
    axis.set_title("")

    figure.tight_layout()
    save_figure(figure, "Figure_09B_county_performance_heatmap")


def plot_county_bar_metric(
    county_metrics: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    figure, axes = plt.subplots(
        2,
        5,
        figsize=(18.0, 7.5),
        sharey=True,
    )

    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for axis, county in zip(
        axes.flatten(),
        COUNTY_ORDER,
    ):
        group = (
            county_metrics[
                county_metrics["county"] == county
            ]
            .set_index("model")
            .reindex(MODEL_ORDER)
        )

        values = group[metric].to_numpy(dtype=float)

        bars = axis.bar(
            np.arange(len(MODEL_ORDER)),
            values,
            edgecolor="black",
            linewidth=1.0,
            color=colors
        )

        axis.set_title(
            COUNTY_LABELS[county],
            weight="bold",
        )

        axis.set_xticks(np.arange(len(MODEL_ORDER)))
        axis.set_xticklabels(
            ["Logistic", "XGBoost", "LSTM"],
            rotation=35,
            ha="right",
            weight="bold",
        )

        if ylim is not None:
            axis.set_ylim(*ylim)

        axis.grid(False)

        for bar, value in zip(bars, values):
            if np.isfinite(value):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    weight="bold",
                )

    figure.supylabel(ylabel, weight="bold")
    figure.tight_layout()
    save_figure(figure, filename)


def plot_confusion_matrices(
    predictions: pd.DataFrame,
    model_name: str,
    filename: str,
) -> None:
    figure, axes = plt.subplots(
        2,
        5,
        figsize=(17.0, 7.5),
    )

    image = None

    for axis, county in zip(
        axes.flatten(),
        COUNTY_ORDER,
    ):
        group = predictions[
            (predictions["model"] == model_name)
            & (predictions["county"] == county)
        ]

        raw = confusion_matrix(
            group["target_next_day"],
            group["predicted_label"],
            labels=[0, 1],
        ).astype(int)

        row_totals = raw.sum(
            axis=1,
            keepdims=True,
        )

        normalized = np.divide(
            raw,
            row_totals,
            out=np.zeros_like(
                raw,
                dtype=float,
            ),
            where=row_totals != 0,
        )

        image = axis.imshow(
            normalized,
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
        )

        axis.set_title(
            COUNTY_LABELS[county],
            weight="bold",
        )

        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])
        axis.set_xticklabels(
            ["No warning", "Warning"],
            rotation=25,
            ha="right",
            weight="bold",
        )
        axis.set_yticklabels(
            ["No ignition", "Ignition"],
            weight="bold",
        )
        axis.grid(False)

        for i in range(2):
            for j in range(2):
                fraction = normalized[i, j]
                axis.text(
                    j,
                    i,
                    f"{fraction:.2f}\n(n={raw[i, j]})",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    weight="bold",
                    color="white" if fraction >= 0.55 else "black",
                )

        axis.set_xlabel("Predicted", weight="bold")
        axis.set_ylabel("Observed", weight="bold")

    if image is not None:
        colorbar = figure.colorbar(
            image,
            ax=axes.flatten().tolist(),
            fraction=0.018,
            pad=0.015,
        )
        colorbar.set_label(
            "Row-normalized fraction",
            weight="bold",
        )

    figure.tight_layout(
        rect=[0.0, 0.0, 0.97, 0.97]
    )
    save_figure(figure, filename)


def plot_panel_pr_roc_curves(
    predictions: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.5),
    )

    prevalence = float(
        predictions["target_next_day"].mean()
    )

    for model_name in MODEL_ORDER:
        group = predictions[
            predictions["model"] == model_name
        ]

        precision, recall, _ = precision_recall_curve(
            group["target_next_day"],
            group["calibrated_probability"],
        )

        ap = average_precision_score(
            group["target_next_day"],
            group["calibrated_probability"],
        )

        axes[0].plot(
            recall,
            precision,
            label=f"{MODEL_LABELS[model_name]} (AP={ap:.3f})",
        )

        fpr, tpr, _ = roc_curve(
            group["target_next_day"],
            group["calibrated_probability"],
        )

        auc_value = roc_auc_score(
            group["target_next_day"],
            group["calibrated_probability"],
        )

        axes[1].plot(
            fpr,
            tpr,
            label=f"{MODEL_LABELS[model_name]} (AUC={auc_value:.3f})",
        )

    axes[0].axhline(
        prevalence,
        linestyle="--",
        linewidth=1.5,
        color="gray",
        label=f"Prevalence={prevalence:.3f}",
    )

    axes[0].set_xlabel("Recall", weight="bold")
    axes[0].set_ylabel("Precision", weight="bold")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(False)
    axes[0].legend(frameon=False)

    axes[1].plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.5,
        color="gray",
        label="No-skill reference",
    )
    axes[1].set_xlabel("False-Positive Rate", weight="bold")
    axes[1].set_ylabel("True-Positive Rate", weight="bold")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(False)
    axes[1].legend(frameon=False)

    figure.tight_layout()
    save_figure(figure, "Figure_09G_panel_PR_ROC_curves")


def plot_county_calibration_curves(
    calibration_curves: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(
        2,
        5,
        figsize=(18.0, 7.5),
        sharex=True,
        sharey=True,
    )

    for axis, county in zip(
        axes.flatten(),
        COUNTY_ORDER,
    ):
        for model_name in MODEL_ORDER:
            group = calibration_curves[
                (calibration_curves["county"] == county)
                & (calibration_curves["model"] == model_name)
            ].sort_values("bin_id")

            if group.empty:
                continue

            axis.plot(
                group["mean_predicted_probability"],
                group["observed_event_fraction"],
                marker="o",
                label=MODEL_LABELS[model_name],
            )

        axis.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            color="gray",
            linewidth=1.5,
        )

        axis.set_title(
            COUNTY_LABELS[county],
            weight="bold",
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.grid(False)

    axes.flatten()[0].legend(
        frameon=False,
    )

    figure.supxlabel("Mean Predicted Probability", weight="bold")
    figure.supylabel("Observed Ignition Frequency", weight="bold")
    figure.tight_layout()
    save_figure(figure, "Figure_09H_county_calibration_curves")


def plot_yearly_performance(
    yearly_metrics: pd.DataFrame,
) -> None:
    metric_specs = [
        ("AP", "Average Precision"),
        ("ROC_AUC", "ROC-AUC"),
        ("recall", "Recall"),
        ("F1", "F1 Score"),
    ]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 8.5),
    )

    for axis, (metric, label) in zip(
        axes.flatten(),
        metric_specs,
    ):
        for model_name in MODEL_ORDER:
            group = yearly_metrics[
                yearly_metrics["model"] == model_name
            ].sort_values("target_year")

            axis.plot(
                group["target_year"],
                group[metric],
                marker="o",
                label=MODEL_LABELS[model_name],
            )

        axis.set_xlabel("Target Year", weight="bold")
        axis.set_ylabel(label, weight="bold")
        axis.set_xticks(TARGET_YEARS)
        axis.grid(False)

    axes.flatten()[0].legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, "Figure_09I_yearly_OOF_performance")


# ============================================================
# 6. COUNTY THRESHOLD FIGURES
# ============================================================

def plot_threshold_by_county_model_year(
    thresholds: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(
        2,
        5,
        figsize=(18.0, 7.5),
        sharex=True,
        sharey=True,
    )

    threshold_min = float(
        thresholds["selected_threshold"].min()
    )
    threshold_max = float(
        thresholds["selected_threshold"].max()
    )

    margin = max(
        0.005,
        0.10 * (threshold_max - threshold_min),
    )

    lower_limit = max(0.0, threshold_min - margin)
    upper_limit = min(1.0, threshold_max + margin)

    if upper_limit <= lower_limit:
        upper_limit = lower_limit + 0.05

    for axis, county in zip(
        axes.flatten(),
        COUNTY_ORDER,
    ):
        county_data = thresholds[
            thresholds["county"] == county
        ]

        for model_name in MODEL_ORDER:
            group = county_data[
                county_data["model"] == model_name
            ].sort_values("target_year")

            axis.plot(
                group["target_year"],
                group["selected_threshold"],
                marker="o",
                label=MODEL_LABELS[model_name],
            )

            for _, row in group.iterrows():
                axis.text(
                    row["target_year"],
                    row["selected_threshold"],
                    f"{row['selected_threshold']:.3f}",
                    fontsize=8,
                    weight="bold",
                    ha="center",
                    va="bottom",
                )

        axis.set_title(
            COUNTY_LABELS[county],
            weight="bold",
        )
        axis.set_xticks(TARGET_YEARS)
        axis.set_ylim(lower_limit, upper_limit)
        axis.grid(False)

    axes.flatten()[0].legend(
        frameon=False,
    )

    figure.supxlabel("Outer Target Year", weight="bold")
    figure.supylabel("Selected County Threshold", weight="bold")
    figure.tight_layout()
    save_figure(figure, "Figure_09L_county_threshold_by_model_year")


def plot_mean_threshold_heatmap(
    thresholds: pd.DataFrame,
) -> None:
    mean_table = (
        thresholds
        .groupby(
            ["county", "model"],
            as_index=False,
        )
        .agg(
            mean_threshold=("selected_threshold", "mean")
        )
    )

    matrix = np.full(
        (
            len(COUNTY_ORDER),
            len(MODEL_ORDER),
        ),
        np.nan,
    )

    for i, county in enumerate(COUNTY_ORDER):
        for j, model_name in enumerate(MODEL_ORDER):
            selected = mean_table[
                (mean_table["county"] == county)
                & (mean_table["model"] == model_name)
            ]

            if not selected.empty:
                matrix[i, j] = float(
                    selected.iloc[0]["mean_threshold"]
                )

    finite_values = matrix[np.isfinite(matrix)]
    value_min = float(finite_values.min())
    value_max = float(finite_values.max())

    padding = max(
        0.002,
        0.10 * (value_max - value_min),
    )

    color_min = max(0.0, value_min - padding)
    color_max = min(1.0, value_max + padding)

    if color_max <= color_min:
        color_max = color_min + 0.01

    figure, axis = plt.subplots(
        figsize=(8.5, 7.5)
    )

    image = axis.imshow(
        matrix,
        aspect="auto",
        vmin=color_min,
        vmax=color_max,
        cmap="YlGnBu",
    )

    axis.set_xticks(np.arange(len(MODEL_ORDER)))
    axis.set_xticklabels(
        ["Logistic", "XGBoost", "LSTM"],
        rotation=25,
        ha="right",
        weight="bold",
    )

    axis.set_yticks(np.arange(len(COUNTY_ORDER)))
    axis.set_yticklabels(
        [COUNTY_LABELS[county] for county in COUNTY_ORDER],
        weight="bold",
    )
    axis.grid(False)

    midpoint = (color_min + color_max) / 2.0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                axis.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    weight="bold",
                    color="white" if value > midpoint else "black",
                )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Mean Selected Threshold", weight="bold")

    figure.tight_layout()
    save_figure(figure, "Figure_09M_mean_county_model_threshold")


# ============================================================
# 7. THRESHOLD-SELECTION CURVES
# ============================================================

def plot_threshold_selection_curves(
    threshold_search: pd.DataFrame,
    thresholds: pd.DataFrame,
    model_name: str,
    filename: str,
) -> None:
    model_search = threshold_search[
        (threshold_search["model"] == model_name)
        & (threshold_search["search_scope"] == "county")
    ].copy()

    model_thresholds = thresholds[
        thresholds["model"] == model_name
    ].copy()

    figure, axes = plt.subplots(
        2,
        5,
        figsize=(18.0, 7.5),
        sharex=True,
        sharey=True,
    )

    for axis, county in zip(
        axes.flatten(),
        COUNTY_ORDER,
    ):
        county_search = model_search[
            model_search["county"] == county
        ]

        summary = (
            county_search
            .groupby(
                "threshold",
                as_index=False,
            )
            .agg(
                mean_score=("custom_score", "mean"),
                mean_recall=("recall", "mean"),
                mean_specificity=("specificity", "mean"),
            )
        )

        axis.plot(
            summary["threshold"],
            summary["mean_score"],
            label="Recall × Specificity",
        )

        axis.plot(
            summary["threshold"],
            summary["mean_recall"],
            linestyle="--",
            label="Recall",
        )

        axis.plot(
            summary["threshold"],
            summary["mean_specificity"],
            linestyle=":",
            label="Specificity",
        )

        mean_selected = float(
            model_thresholds[
                model_thresholds["county"] == county
            ]["selected_threshold"].mean()
        )

        axis.axvline(
            mean_selected,
            linestyle="-.",
            linewidth=1.5,
            color="red",
            label=f"Mean selected={mean_selected:.3f}",
        )

        axis.set_title(
            COUNTY_LABELS[county],
            weight="bold",
        )
        axis.grid(False)

    axes.flatten()[0].legend(
        frameon=False,
    )

    figure.supxlabel("Candidate Operational Threshold", weight="bold")
    figure.supylabel("Mean Inner-OOF Performance Metric", weight="bold")
    figure.tight_layout()
    save_figure(figure, filename)


# ============================================================
# 8. THRESHOLD SENSITIVITY
# ============================================================

def plot_threshold_sensitivity(
    sensitivity: pd.DataFrame,
    model_name: str,
    filename: str,
) -> None:
    model_data = sensitivity[
        sensitivity["model"] == model_name
    ]

    figure, axes = plt.subplots(
        2,
        5,
        figsize=(18.0, 7.5),
        sharex=True,
    )

    for axis, county in zip(
        axes.flatten(),
        COUNTY_ORDER,
    ):
        county_data = model_data[
            model_data["county"] == county
        ].sort_values("threshold_delta")

        axis.plot(
            county_data["threshold_delta"],
            county_data["mean_custom_score"],
            marker="o",
            label="Recall × Specificity",
        )

        axis.plot(
            county_data["threshold_delta"],
            county_data["mean_warning_rate"],
            marker="s",
            linestyle="--",
            label="Warning Rate",
        )

        axis.axvline(
            0.0,
            linestyle=":",
            color="gray",
            linewidth=1.5,
        )

        axis.set_title(
            COUNTY_LABELS[county],
            weight="bold",
        )
        axis.grid(False)

    axes.flatten()[0].legend(
        frameon=False,
    )

    figure.supxlabel("Decision Threshold Shifts from Optimal Calibration Point ($\Delta$)", weight="bold")
    figure.supylabel("Mean Objective Scores", weight="bold")
    figure.tight_layout()
    save_figure(figure, filename)


# ============================================================
# 9. CONSOLIDATED FEATURE SENSITIVITY
# ============================================================

def plot_consolidated_feature_sensitivity(
    feature_sensitivity: pd.DataFrame,
    filename: str = "Figure_09T_U_V_consolidated_sensitivity",
) -> None:
    """
    Combines Figures 09T, 09U, and 09V into a single multi-panel horizontal standard
    subfigure layout with enhanced line properties and maximum axis label/tick legibility.
    """
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(22.0, 6.8),
    )

    # Panel A: Logistic Regression Sensitivity
    ax_log = axes[0]
    log_data = feature_sensitivity[
        (feature_sensitivity["model"] == "logistic_regression")
        & (feature_sensitivity["sensitivity_type"] == "weather_feature_subset")
    ].copy()

    if not log_data.empty:
        for fold_id, group in log_data.groupby("outer_fold_or_year"):
            group = group.copy()
            order_map = {"10": 0, "15": 1, "20": 2, "all": 3}
            group["complexity_label"] = group["complexity_value"].astype(str).map(normalize_subset_label)
            group["plot_order"] = group["complexity_label"].map(order_map)
            group = group.sort_values("plot_order")
            fold_summary = group.groupby("complexity_label", as_index=False)["inner_AP"].mean()
            fold_summary["plot_order"] = fold_summary["complexity_label"].map(order_map)
            fold_summary = fold_summary.sort_values("plot_order")
            
            ax_log.plot(fold_summary["complexity_label"], fold_summary["inner_AP"], marker="o", label=f"Fold {fold_id}")
        
        ax_log.set_xlabel("Number of Weather Predictors", weight="bold", fontsize=15.0)
        ax_log.set_ylabel("Mean Inner-Validation AP", weight="bold", fontsize=15.0)
        ax_log.tick_params(axis="both", labelsize=12.5)
        for tick in ax_log.get_xticklabels() + ax_log.get_yticklabels():
            tick.set_weight("bold")
        ax_log.grid(False)
        ax_log.legend(frameon=False, ncol=2, fontsize=10.5)
        ax_log.text(-0.05, 1.03, "(a)", transform=ax_log.transAxes, fontsize=16, weight="bold", va="bottom", ha="left")

    # Panel B: XGBoost Sensitivity
    ax_xgb = axes[1]
    xgb_data = feature_sensitivity[
        (feature_sensitivity["model"] == "xgboost")
        & (feature_sensitivity["sensitivity_type"] == "weather_feature_subset")
    ].copy()

    if not xgb_data.empty:
        for fold_id, group in xgb_data.groupby("outer_fold_or_year"):
            group = group.copy()
            order_map = {"10": 0, "15": 1, "20": 2, "all": 3}
            group["complexity_label"] = group["complexity_value"].astype(str).map(normalize_subset_label)
            group["plot_order"] = group["complexity_label"].map(order_map)
            group = group.sort_values("plot_order")
            fold_summary = group.groupby("complexity_label", as_index=False)["inner_AP"].mean()
            fold_summary["plot_order"] = fold_summary["complexity_label"].map(order_map)
            fold_summary = fold_summary.sort_values("plot_order")
            
            ax_xgb.plot(fold_summary["complexity_label"], fold_summary["inner_AP"], marker="o", label=f"Fold {fold_id}")
        
        ax_xgb.set_xlabel("Number of Weather Predictors", weight="bold", fontsize=15.0)
        ax_xgb.tick_params(axis="both", labelsize=12.5)
        for tick in ax_xgb.get_xticklabels() + ax_xgb.get_yticklabels():
            tick.set_weight("bold")
        ax_xgb.grid(False)
        ax_xgb.legend(frameon=False, ncol=2, fontsize=10.5)
        ax_xgb.text(-0.05, 1.03, "(b)", transform=ax_xgb.transAxes, fontsize=16, weight="bold", va="bottom", ha="left")

    # Panel C: LSTM Sensitivity
    ax_lstm = axes[2]
    lstm_data = feature_sensitivity[
        (feature_sensitivity["model"] == "lstm")
        & (feature_sensitivity["sensitivity_type"] == "lookback_days")
    ].copy()

    if not lstm_data.empty:
        for fold_id, group in lstm_data.groupby("outer_fold_or_year"):
            group = group.copy()
            group["complexity_numeric"] = pd.to_numeric(group["complexity_value"], errors="coerce")
            group = group.sort_values("complexity_numeric")
            fold_summary = group.groupby("complexity_numeric", as_index=False)["inner_AP"].mean()
            
            ax_lstm.plot(fold_summary["complexity_numeric"], fold_summary["inner_AP"], marker="o", label=f"Fold {fold_id}")
        
        ax_lstm.set_xlabel("Lookback Length (Days)", weight="bold", fontsize=15.0)
        ax_lstm.tick_params(axis="both", labelsize=12.5)
        for tick in ax_lstm.get_xticklabels() + ax_lstm.get_yticklabels():
            tick.set_weight("bold")
        ax_lstm.grid(False)
        ax_lstm.legend(frameon=False, ncol=2, fontsize=10.5)
        ax_lstm.text(-0.05, 1.03, "(c)", transform=ax_lstm.transAxes, fontsize=16, weight="bold", va="bottom", ha="left")

    figure.tight_layout()
    save_figure(figure, filename)


def plot_selected_complexity(
    selected_complexity: pd.DataFrame,
) -> None:
    if selected_complexity.empty:
        print("Skipped selected complexity figure: table is empty.")
        return

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.5),
    )

    # Left: Weather subset
    left_data = selected_complexity[
        selected_complexity["complexity_type"] == "weather_feature_subset"
    ].copy()

    if not left_data.empty:
        left_data["selected_numeric"] = left_data["selected_value"].map(
            lambda x: 25 if str(x).lower() == "all" else float(x)
        )

        for model_name in [
            "logistic_regression",
            "xgboost",
        ]:
            group = left_data[
                left_data["model"] == model_name
            ].sort_values("target_year")

            axes[0].plot(
                group["target_year"],
                group["selected_numeric"],
                marker="o",
                label=MODEL_LABELS[model_name],
            )

        axes[0].set_xticks(TARGET_YEARS)
        axes[0].set_yticks([10, 15, 20, 25])
        axes[0].set_yticklabels(["10", "15", "20", "all"], weight="bold")
        axes[0].set_xlabel("Target Year", weight="bold")
        axes[0].set_ylabel("Selected Weather Subset Size", weight="bold")
        axes[0].set_title("Optimal Feature Dimensions Across Test Years", weight="bold")
        axes[0].grid(False)
        axes[0].legend(frameon=False)

    # Right: LSTM lookback
    right_data = selected_complexity[
        selected_complexity["complexity_type"] == "lookback_days"
    ].copy()

    if not right_data.empty:
        right_data["selected_numeric"] = pd.to_numeric(
            right_data["selected_value"],
            errors="coerce",
        )

        group = right_data[
            right_data["model"] == "lstm"
        ].sort_values("target_year")

        axes[1].plot(
            group["target_year"],
            group["selected_numeric"],
            marker="o",
            color="#55A868",
            label="LSTM",
        )

        axes[1].set_xticks(TARGET_YEARS)
        axes[1].set_xlabel("Target Year", weight="bold")
        axes[1].set_ylabel("Lookback Window Size (Days)", weight="bold")
        axes[1].set_title("Optimal Memory Depth", weight="bold")
        axes[1].grid(False)
        axes[1].legend(frameon=False)

    figure.suptitle(
        "Data Complexity Parameter Evolution Across Cross-Validation Epochs",
        weight="bold",
    )
    figure.tight_layout()
    save_figure(figure, "Figure_09W_selected_model_complexity")


# ============================================================
# 10. FEATURE IMPORTANCE RANKING (Figure 09X)
# ============================================================

def _lstm_permutation_importance(
    predictions: pd.DataFrame,
    causal_features: pd.DataFrame,
    top_n: int,
    n_repeats: int,
    random_state: int,
) -> pd.Series:
    rng = np.random.default_rng(random_state)

    lstm_preds = predictions[
        predictions["model"] == "lstm"
    ].copy().reset_index(drop=True)

    if lstm_preds.empty:
        return pd.Series(dtype=float)

    feature_cols = [
        col for col in causal_features.columns
        if col not in {
            "date", "county", "target_next_day", "target_date",
            "time", "fire_ignition", "incident_count",
        }
        and causal_features[col].dtype.kind in "fiu"
    ]

    present_in_preds = [
        col for col in feature_cols
        if col in lstm_preds.columns
    ]

    if not present_in_preds:
        merge_keys = [
            k for k in ["date", "county"]
            if k in lstm_preds.columns and k in causal_features.columns
        ]
        if merge_keys:
            causal_subset = causal_features[
                merge_keys + feature_cols
            ].copy()
            if "date" in merge_keys:
                causal_subset["date"] = pd.to_datetime(
                    causal_subset["date"], errors="coerce"
                )
                lstm_preds["date"] = pd.to_datetime(
                    lstm_preds.get("date", lstm_preds.get("target_date")),
                    errors="coerce",
                )
            lstm_preds = lstm_preds.merge(
                causal_subset,
                on=merge_keys,
                how="left",
            )
            present_in_preds = [
                col for col in feature_cols
                if col in lstm_preds.columns
            ]

    if not present_in_preds:
        return pd.Series(dtype=float)

    y_true = lstm_preds["target_next_day"].to_numpy()
    y_prob_base = lstm_preds["calibrated_probability"].to_numpy()
    baseline_ap = average_precision_score(y_true, y_prob_base)

    importance_dict: dict[str, float] = {}

    for col in present_in_preds:
        original_values = lstm_preds[col].to_numpy(dtype=float).copy()
        drops: list[float] = []

        for _ in range(n_repeats):
            shuffled = original_values.copy()
            rng.shuffle(shuffled)
            lstm_preds[col] = shuffled

            corr_original = float(
                pd.Series(original_values).corr(
                    pd.Series(y_prob_base), method="spearman"
                )
            )
            corr_shuffled = float(
                pd.Series(shuffled).corr(
                    pd.Series(y_prob_base), method="spearman"
                )
            )
            drop = max(0.0, abs(corr_original) - abs(corr_shuffled))
            drops.append(drop * baseline_ap)

        lstm_preds[col] = original_values
        importance_dict[col] = float(np.mean(drops))

    importance = pd.Series(importance_dict).sort_values(ascending=False)
    return importance.head(top_n)


def plot_feature_importance_ranking(
    logistic_coefficients: pd.DataFrame,
    xgboost_importance: pd.DataFrame,
    predictions: pd.DataFrame,
    causal_features: pd.DataFrame,
    top_n: int = 15,
    n_lstm_repeats: int = 10,
    random_state: int = 42,
) -> None:
    # Logistic
    logistic_coef = logistic_coefficients[
        logistic_coefficients["predictor"] != "intercept"
    ].copy()

    logistic_summary = (
        logistic_coef
        .groupby("predictor")["absolute_standardized_coefficient"]
        .agg(mean_importance="mean", std_importance="std")
        .reset_index()
        .sort_values("mean_importance", ascending=False)
        .head(top_n)
    )
    logistic_summary["std_importance"] = logistic_summary["std_importance"].fillna(0.0)

    # XGBoost
    xgb_summary = (
        xgboost_importance
        .groupby("predictor")["importance_value"]
        .agg(mean_importance="mean", std_importance="std")
        .reset_index()
        .sort_values("mean_importance", ascending=False)
        .head(top_n)
    )
    xgb_summary["std_importance"] = xgb_summary["std_importance"].fillna(0.0)

    # LSTM
    lstm_importance = _lstm_permutation_importance(
        predictions=predictions,
        causal_features=causal_features,
        top_n=top_n,
        n_repeats=n_lstm_repeats,
        random_state=random_state,
    )

    n_panels = 3 if not lstm_importance.empty else 2
    figure, axes = plt.subplots(
        1,
        n_panels,
        figsize=(7.0 * n_panels, max(6.5, top_n * 0.5 + 2.5)),
    )

    axes = list(axes) if n_panels > 1 else [axes]

    color_logistic = "#4C72B0"
    color_xgb = "#DD8452"
    color_lstm = "#55A868"

    # ---- Panel 1: Logistic Regression ----
    ax_log = axes[0]
    y_pos = np.arange(len(logistic_summary))[::-1]

    ax_log.barh(
        y_pos,
        logistic_summary["mean_importance"].to_numpy(),
        xerr=logistic_summary["std_importance"].to_numpy(),
        color=color_logistic,
        edgecolor="black",
        linewidth=0.8,
        error_kw={"elinewidth": 1.0, "capsize": 3.0},
        height=0.65,
    )

    ax_log.set_yticks(y_pos)
    ax_log.set_yticklabels(
        logistic_summary["predictor"].tolist(),
        fontsize=11.5,             
        weight="bold",
    )
    ax_log.set_xlabel("Mean |Standardized Coefficient|", weight="bold", fontsize=13.5)
    ax_log.tick_params(axis="x", labelsize=11.5)
    ax_log.set_title("", weight="bold")
    ax_log.grid(False)
    ax_log.text(-0.05, 1.03, "(a)", transform=ax_log.transAxes, fontsize=14, weight="bold", va="bottom", ha="left")

    for rank, (bar_y, imp) in enumerate(
        zip(y_pos, logistic_summary["mean_importance"].tolist()), 1
    ):
        ax_log.text(
            imp,
            bar_y,
            f"   #{rank}",
            va="center",
            fontsize=10.0,
            weight="bold",
            color="black",
        )

    # ---- Panel 2: XGBoost ----
    ax_xgb = axes[1]
    y_pos_x = np.arange(len(xgb_summary))[::-1]

    ax_xgb.barh(
        y_pos_x,
        xgb_summary["mean_importance"].to_numpy(),
        xerr=xgb_summary["std_importance"].to_numpy(),
        color=color_xgb,
        edgecolor="black",
        linewidth=0.8,
        error_kw={"elinewidth": 1.0, "capsize": 3.0},
        height=0.65,
    )

    ax_xgb.set_yticks(y_pos_x)
    ax_xgb.set_yticklabels(
        xgb_summary["predictor"].tolist(),
        fontsize=11.5,             
        weight="bold",
    )
    ax_xgb.set_xlabel("Mean Gain Importance", weight="bold", fontsize=13.5)
    ax_xgb.tick_params(axis="x", labelsize=11.5)
    ax_xgb.set_title("", weight="bold")
    ax_xgb.grid(False)
    ax_xgb.text(-0.05, 1.03, "(b)", transform=ax_xgb.transAxes, fontsize=14, weight="bold", va="bottom", ha="left")

    for rank, (bar_y, imp) in enumerate(
        zip(y_pos_x, xgb_summary["mean_importance"].tolist()), 1
    ):
        ax_xgb.text(
            imp,
            bar_y,
            f"   #{rank}",
            va="center",
            fontsize=10.0,
            weight="bold",
            color="black",
        )

    # ---- Panel 3: LSTM ----
    if not lstm_importance.empty and n_panels == 3:
        ax_lstm = axes[2]
        y_pos_l = np.arange(len(lstm_importance))[::-1]

        ax_lstm.barh(
            y_pos_l,
            lstm_importance.to_numpy(),
            color=color_lstm,
            edgecolor="black",
            linewidth=0.8,
            height=0.65,
        )

        ax_lstm.set_yticks(y_pos_l)
        ax_lstm.set_yticklabels(
            lstm_importance.index.tolist(),
            fontsize=11.5,         
            weight="bold",
        )
        ax_lstm.set_xlabel("Proxy Permutation Importance ($\Delta$ AP)", weight="bold", fontsize=13.5)
        ax_lstm.tick_params(axis="x", labelsize=11.5)
        ax_lstm.set_title("", weight="bold")
        ax_lstm.grid(False)
        ax_lstm.text(-0.05, 1.03, "(c)", transform=ax_lstm.transAxes, fontsize=14, weight="bold", va="bottom", ha="left")

        for rank, (bar_y, imp) in enumerate(
            zip(y_pos_l, lstm_importance.tolist()), 1
        ):
            ax_lstm.text(
                imp,
                bar_y,
                f"   #{rank}",
                va="center",
                fontsize=10.0,
                weight="bold",
                color="black",
            )

    figure.tight_layout()
    save_figure(figure, "Figure_09X_feature_importance_ranking")


# ============================================================
# 11. MAIN RUNTIME
# ============================================================

def main() -> None:
    print("\n" + "=" * 82)
    print("CODE 09: COMPLETE NATURE-STYLE COUNTY-THRESHOLD FIGURES")
    print("=" * 82)

    set_nature_style()

    data = load_inputs()

    predictions = data["predictions"]
    thresholds = data["thresholds"]
    threshold_search = data["threshold_search"]
    panel_metrics = data["panel_metrics"]
    county_metrics = data["county_metrics"]
    yearly_metrics = data["yearly_metrics"]
    calibration_curves = data["calibration_curves"]
    threshold_sensitivity = data["threshold_sensitivity"]
    feature_sensitivity = data["feature_sensitivity"]
    selected_complexity = data["selected_complexity"]
    logistic_coefficients = data["logistic_coefficients"]
    xgboost_importance = data["xgboost_importance"]
    causal_features = data["causal_features"]

    print("\n1. Plotting model-performance figures...")
    plot_panel_model_comparison(panel_metrics)
    plot_county_performance_heatmap(county_metrics)

    plot_county_bar_metric(
        county_metrics=county_metrics,
        metric="custom_score",
        ylabel="Recall × Specificity",
        title="",  
        filename="Figure_09C_county_warning_score",
        ylim=(0.0, 1.0),
    )

    plot_confusion_matrices(predictions, "logistic_regression", "Figure_09D_logistic_county_confusion_matrices")
    plot_confusion_matrices(predictions, "xgboost", "Figure_09E_xgboost_county_confusion_matrices")
    plot_confusion_matrices(predictions, "lstm", "Figure_09F_lstm_county_confusion_matrices")

    plot_panel_pr_roc_curves(predictions)
    plot_county_calibration_curves(calibration_curves)
    plot_yearly_performance(yearly_metrics)

    plot_county_bar_metric(
        county_metrics=county_metrics,
        metric="PR_lift",
        ylabel="PR Lift Over Baseline Prevalence",
        title="",  
        filename="Figure_09J_county_PR_lift",
    )

    plot_county_bar_metric(
        county_metrics=county_metrics,
        metric="false_alarms_per_detected_ignition",
        ylabel="False Alarms per Detected Ignition Event",
        title="",  
        filename="Figure_09K_false_alarm_burden",
    )

    print("2. Plotting threshold validation trends...")
    plot_threshold_by_county_model_year(thresholds)
    plot_mean_threshold_heatmap(thresholds)

    plot_threshold_selection_curves(threshold_search, thresholds, "logistic_regression", "Figure_09N_logistic_threshold_selection")
    plot_threshold_selection_curves(threshold_search, thresholds, "xgboost", "Figure_09O_xgboost_threshold_selection")
    plot_threshold_selection_curves(threshold_search, thresholds, "lstm", "Figure_09P_lstm_threshold_selection")

    plot_threshold_sensitivity(threshold_sensitivity, "logistic_regression", "Figure_09Q_logistic_threshold_sensitivity")
    plot_threshold_sensitivity(threshold_sensitivity, "xgboost", "Figure_09R_xgboost_threshold_sensitivity")
    plot_threshold_sensitivity(threshold_sensitivity, "lstm", "Figure_09S_lstm_threshold_sensitivity")

    print("3. Plotting dynamic complexity feature-sensitivity curves (Consolidated)...")
    plot_consolidated_feature_sensitivity(feature_sensitivity)

    plot_selected_complexity(selected_complexity)

    print("4. Plotting standardized structural feature importance matrices...")
    plot_feature_importance_ranking(
        logistic_coefficients=logistic_coefficients,
        xgboost_importance=xgboost_importance,
        predictions=predictions,
        causal_features=causal_features,
        top_n=15,
        n_lstm_repeats=10,
        random_state=42,
    )

    print("\n" + "=" * 82)
    print("CODE 09 COMPLETED SUCCESSFULLY")
    print("=" * 82)
    print(f"\nAll publication graphics exported directly to:\n{FIGURE_DIR}")


if __name__ == "__main__":
    main()
