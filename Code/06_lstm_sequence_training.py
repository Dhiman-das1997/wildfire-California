#!/usr/bin/env python3
"""
CODE 06: LSTM sequence preparation and training
================================================

Run this script after Codes 01-05.

Inputs
------
1. 02_causal_feature_table.csv
2. 03_outer_fold_registry.csv
3. 03_inner_fold_registry.csv

Important
---------
The LSTM is trained independently of Logistic Regression and XGBoost.
It uses the same leakage-free temporal folds.

For every county and issue date t, the LSTM receives an ordered weather
sequence:

    [t-L+1, ..., t]

with the three raw weather channels:

    t2m_max_C, rh_min, wind_max

The model predicts wildfire ignition on target date t+1.

Seasonality variables and county identity are not repeated through the
sequence. They are appended to the final LSTM hidden state before the
output layer.

Leakage-control rules
---------------------
1. Sequences never cross county boundaries.
2. Sequences require consecutive daily dates.
3. Scaling is fitted on training sequences only.
4. Class weights are calculated from training labels only.
5. Hyperparameters and epoch count are selected using inner folds only.
6. Outer-test labels are used only after outer-test probabilities are frozen.

Outputs
-------
06_lstm_oof_raw_predictions.csv
06_lstm_hyperparameters.csv
06_lstm_inner_validation_results.csv
06_lstm_training_history.csv
06_lstm_fold_audit.csv
"""

from __future__ import annotations

from pathlib import Path
import copy
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as error:
    raise ImportError(
        "PyTorch is not installed.\n"
        "Install it using:\n"
        "    pip install torch"
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
    OUTPUT_DIR / "06_lstm_oof_raw_predictions.csv"
)

HYPERPARAMETER_FILE: Path = (
    OUTPUT_DIR / "06_lstm_hyperparameters.csv"
)

INNER_RESULT_FILE: Path = (
    OUTPUT_DIR / "06_lstm_inner_validation_results.csv"
)


INNER_OOF_PREDICTION_FILE: Path = (
    OUTPUT_DIR / "06_lstm_inner_oof_predictions.csv"
)

TRAINING_HISTORY_FILE: Path = (
    OUTPUT_DIR / "06_lstm_training_history.csv"
)

AUDIT_FILE: Path = (
    OUTPUT_DIR / "06_lstm_fold_audit.csv"
)


# ============================================================
# 3. REPRODUCIBILITY AND DEVICE
# ============================================================

RANDOM_SEED: int = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Small recurrent models are usually faster and more stable when
# excessive CPU thread use is avoided.
torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# 4. DATA SETTINGS
# ============================================================

SEQUENCE_COLUMNS: list[str] = [
    "t2m_max_C",
    "rh_min",
    "wind_max",
]

STATIC_COLUMNS: list[str] = [
    "doy_sin",
    "doy_cos",
]

REFERENCE_COUNTY: str = "Butte"

REQUIRED_COLUMNS: list[str] = [
    "county",
    "date",
    "target_date",
    "target_next_day",
    "t2m_max_C",
    "rh_min",
    "wind_max",
    "doy_sin",
    "doy_cos",
]


# ============================================================
# 5. LSTM SETTINGS
# ============================================================

BATCH_SIZE: int = 256

LEARNING_RATE: float = 0.001

EARLY_STOPPING_PATIENCE: int = 5

MAX_EPOCHS_INNER: int = 40

MINIMUM_EPOCHS: int = 3

AP_TIE_TOLERANCE: float = 0.005

# Beginner-friendly compact grid.
# Increase the grid later only after the complete pipeline works.
LSTM_PARAMETER_GRID: list[dict] = [
    {
        "lookback": 7,
        "hidden_dim": 32,
        "dropout": 0.10,
        "weight_decay": 0.0,
    },
    {
        "lookback": 14,
        "hidden_dim": 32,
        "dropout": 0.20,
        "weight_decay": 1e-5,
    },
    {
        "lookback": 30,
        "hidden_dim": 32,
        "dropout": 0.20,
        "weight_decay": 1e-5,
    },
    {
        "lookback": 14,
        "hidden_dim": 64,
        "dropout": 0.20,
        "weight_decay": 1e-5,
    },
]


# ============================================================
# 6. REPRODUCIBILITY FUNCTION
# ============================================================

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 7. BASIC VALIDATION
# ============================================================

def check_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """Stop when an essential column is missing."""

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

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    if np.unique(y_true).size < 2:
        return np.nan

    return float(
        average_precision_score(
            y_true,
            probability,
        )
    )


# ============================================================
# 8. ADD COUNTY FIXED EFFECTS
# ============================================================

def add_county_fixed_effects(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Add predetermined one-hot county columns.

    Butte is used as the reference county.
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
        ).astype(float)

        county_columns.append(
            column_name
        )

    return dataframe, county_columns


# ============================================================
# 9. CHECK DAILY CONTINUITY
# ============================================================

def check_daily_continuity(
    dataframe: pd.DataFrame,
) -> None:
    """
    Verify that each county has one consecutive daily time series.
    """

    for county, county_data in dataframe.groupby("county"):

        county_data = county_data.sort_values(
            "date"
        )

        duplicate_dates = int(
            county_data["date"]
            .duplicated()
            .sum()
        )

        if duplicate_dates > 0:
            raise ValueError(
                f"{county} contains {duplicate_dates} duplicate dates."
            )

        date_difference = (
            county_data["date"]
            .diff()
            .dropna()
        )

        non_daily_gaps = int(
            (date_difference != pd.Timedelta(days=1))
            .sum()
        )

        if non_daily_gaps > 0:
            raise ValueError(
                f"{county} contains {non_daily_gaps} non-daily gaps. "
                "An LSTM sequence must not cross missing dates."
            )


# ============================================================
# 10. BUILD ALL POSSIBLE SEQUENCES
# ============================================================

def build_sequences(
    dataframe: pd.DataFrame,
    lookback: int,
    static_columns: list[str],
) -> dict:
    """
    Build county-specific LSTM sequences.

    The endpoint of every sequence is issue date t.

    Sequence:
        t-lookback+1, ..., t

    Target:
        target_next_day at t, representing ignition on t+1.
    """

    sequence_values: list[np.ndarray] = []
    static_values: list[np.ndarray] = []
    labels: list[int] = []

    counties: list[str] = []
    issue_dates: list[pd.Timestamp] = []
    target_dates: list[pd.Timestamp] = []

    for county, county_data in dataframe.groupby(
        "county",
        sort=False,
    ):

        county_data = (
            county_data
            .sort_values("date")
            .reset_index(drop=True)
        )

        raw_weather = county_data[
            SEQUENCE_COLUMNS
        ].to_numpy(dtype=np.float32)

        static_data = county_data[
            static_columns
        ].to_numpy(dtype=np.float32)

        county_labels = county_data[
            "target_next_day"
        ].to_numpy(dtype=np.int64)

        county_issue_dates = county_data[
            "date"
        ].to_numpy()

        county_target_dates = county_data[
            "target_date"
        ].to_numpy()

        for endpoint in range(
            lookback - 1,
            len(county_data),
        ):

            startpoint = endpoint - lookback + 1

            sequence = raw_weather[
                startpoint : endpoint + 1
            ]

            if np.isnan(sequence).any():
                continue

            sequence_values.append(
                sequence
            )

            static_values.append(
                static_data[endpoint]
            )

            labels.append(
                int(county_labels[endpoint])
            )

            counties.append(
                county
            )

            issue_dates.append(
                pd.Timestamp(
                    county_issue_dates[endpoint]
                )
            )

            target_dates.append(
                pd.Timestamp(
                    county_target_dates[endpoint]
                )
            )

    if not sequence_values:
        raise ValueError(
            f"No valid sequences were created for lookback={lookback}."
        )

    return {
        "X_sequence": np.asarray(
            sequence_values,
            dtype=np.float32,
        ),
        "X_static": np.asarray(
            static_values,
            dtype=np.float32,
        ),
        "y": np.asarray(
            labels,
            dtype=np.int64,
        ),
        "county": np.asarray(
            counties,
            dtype=object,
        ),
        "issue_date": np.asarray(
            issue_dates,
            dtype="datetime64[ns]",
        ),
        "target_date": np.asarray(
            target_dates,
            dtype="datetime64[ns]",
        ),
    }


# ============================================================
# 11. SELECT SEQUENCES BY ISSUE-DATE RANGE
# ============================================================

def select_sequence_indices(
    sequence_data: dict,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> np.ndarray:
    """Return sequence endpoints whose issue dates lie in the interval."""

    issue_dates = pd.to_datetime(
        sequence_data["issue_date"]
    )

    mask = (
        (issue_dates >= start_date)
        & (issue_dates <= end_date)
    )

    return np.flatnonzero(
        mask
    )


# ============================================================
# 12. FIT TRAINING-ONLY SCALERS
# ============================================================

def fit_sequence_scaler(
    X_sequence_train: np.ndarray,
) -> StandardScaler:
    """
    Fit one scaler for the raw weather channels using training sequences only.

    The sequence dimensions are flattened temporarily:

        samples x lookback x channels
        -> all training time steps x channels
    """

    number_of_channels = (
        X_sequence_train.shape[2]
    )

    flattened_training_values = (
        X_sequence_train
        .reshape(
            -1,
            number_of_channels,
        )
    )

    scaler = StandardScaler()

    scaler.fit(
        flattened_training_values
    )

    return scaler


def transform_sequences(
    X_sequence: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """Apply the fitted weather scaler to a sequence tensor."""

    number_of_samples = X_sequence.shape[0]
    lookback = X_sequence.shape[1]
    number_of_channels = X_sequence.shape[2]

    flattened_values = X_sequence.reshape(
        -1,
        number_of_channels,
    )

    transformed_values = scaler.transform(
        flattened_values
    )

    return transformed_values.reshape(
        number_of_samples,
        lookback,
        number_of_channels,
    ).astype(np.float32)


def fit_static_scaler(
    X_static_train: np.ndarray,
) -> StandardScaler:
    """
    Fit scaling for seasonality and county fixed effects using training rows only.
    """

    scaler = StandardScaler()

    scaler.fit(
        X_static_train
    )

    return scaler


# ============================================================
# 13. CREATE PYTORCH DATA LOADER
# ============================================================

def create_data_loader(
    X_sequence: np.ndarray,
    X_static: np.ndarray,
    y: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    """Create a PyTorch data loader."""

    dataset = TensorDataset(
        torch.tensor(
            X_sequence,
            dtype=torch.float32,
        ),
        torch.tensor(
            X_static,
            dtype=torch.float32,
        ),
        torch.tensor(
            y,
            dtype=torch.float32,
        ),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
    )


# ============================================================
# 14. LSTM MODEL
# ============================================================

class WildfireLSTM(nn.Module):
    """
    Compact unidirectional LSTM for next-day wildfire prediction.
    """

    def __init__(
        self,
        sequence_input_dim: int,
        static_input_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=sequence_input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(
            p=dropout
        )

        self.output_layer = nn.Linear(
            hidden_dim + static_input_dim,
            1,
        )

    def forward(
        self,
        sequence_input: torch.Tensor,
        static_input: torch.Tensor,
    ) -> torch.Tensor:

        _, (
            final_hidden_state,
            _,
        ) = self.lstm(
            sequence_input
        )

        final_hidden = (
            final_hidden_state[-1]
        )

        combined_features = torch.cat(
            [
                final_hidden,
                static_input,
            ],
            dim=1,
        )

        combined_features = self.dropout(
            combined_features
        )

        logits = self.output_layer(
            combined_features
        ).squeeze(1)

        return logits


# ============================================================
# 15. CLASS IMBALANCE WEIGHT
# ============================================================

def calculate_positive_weight(
    labels: np.ndarray,
) -> float:
    """
    Calculate:

        positive weight = number of negatives / number of positives
    """

    labels = np.asarray(
        labels,
        dtype=int,
    )

    number_of_positives = int(
        np.sum(labels == 1)
    )

    number_of_negatives = int(
        np.sum(labels == 0)
    )

    if number_of_positives == 0:
        raise ValueError(
            "Training labels contain no positive class."
        )

    return (
        number_of_negatives
        / number_of_positives
    )


# ============================================================
# 16. PREDICT PROBABILITIES
# ============================================================

def predict_probabilities(
    model: nn.Module,
    data_loader: DataLoader,
) -> np.ndarray:
    """Return positive-class probabilities."""

    model.eval()

    probabilities: list[np.ndarray] = []

    with torch.no_grad():

        for (
            sequence_batch,
            static_batch,
            _,
        ) in data_loader:

            sequence_batch = sequence_batch.to(
                DEVICE
            )

            static_batch = static_batch.to(
                DEVICE
            )

            logits = model(
                sequence_batch,
                static_batch,
            )

            batch_probabilities = torch.sigmoid(
                logits
            )

            probabilities.append(
                batch_probabilities
                .cpu()
                .numpy()
            )

    return np.concatenate(
        probabilities
    )


# ============================================================
# 17. TRAIN ONE INNER-FOLD MODEL
# ============================================================

def train_with_early_stopping(
    X_sequence_train: np.ndarray,
    X_static_train: np.ndarray,
    y_train: np.ndarray,
    X_sequence_validation: np.ndarray,
    X_static_validation: np.ndarray,
    y_validation: np.ndarray,
    parameters: dict,
    outer_fold_id: int,
    inner_fold_id: int,
) -> tuple[
    nn.Module,
    float,
    int,
    list[dict],
]:
    """
    Train one model and use validation AP for early stopping.
    """

    set_seed(
        RANDOM_SEED
        + outer_fold_id * 100
        + inner_fold_id
    )

    training_loader = create_data_loader(
        X_sequence=X_sequence_train,
        X_static=X_static_train,
        y=y_train,
        shuffle=True,
    )

    validation_loader = create_data_loader(
        X_sequence=X_sequence_validation,
        X_static=X_static_validation,
        y=y_validation,
        shuffle=False,
    )

    model = WildfireLSTM(
        sequence_input_dim=X_sequence_train.shape[2],
        static_input_dim=X_static_train.shape[1],
        hidden_dim=int(
            parameters["hidden_dim"]
        ),
        dropout=float(
            parameters["dropout"]
        ),
    ).to(DEVICE)

    positive_weight = calculate_positive_weight(
        labels=y_train
    )

    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            positive_weight,
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=float(
            parameters["weight_decay"]
        ),
    )

    best_validation_ap = -np.inf
    best_epoch = 1
    best_model_state = copy.deepcopy(
        model.state_dict()
    )

    epochs_without_improvement = 0

    history_rows: list[dict] = []

    for epoch in range(
        1,
        MAX_EPOCHS_INNER + 1,
    ):

        model.train()

        epoch_losses: list[float] = []

        for (
            sequence_batch,
            static_batch,
            label_batch,
        ) in training_loader:

            sequence_batch = sequence_batch.to(
                DEVICE
            )

            static_batch = static_batch.to(
                DEVICE
            )

            label_batch = label_batch.to(
                DEVICE
            )

            optimizer.zero_grad()

            logits = model(
                sequence_batch,
                static_batch,
            )

            loss = loss_function(
                logits,
                label_batch,
            )

            loss.backward()

            optimizer.step()

            epoch_losses.append(
                float(
                    loss.item()
                )
            )

        validation_probability = predict_probabilities(
            model=model,
            data_loader=validation_loader,
        )

        validation_ap = safe_average_precision(
            y_true=y_validation,
            probability=validation_probability,
        )

        mean_training_loss = float(
            np.mean(epoch_losses)
        )

        history_rows.append(
            {
                "outer_fold_id": outer_fold_id,
                "inner_fold_id": inner_fold_id,
                "epoch": epoch,
                "training_loss": mean_training_loss,
                "validation_AP": validation_ap,
            }
        )

        if (
            np.isfinite(validation_ap)
            and validation_ap
            > best_validation_ap + 1e-6
        ):

            best_validation_ap = validation_ap
            best_epoch = epoch

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epoch >= MINIMUM_EPOCHS
            and epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            break

    model.load_state_dict(
        best_model_state
    )

    return (
        model,
        float(best_validation_ap),
        int(best_epoch),
        history_rows,
    )


# ============================================================
# 18. TRAIN FINAL OUTER-FOLD MODEL
# ============================================================

def train_final_model(
    X_sequence_train: np.ndarray,
    X_static_train: np.ndarray,
    y_train: np.ndarray,
    parameters: dict,
    number_of_epochs: int,
    outer_fold_id: int,
) -> tuple[nn.Module, list[dict], float]:
    """
    Train the final outer-fold model for the inner-selected epoch count.

    No outer-test labels are used for early stopping.
    """

    set_seed(
        RANDOM_SEED
        + outer_fold_id * 1000
    )

    training_loader = create_data_loader(
        X_sequence=X_sequence_train,
        X_static=X_static_train,
        y=y_train,
        shuffle=True,
    )

    model = WildfireLSTM(
        sequence_input_dim=X_sequence_train.shape[2],
        static_input_dim=X_static_train.shape[1],
        hidden_dim=int(
            parameters["hidden_dim"]
        ),
        dropout=float(
            parameters["dropout"]
        ),
    ).to(DEVICE)

    positive_weight = calculate_positive_weight(
        labels=y_train
    )

    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            positive_weight,
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=float(
            parameters["weight_decay"]
        ),
    )

    history_rows: list[dict] = []

    for epoch in range(
        1,
        number_of_epochs + 1,
    ):

        model.train()

        epoch_losses: list[float] = []

        for (
            sequence_batch,
            static_batch,
            label_batch,
        ) in training_loader:

            sequence_batch = sequence_batch.to(
                DEVICE
            )

            static_batch = static_batch.to(
                DEVICE
            )

            label_batch = label_batch.to(
                DEVICE
            )

            optimizer.zero_grad()

            logits = model(
                sequence_batch,
                static_batch,
            )

            loss = loss_function(
                logits,
                label_batch,
            )

            loss.backward()

            optimizer.step()

            epoch_losses.append(
                float(
                    loss.item()
                )
            )

        history_rows.append(
            {
                "outer_fold_id": outer_fold_id,
                "inner_fold_id": 0,
                "epoch": epoch,
                "training_loss": float(
                    np.mean(epoch_losses)
                ),
                "validation_AP": np.nan,
            }
        )

    return (
        model,
        history_rows,
        positive_weight,
    )


# ============================================================
# 19. TUNE LSTM WITH INNER FOLDS
# ============================================================

def tune_lstm(
    full_data: pd.DataFrame,
    outer_train_start: pd.Timestamp,
    outer_train_end: pd.Timestamp,
    inner_registry_for_outer_fold: pd.DataFrame,
    static_columns: list[str],
    outer_fold_id: int,
) -> tuple[
    dict,
    int,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Select LSTM hyperparameters using mean inner-validation AP.
    """

    result_rows: list[dict] = []
    history_rows: list[dict] = []

    for parameter_id, parameters in enumerate(
        LSTM_PARAMETER_GRID,
        start=1,
    ):

        lookback = int(
            parameters["lookback"]
        )

        sequence_data = build_sequences(
            dataframe=full_data,
            lookback=lookback,
            static_columns=static_columns,
        )

        for inner_fold in (
            inner_registry_for_outer_fold
            .sort_values("inner_fold_id")
            .itertuples(index=False)
        ):

            inner_fold_id = int(
                inner_fold.inner_fold_id
            )

            train_indices = select_sequence_indices(
                sequence_data=sequence_data,
                start_date=pd.Timestamp(
                    inner_fold.inner_train_issue_start
                ),
                end_date=pd.Timestamp(
                    inner_fold.inner_train_issue_end
                ),
            )

            validation_indices = select_sequence_indices(
                sequence_data=sequence_data,
                start_date=pd.Timestamp(
                    inner_fold.inner_validation_issue_start
                ),
                end_date=pd.Timestamp(
                    inner_fold.inner_validation_issue_end
                ),
            )

            if (
                len(train_indices) == 0
                or len(validation_indices) == 0
            ):
                continue

            y_train = sequence_data[
                "y"
            ][train_indices]

            y_validation = sequence_data[
                "y"
            ][validation_indices]

            if (
                np.unique(y_train).size < 2
                or np.unique(y_validation).size < 2
            ):
                continue

            X_sequence_train_raw = sequence_data[
                "X_sequence"
            ][train_indices]

            X_sequence_validation_raw = sequence_data[
                "X_sequence"
            ][validation_indices]

            X_static_train_raw = sequence_data[
                "X_static"
            ][train_indices]

            X_static_validation_raw = sequence_data[
                "X_static"
            ][validation_indices]

            sequence_scaler = fit_sequence_scaler(
                X_sequence_train=X_sequence_train_raw
            )

            static_scaler = fit_static_scaler(
                X_static_train=X_static_train_raw
            )

            X_sequence_train = transform_sequences(
                X_sequence=X_sequence_train_raw,
                scaler=sequence_scaler,
            )

            X_sequence_validation = transform_sequences(
                X_sequence=X_sequence_validation_raw,
                scaler=sequence_scaler,
            )

            X_static_train = static_scaler.transform(
                X_static_train_raw
            ).astype(np.float32)

            X_static_validation = static_scaler.transform(
                X_static_validation_raw
            ).astype(np.float32)

            (
                _,
                validation_ap,
                best_epoch,
                current_history,
            ) = train_with_early_stopping(
                X_sequence_train=X_sequence_train,
                X_static_train=X_static_train,
                y_train=y_train,
                X_sequence_validation=X_sequence_validation,
                X_static_validation=X_static_validation,
                y_validation=y_validation,
                parameters=parameters,
                outer_fold_id=outer_fold_id,
                inner_fold_id=inner_fold_id,
            )

            for row in current_history:

                row.update(
                    {
                        "parameter_id": parameter_id,
                        "lookback": lookback,
                        "hidden_dim": parameters[
                            "hidden_dim"
                        ],
                        "dropout": parameters[
                            "dropout"
                        ],
                        "weight_decay": parameters[
                            "weight_decay"
                        ],
                        "training_stage": "inner_tuning",
                    }
                )

                history_rows.append(
                    row
                )

            result_rows.append(
                {
                    "outer_fold_id": outer_fold_id,
                    "inner_fold_id": inner_fold_id,
                    "validation_target_year": int(
                        inner_fold.validation_target_year
                    ),
                    "parameter_id": parameter_id,
                    "lookback": lookback,
                    "hidden_dim": parameters[
                        "hidden_dim"
                    ],
                    "dropout": parameters[
                        "dropout"
                    ],
                    "weight_decay": parameters[
                        "weight_decay"
                    ],
                    "best_epoch": best_epoch,
                    "validation_AP": validation_ap,
                    "training_sequences": len(
                        train_indices
                    ),
                    "validation_sequences": len(
                        validation_indices
                    ),
                }
            )

    inner_results = pd.DataFrame(
        result_rows
    )

    if inner_results.empty:
        raise ValueError(
            f"No valid LSTM inner result was created "
            f"for outer fold {outer_fold_id}."
        )

    summary = (
        inner_results
        .groupby(
            [
                "parameter_id",
                "lookback",
                "hidden_dim",
                "dropout",
                "weight_decay",
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
            median_best_epoch=(
                "best_epoch",
                "median",
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

    # Prefer the simpler model when AP values are practically tied.
    selected_row = (
        practically_tied
        .sort_values(
            [
                "lookback",
                "hidden_dim",
                "dropout",
                "weight_decay",
                "parameter_id",
            ]
        )
        .iloc[0]
    )

    selected_parameter_id = int(
        selected_row["parameter_id"]
    )

    selected_parameters = (
        LSTM_PARAMETER_GRID[
            selected_parameter_id - 1
        ].copy()
    )

    selected_epochs = max(
        1,
        int(
            round(
                float(
                    selected_row[
                        "median_best_epoch"
                    ]
                )
            )
        ),
    )

    inner_results = inner_results.merge(
        summary,
        on=[
            "parameter_id",
            "lookback",
            "hidden_dim",
            "dropout",
            "weight_decay",
        ],
        how="left",
    )

    return (
        selected_parameters,
        selected_epochs,
        inner_results,
        pd.DataFrame(history_rows),
    )



# ============================================================
# 19B. GENERATE INNER OOF PREDICTIONS
# ============================================================

def generate_selected_inner_oof_predictions(
    full_data: pd.DataFrame,
    inner_registry_for_outer_fold: pd.DataFrame,
    static_columns: list[str],
    selected_parameters: dict,
    outer_fold_id: int,
    outer_target_year: int,
) -> pd.DataFrame:
    """Generate row-level LSTM inner-validation probabilities."""

    lookback = int(selected_parameters["lookback"])

    sequence_data = build_sequences(
        dataframe=full_data,
        lookback=lookback,
        static_columns=static_columns,
    )

    prediction_tables: list[pd.DataFrame] = []

    for inner_fold in (
        inner_registry_for_outer_fold
        .sort_values("inner_fold_id")
        .itertuples(index=False)
    ):
        inner_fold_id = int(inner_fold.inner_fold_id)

        train_indices = select_sequence_indices(
            sequence_data=sequence_data,
            start_date=pd.Timestamp(
                inner_fold.inner_train_issue_start
            ),
            end_date=pd.Timestamp(
                inner_fold.inner_train_issue_end
            ),
        )

        validation_indices = select_sequence_indices(
            sequence_data=sequence_data,
            start_date=pd.Timestamp(
                inner_fold.inner_validation_issue_start
            ),
            end_date=pd.Timestamp(
                inner_fold.inner_validation_issue_end
            ),
        )

        if (
            len(train_indices) == 0
            or len(validation_indices) == 0
        ):
            continue

        y_train = sequence_data["y"][train_indices]
        y_validation = sequence_data["y"][validation_indices]

        if (
            np.unique(y_train).size < 2
            or np.unique(y_validation).size < 2
        ):
            continue

        X_sequence_train_raw = sequence_data[
            "X_sequence"
        ][train_indices]

        X_sequence_validation_raw = sequence_data[
            "X_sequence"
        ][validation_indices]

        X_static_train_raw = sequence_data[
            "X_static"
        ][train_indices]

        X_static_validation_raw = sequence_data[
            "X_static"
        ][validation_indices]

        sequence_scaler = fit_sequence_scaler(
            X_sequence_train=X_sequence_train_raw
        )

        static_scaler = fit_static_scaler(
            X_static_train=X_static_train_raw
        )

        X_sequence_train = transform_sequences(
            X_sequence=X_sequence_train_raw,
            scaler=sequence_scaler,
        )

        X_sequence_validation = transform_sequences(
            X_sequence=X_sequence_validation_raw,
            scaler=sequence_scaler,
        )

        X_static_train = static_scaler.transform(
            X_static_train_raw
        ).astype(np.float32)

        X_static_validation = static_scaler.transform(
            X_static_validation_raw
        ).astype(np.float32)

        (
            selected_model,
            _,
            _,
            _,
        ) = train_with_early_stopping(
            X_sequence_train=X_sequence_train,
            X_static_train=X_static_train,
            y_train=y_train,
            X_sequence_validation=X_sequence_validation,
            X_static_validation=X_static_validation,
            y_validation=y_validation,
            parameters=selected_parameters,
            outer_fold_id=outer_fold_id,
            inner_fold_id=inner_fold_id,
        )

        validation_loader = create_data_loader(
            X_sequence=X_sequence_validation,
            X_static=X_static_validation,
            y=y_validation,
            shuffle=False,
        )

        raw_probability = predict_probabilities(
            model=selected_model,
            data_loader=validation_loader,
        )

        fold_predictions = pd.DataFrame(
            {
                "model": "lstm",
                "outer_fold_id": outer_fold_id,
                "outer_target_year": outer_target_year,
                "inner_fold_id": inner_fold_id,
                "validation_target_year": int(
                    inner_fold.validation_target_year
                ),
                "county": sequence_data[
                    "county"
                ][validation_indices],
                "date": pd.to_datetime(
                    sequence_data[
                        "issue_date"
                    ][validation_indices]
                ),
                "target_date": pd.to_datetime(
                    sequence_data[
                        "target_date"
                    ][validation_indices]
                ),
                "target_next_day": y_validation,
                "raw_probability": raw_probability,
            }
        )

        prediction_tables.append(fold_predictions)

    if not prediction_tables:
        raise ValueError(
            f"No LSTM inner OOF predictions were generated for "
            f"outer fold {outer_fold_id}."
        )

    return pd.concat(prediction_tables, ignore_index=True)


# ============================================================
# 20. MAIN PROGRAM
# ============================================================

def main() -> None:
    """Train and evaluate the LSTM across all outer folds."""

    print("\n" + "=" * 78)
    print("CODE 06: LSTM SEQUENCE PREPARATION AND TRAINING")
    print("=" * 78)

    print(f"\nPyTorch device: {DEVICE}")

    # --------------------------------------------------------
    # 20.1 Check required files
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
    # 20.2 Load data
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

    feature_data = feature_data.sort_values(
        [
            "county",
            "date",
        ]
    ).reset_index(drop=True)

    check_daily_continuity(
        dataframe=feature_data
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
    # 20.3 Add county fixed effects
    # --------------------------------------------------------

    feature_data, county_columns = add_county_fixed_effects(
        dataframe=feature_data
    )

    static_columns = (
        STATIC_COLUMNS
        + county_columns
    )

    print(
        f"Sequence channels: {SEQUENCE_COLUMNS}"
    )

    print(
        f"Static features  : {len(static_columns)}"
    )

    # --------------------------------------------------------
    # 20.4 Output containers
    # --------------------------------------------------------

    all_oof_predictions: list[pd.DataFrame] = []
    hyperparameter_rows: list[dict] = []
    all_inner_results: list[pd.DataFrame] = []
    all_inner_oof_predictions: list[pd.DataFrame] = []
    all_training_history: list[pd.DataFrame] = []
    audit_rows: list[dict] = []

    # --------------------------------------------------------
    # 20.5 Process outer folds
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

        print("\n" + "-" * 78)
        print(
            f"Outer fold {fold_id}: target year {target_year}"
        )
        print("-" * 78)

        inner_for_outer_fold = inner_registry[
            inner_registry["outer_fold_id"]
            == fold_id
        ].copy()

        if inner_for_outer_fold.empty:
            raise ValueError(
                f"No inner folds were found for outer fold {fold_id}."
            )

        # ----------------------------------------------------
        # Tune LSTM using inner folds only
        # ----------------------------------------------------

        (
            selected_parameters,
            selected_epochs,
            inner_results,
            inner_history,
        ) = tune_lstm(
            full_data=feature_data,
            outer_train_start=pd.Timestamp(
                outer_fold.train_issue_start
            ),
            outer_train_end=pd.Timestamp(
                outer_fold.train_issue_end
            ),
            inner_registry_for_outer_fold=inner_for_outer_fold,
            static_columns=static_columns,
            outer_fold_id=fold_id,
        )

        all_inner_results.append(
            inner_results
        )

        all_training_history.append(
            inner_history
        )


        selected_inner_oof = generate_selected_inner_oof_predictions(
            full_data=feature_data,
            inner_registry_for_outer_fold=inner_for_outer_fold,
            static_columns=static_columns,
            selected_parameters=selected_parameters,
            outer_fold_id=fold_id,
            outer_target_year=target_year,
        )

        all_inner_oof_predictions.append(selected_inner_oof)

        lookback = int(
            selected_parameters["lookback"]
        )

        # ----------------------------------------------------
        # Build sequences for selected lookback
        # ----------------------------------------------------

        sequence_data = build_sequences(
            dataframe=feature_data,
            lookback=lookback,
            static_columns=static_columns,
        )

        outer_train_indices = select_sequence_indices(
            sequence_data=sequence_data,
            start_date=pd.Timestamp(
                outer_fold.train_issue_start
            ),
            end_date=pd.Timestamp(
                outer_fold.train_issue_end
            ),
        )

        outer_test_indices = select_sequence_indices(
            sequence_data=sequence_data,
            start_date=pd.Timestamp(
                outer_fold.test_issue_start
            ),
            end_date=pd.Timestamp(
                outer_fold.test_issue_end
            ),
        )

        if (
            len(outer_train_indices) == 0
            or len(outer_test_indices) == 0
        ):
            raise ValueError(
                f"Outer fold {fold_id} has empty LSTM train or test sequences."
            )

        y_outer_train = sequence_data[
            "y"
        ][outer_train_indices]

        y_outer_test = sequence_data[
            "y"
        ][outer_test_indices]

        if (
            np.unique(y_outer_train).size < 2
            or np.unique(y_outer_test).size < 2
        ):
            raise ValueError(
                f"Outer fold {fold_id} does not contain both classes."
            )

        X_sequence_outer_train_raw = sequence_data[
            "X_sequence"
        ][outer_train_indices]

        X_sequence_outer_test_raw = sequence_data[
            "X_sequence"
        ][outer_test_indices]

        X_static_outer_train_raw = sequence_data[
            "X_static"
        ][outer_train_indices]

        X_static_outer_test_raw = sequence_data[
            "X_static"
        ][outer_test_indices]

        # ----------------------------------------------------
        # Fit preprocessing on outer training sequences only
        # ----------------------------------------------------

        sequence_scaler = fit_sequence_scaler(
            X_sequence_train=X_sequence_outer_train_raw
        )

        static_scaler = fit_static_scaler(
            X_static_train=X_static_outer_train_raw
        )

        X_sequence_outer_train = transform_sequences(
            X_sequence=X_sequence_outer_train_raw,
            scaler=sequence_scaler,
        )

        X_sequence_outer_test = transform_sequences(
            X_sequence=X_sequence_outer_test_raw,
            scaler=sequence_scaler,
        )

        X_static_outer_train = static_scaler.transform(
            X_static_outer_train_raw
        ).astype(np.float32)

        X_static_outer_test = static_scaler.transform(
            X_static_outer_test_raw
        ).astype(np.float32)

        # ----------------------------------------------------
        # Train final model without outer-test early stopping
        # ----------------------------------------------------

        (
            final_model,
            final_history_rows,
            positive_weight,
        ) = train_final_model(
            X_sequence_train=X_sequence_outer_train,
            X_static_train=X_static_outer_train,
            y_train=y_outer_train,
            parameters=selected_parameters,
            number_of_epochs=selected_epochs,
            outer_fold_id=fold_id,
        )

        final_history = pd.DataFrame(
            final_history_rows
        )

        final_history["parameter_id"] = np.nan
        final_history["lookback"] = lookback
        final_history["hidden_dim"] = selected_parameters[
            "hidden_dim"
        ]
        final_history["dropout"] = selected_parameters[
            "dropout"
        ]
        final_history["weight_decay"] = selected_parameters[
            "weight_decay"
        ]
        final_history["training_stage"] = "outer_final_fit"

        all_training_history.append(
            final_history
        )

        # ----------------------------------------------------
        # Freeze outer-test probabilities
        # ----------------------------------------------------

        test_loader = create_data_loader(
            X_sequence=X_sequence_outer_test,
            X_static=X_static_outer_test,
            y=y_outer_test,
            shuffle=False,
        )

        raw_test_probability = predict_probabilities(
            model=final_model,
            data_loader=test_loader,
        )

        outer_test_ap = safe_average_precision(
            y_true=y_outer_test,
            probability=raw_test_probability,
        )

        print(
            f"Selected lookback      : {lookback}"
        )

        print(
            f"Selected hidden units  : "
            f"{selected_parameters['hidden_dim']}"
        )

        print(
            f"Selected dropout       : "
            f"{selected_parameters['dropout']}"
        )

        print(
            f"Selected epochs        : {selected_epochs}"
        )

        print(
            f"Positive class weight  : {positive_weight:.4f}"
        )

        print(
            f"Outer-test raw AP      : {outer_test_ap:.6f}"
        )

        # ----------------------------------------------------
        # Save OOF predictions
        # ----------------------------------------------------

        fold_predictions = pd.DataFrame(
            {
                "model": "lstm",
                "outer_fold_id": fold_id,
                "target_year": target_year,
                "county": sequence_data[
                    "county"
                ][outer_test_indices],
                "date": pd.to_datetime(
                    sequence_data[
                        "issue_date"
                    ][outer_test_indices]
                ),
                "target_date": pd.to_datetime(
                    sequence_data[
                        "target_date"
                    ][outer_test_indices]
                ),
                "target_next_day": y_outer_test,
                "raw_probability": raw_test_probability,
            }
        )

        all_oof_predictions.append(
            fold_predictions
        )

        # ----------------------------------------------------
        # Save hyperparameters
        # ----------------------------------------------------

        hyperparameter_rows.append(
            {
                "outer_fold_id": fold_id,
                "target_year": target_year,
                "lookback": lookback,
                "hidden_dim": selected_parameters[
                    "hidden_dim"
                ],
                "dropout": selected_parameters[
                    "dropout"
                ],
                "weight_decay": selected_parameters[
                    "weight_decay"
                ],
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "selected_epochs": selected_epochs,
                "positive_class_weight": positive_weight,
                "sequence_channels": ",".join(
                    SEQUENCE_COLUMNS
                ),
                "number_of_static_features": len(
                    static_columns
                ),
                "outer_test_raw_AP": outer_test_ap,
            }
        )

        # ----------------------------------------------------
        # Save fold audit
        # ----------------------------------------------------

        test_issue_dates = pd.to_datetime(
            sequence_data[
                "issue_date"
            ][outer_test_indices]
        )

        test_target_dates = pd.to_datetime(
            sequence_data[
                "target_date"
            ][outer_test_indices]
        )

        audit_rows.append(
            {
                "outer_fold_id": fold_id,
                "target_year": target_year,
                "training_sequences": len(
                    outer_train_indices
                ),
                "test_sequences": len(
                    outer_test_indices
                ),
                "training_positive_labels": int(
                    y_outer_train.sum()
                ),
                "test_positive_labels": int(
                    y_outer_test.sum()
                ),
                "lookback": lookback,
                "training_issue_start": pd.to_datetime(
                    sequence_data[
                        "issue_date"
                    ][outer_train_indices]
                ).min(),
                "training_issue_end": pd.to_datetime(
                    sequence_data[
                        "issue_date"
                    ][outer_train_indices]
                ).max(),
                "test_issue_start": test_issue_dates.min(),
                "test_issue_end": test_issue_dates.max(),
                "test_target_start": test_target_dates.min(),
                "test_target_end": test_target_dates.max(),
                "outer_test_raw_AP": outer_test_ap,
            }
        )

    # --------------------------------------------------------
    # 20.6 Combine outputs
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

    hyperparameters = pd.DataFrame(
        hyperparameter_rows
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

    training_history = pd.concat(
        all_training_history,
        ignore_index=True,
    )

    audit_table = pd.DataFrame(
        audit_rows
    )

    # --------------------------------------------------------
    # 20.7 Save outputs
    # --------------------------------------------------------

    oof_predictions.to_csv(
        OOF_PREDICTION_FILE,
        index=False,
    )

    hyperparameters.to_csv(
        HYPERPARAMETER_FILE,
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

    training_history.to_csv(
        TRAINING_HISTORY_FILE,
        index=False,
    )

    audit_table.to_csv(
        AUDIT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # 20.8 Final summary
    # --------------------------------------------------------

    pooled_raw_ap = safe_average_precision(
        y_true=oof_predictions[
            "target_next_day"
        ].to_numpy(dtype=int),
        probability=oof_predictions[
            "raw_probability"
        ].to_numpy(dtype=float),
    )

    print("\n" + "=" * 78)
    print("CODE 06 COMPLETED SUCCESSFULLY")
    print("=" * 78)

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
        f"\nSelected hyperparameters saved at:\n"
        f"{HYPERPARAMETER_FILE}"
    )

    print(
        f"\nInner-validation results saved at:\n"
        f"{INNER_RESULT_FILE}"
    )

    print(
        f"\nTraining history saved at:\n"
        f"{TRAINING_HISTORY_FILE}"
    )

    print(
        f"\nFold audit saved at:\n"
        f"{AUDIT_FILE}"
    )


# ============================================================
# 21. RUN THE PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
