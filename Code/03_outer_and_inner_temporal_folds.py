#!/usr/bin/env python3
"""
CODE 03: Outer and inner temporal folds
=======================================

This script reads the causal feature table produced by Code 02:

    02_causal_feature_table.csv

It then:

1. Constructs five expanding-window outer folds for target years 2019-2023.
2. Applies a 30-day embargo before every outer test period.
3. Constructs inner forward-chaining validation folds inside each outer-training set.
4. Uses the same 30-day embargo before every inner validation period.
5. Assigns every county-day row to train, embargo, test, or unused.
6. Saves the outer-fold registry, inner-fold registry, and row-level fold assignments.

Important:
The fold boundaries are defined using issue dates because the predictors belong
to issue day t, while the target belongs to target day t+1.
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

INPUT_FILE: Path = (
    OUTPUT_DIR / "02_causal_feature_table.csv"
)

OUTER_REGISTRY_FILE: Path = (
    OUTPUT_DIR / "03_outer_fold_registry.csv"
)

INNER_REGISTRY_FILE: Path = (
    OUTPUT_DIR / "03_inner_fold_registry.csv"
)

ROW_ASSIGNMENT_FILE: Path = (
    OUTPUT_DIR / "03_fold_row_assignments.csv"
)

AUDIT_FILE: Path = (
    OUTPUT_DIR / "03_fold_audit.csv"
)


# ============================================================
# 2. TEMPORAL SETTINGS
# ============================================================

OUTER_TARGET_YEARS: list[int] = [
    2019,
    2020,
    2021,
    2022,
    2023,
]

EMBARGO_DAYS: int = 30

FIRST_TRAINING_ISSUE_DATE: str = "2014-01-01"


# ============================================================
# 3. REQUIRED INPUT COLUMNS
# ============================================================

REQUIRED_COLUMNS: list[str] = [
    "county",
    "date",
    "target_date",
    "target_next_day",
]


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

def check_required_columns(
    dataframe: pd.DataFrame,
) -> None:
    """Stop when Code 02 output is missing an essential column."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The Code 02 output is missing these columns:\n"
            f"{missing_columns}"
        )


# ============================================================
# 5. CREATE ONE OUTER FOLD
# ============================================================

def create_outer_fold(
    fold_id: int,
    target_year: int,
) -> dict:
    """
    Create one expanding-window outer fold.

    Example for target year 2019:

    Training issue dates:
        2014-01-01 to 2018-11-30

    Embargo issue dates:
        2018-12-01 to 2018-12-30

    Test issue dates:
        2018-12-31 to 2019-12-30

    Test target dates:
        2019-01-01 to 2019-12-31
    """

    train_issue_start = pd.Timestamp(
        FIRST_TRAINING_ISSUE_DATE
    )

    test_target_start = pd.Timestamp(
        year=target_year,
        month=1,
        day=1,
    )

    test_target_end = pd.Timestamp(
        year=target_year,
        month=12,
        day=31,
    )

    # The target at t+1 belongs to the weather row at issue day t.
    test_issue_start = (
        test_target_start
        - pd.Timedelta(days=1)
    )

    test_issue_end = (
        test_target_end
        - pd.Timedelta(days=1)
    )

    embargo_issue_end = (
        test_issue_start
        - pd.Timedelta(days=1)
    )

    embargo_issue_start = (
        test_issue_start
        - pd.Timedelta(days=EMBARGO_DAYS)
    )

    train_issue_end = (
        embargo_issue_start
        - pd.Timedelta(days=1)
    )

    return {
        "fold_id": fold_id,
        "target_year": target_year,
        "train_issue_start": train_issue_start,
        "train_issue_end": train_issue_end,
        "embargo_issue_start": embargo_issue_start,
        "embargo_issue_end": embargo_issue_end,
        "test_issue_start": test_issue_start,
        "test_issue_end": test_issue_end,
        "test_target_start": test_target_start,
        "test_target_end": test_target_end,
        "embargo_days": EMBARGO_DAYS,
    }


# ============================================================
# 6. BUILD OUTER FOLD REGISTRY
# ============================================================

def build_outer_fold_registry() -> pd.DataFrame:
    """Create the complete 2019-2023 outer-fold registry."""

    rows: list[dict] = []

    for fold_id, target_year in enumerate(
        OUTER_TARGET_YEARS,
        start=1,
    ):

        rows.append(
            create_outer_fold(
                fold_id=fold_id,
                target_year=target_year,
            )
        )

    return pd.DataFrame(rows)


# ============================================================
# 7. BUILD INNER FORWARD-CHAINING FOLDS
# ============================================================

def build_inner_fold_registry(
    outer_registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create inner validation folds inside each outer-training interval.

    For each outer fold, complete calendar years are used as inner
    validation target years.

    Example inside the 2019 outer fold:
        Inner validation target years may be 2016, 2017, and 2018.

    Each inner fold uses:
        earlier issue dates for training,
        30 embargo issue dates,
        one complete following target year for validation.
    """

    rows: list[dict] = []

    for outer_row in outer_registry.itertuples(index=False):

        outer_fold_id = int(
            outer_row.fold_id
        )

        outer_target_year = int(
            outer_row.target_year
        )

        outer_train_end = pd.Timestamp(
            outer_row.train_issue_end
        )

        inner_fold_id = 0

        # Start with validation target year 2016 so that at least
        # two full target years, 2014 and 2015, are available before it.
        candidate_validation_years = range(
            2016,
            outer_target_year,
        )

        for validation_year in candidate_validation_years:

            validation_target_start = pd.Timestamp(
                year=validation_year,
                month=1,
                day=1,
            )

            validation_target_end = pd.Timestamp(
                year=validation_year,
                month=12,
                day=31,
            )

            validation_issue_start = (
                validation_target_start
                - pd.Timedelta(days=1)
            )

            validation_issue_end = (
                validation_target_end
                - pd.Timedelta(days=1)
            )

            inner_embargo_end = (
                validation_issue_start
                - pd.Timedelta(days=1)
            )

            inner_embargo_start = (
                validation_issue_start
                - pd.Timedelta(days=EMBARGO_DAYS)
            )

            inner_train_end = (
                inner_embargo_start
                - pd.Timedelta(days=1)
            )

            inner_train_start = pd.Timestamp(
                FIRST_TRAINING_ISSUE_DATE
            )

            # The full inner validation period must remain inside
            # the current outer-training interval.
            if validation_issue_end > outer_train_end:
                continue

            if inner_train_end < inner_train_start:
                continue

            inner_fold_id += 1

            rows.append(
                {
                    "outer_fold_id": outer_fold_id,
                    "outer_target_year": outer_target_year,
                    "inner_fold_id": inner_fold_id,
                    "validation_target_year": validation_year,
                    "inner_train_issue_start": inner_train_start,
                    "inner_train_issue_end": inner_train_end,
                    "inner_embargo_issue_start": inner_embargo_start,
                    "inner_embargo_issue_end": inner_embargo_end,
                    "inner_validation_issue_start": validation_issue_start,
                    "inner_validation_issue_end": validation_issue_end,
                    "inner_validation_target_start": validation_target_start,
                    "inner_validation_target_end": validation_target_end,
                    "embargo_days": EMBARGO_DAYS,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 8. ASSIGN ROWS TO OUTER FOLDS
# ============================================================

def assign_outer_rows(
    data: pd.DataFrame,
    outer_registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a row-level table showing whether each observation belongs to:

        train
        embargo
        test
        unused

    for each outer fold.
    """

    assignment_frames: list[pd.DataFrame] = []

    base_columns = [
        "county",
        "date",
        "target_date",
        "target_next_day",
    ]

    for fold in outer_registry.itertuples(index=False):

        assignment = data[
            base_columns
        ].copy()

        assignment["outer_fold_id"] = int(
            fold.fold_id
        )

        assignment["outer_target_year"] = int(
            fold.target_year
        )

        assignment["outer_role"] = "unused"

        train_mask = assignment["date"].between(
            pd.Timestamp(fold.train_issue_start),
            pd.Timestamp(fold.train_issue_end),
        )

        embargo_mask = assignment["date"].between(
            pd.Timestamp(fold.embargo_issue_start),
            pd.Timestamp(fold.embargo_issue_end),
        )

        test_mask = assignment["date"].between(
            pd.Timestamp(fold.test_issue_start),
            pd.Timestamp(fold.test_issue_end),
        )

        assignment.loc[
            train_mask,
            "outer_role",
        ] = "train"

        assignment.loc[
            embargo_mask,
            "outer_role",
        ] = "embargo"

        assignment.loc[
            test_mask,
            "outer_role",
        ] = "test"

        assignment_frames.append(
            assignment
        )

    return pd.concat(
        assignment_frames,
        ignore_index=True,
    )


# ============================================================
# 9. AUDIT OUTER FOLDS
# ============================================================

def audit_outer_folds(
    assignments: pd.DataFrame,
    outer_registry: pd.DataFrame,
) -> pd.DataFrame:
    """Count rows, classes, dates, and counties in each outer-fold part."""

    audit_rows: list[dict] = []

    for fold in outer_registry.itertuples(index=False):

        fold_data = assignments[
            assignments["outer_fold_id"]
            == int(fold.fold_id)
        ]

        for role in [
            "train",
            "embargo",
            "test",
        ]:

            role_data = fold_data[
                fold_data["outer_role"] == role
            ]

            audit_rows.append(
                {
                    "outer_fold_id": int(fold.fold_id),
                    "outer_target_year": int(fold.target_year),
                    "dataset_role": role,
                    "number_of_rows": len(role_data),
                    "number_of_counties": role_data[
                        "county"
                    ].nunique(),
                    "first_issue_date": (
                        role_data["date"].min()
                        if len(role_data)
                        else pd.NaT
                    ),
                    "last_issue_date": (
                        role_data["date"].max()
                        if len(role_data)
                        else pd.NaT
                    ),
                    "first_target_date": (
                        role_data["target_date"].min()
                        if len(role_data)
                        else pd.NaT
                    ),
                    "last_target_date": (
                        role_data["target_date"].max()
                        if len(role_data)
                        else pd.NaT
                    ),
                    "positive_labels": int(
                        role_data["target_next_day"].sum()
                    ) if len(role_data) else 0,
                    "negative_labels": int(
                        len(role_data)
                        - role_data["target_next_day"].sum()
                    ) if len(role_data) else 0,
                }
            )

    return pd.DataFrame(
        audit_rows
    )


# ============================================================
# 10. VALIDATE OUTER FOLDS
# ============================================================

def validate_outer_folds(
    assignments: pd.DataFrame,
    outer_registry: pd.DataFrame,
) -> None:
    """Apply essential leakage and completeness checks."""

    for fold in outer_registry.itertuples(index=False):

        fold_id = int(
            fold.fold_id
        )

        fold_data = assignments[
            assignments["outer_fold_id"] == fold_id
        ]

        train_data = fold_data[
            fold_data["outer_role"] == "train"
        ]

        embargo_data = fold_data[
            fold_data["outer_role"] == "embargo"
        ]

        test_data = fold_data[
            fold_data["outer_role"] == "test"
        ]

        if len(train_data) == 0:
            raise ValueError(
                f"Outer fold {fold_id} has no training rows."
            )

        if len(embargo_data) == 0:
            raise ValueError(
                f"Outer fold {fold_id} has no embargo rows."
            )

        if len(test_data) == 0:
            raise ValueError(
                f"Outer fold {fold_id} has no test rows."
            )

        if train_data["date"].max() >= embargo_data["date"].min():
            raise ValueError(
                f"Outer fold {fold_id}: training overlaps embargo."
            )

        if embargo_data["date"].max() >= test_data["date"].min():
            raise ValueError(
                f"Outer fold {fold_id}: embargo overlaps test."
            )

        observed_test_target_years = sorted(
            test_data["target_date"]
            .dt.year
            .unique()
            .tolist()
        )

        expected_target_year = [
            int(fold.target_year)
        ]

        if observed_test_target_years != expected_target_year:
            raise ValueError(
                f"Outer fold {fold_id}: expected test target year "
                f"{expected_target_year}, found "
                f"{observed_test_target_years}."
            )

        number_of_embargo_dates = (
            embargo_data["date"]
            .nunique()
        )

        if number_of_embargo_dates != EMBARGO_DAYS:
            raise ValueError(
                f"Outer fold {fold_id}: expected "
                f"{EMBARGO_DAYS} embargo dates, found "
                f"{number_of_embargo_dates}."
            )

        if train_data["target_next_day"].nunique() < 2:
            raise ValueError(
                f"Outer fold {fold_id}: training data does not "
                "contain both classes."
            )

        if test_data["target_next_day"].nunique() < 2:
            raise ValueError(
                f"Outer fold {fold_id}: test data does not "
                "contain both classes."
            )


# ============================================================
# 11. SAVE DATE COLUMNS CLEANLY
# ============================================================

def format_date_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Convert datetime columns into YYYY-MM-DD strings before saving."""

    dataframe = dataframe.copy()

    date_columns = [
        column
        for column in dataframe.columns
        if (
            "date" in column
            or "start" in column
            or "end" in column
        )
    ]

    for column in date_columns:

        if pd.api.types.is_datetime64_any_dtype(
            dataframe[column]
        ):

            dataframe[column] = (
                dataframe[column]
                .dt.strftime("%Y-%m-%d")
            )

    return dataframe


# ============================================================
# 12. MAIN PROGRAM
# ============================================================

def main() -> None:
    """Construct and save the temporal-fold registries."""

    print("\n" + "=" * 70)
    print("CODE 03: OUTER AND INNER TEMPORAL FOLDS")
    print("=" * 70)

    # --------------------------------------------------------
    # 12.1 Load Code 02 output
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "The Code 02 output was not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run Code 02 before running Code 03."
        )

    print(f"\nLoading feature table:\n{INPUT_FILE}")

    data = pd.read_csv(
        INPUT_FILE
    )

    check_required_columns(
        dataframe=data,
    )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    ).dt.normalize()

    data["target_date"] = pd.to_datetime(
        data["target_date"],
        errors="raise",
    ).dt.normalize()

    data["target_next_day"] = (
        data["target_next_day"]
        .astype(int)
    )

    data = data.sort_values(
        ["county", "date"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 12.2 Create outer folds
    # --------------------------------------------------------

    print("\n1. Constructing 2019-2023 outer folds...")

    outer_registry = (
        build_outer_fold_registry()
    )

    # --------------------------------------------------------
    # 12.3 Create inner folds
    # --------------------------------------------------------

    print("2. Constructing inner forward-chaining folds...")

    inner_registry = (
        build_inner_fold_registry(
            outer_registry=outer_registry,
        )
    )

    # --------------------------------------------------------
    # 12.4 Assign rows
    # --------------------------------------------------------

    print("3. Assigning rows to train, embargo, and test...")

    row_assignments = assign_outer_rows(
        data=data,
        outer_registry=outer_registry,
    )

    # --------------------------------------------------------
    # 12.5 Validate folds
    # --------------------------------------------------------

    print("4. Validating temporal separation and embargo...")

    validate_outer_folds(
        assignments=row_assignments,
        outer_registry=outer_registry,
    )

    # --------------------------------------------------------
    # 12.6 Create audit table
    # --------------------------------------------------------

    fold_audit = audit_outer_folds(
        assignments=row_assignments,
        outer_registry=outer_registry,
    )

    # --------------------------------------------------------
    # 12.7 Save all outputs
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outer_registry_to_save = format_date_columns(
        outer_registry
    )

    inner_registry_to_save = format_date_columns(
        inner_registry
    )

    row_assignments_to_save = format_date_columns(
        row_assignments
    )

    fold_audit_to_save = format_date_columns(
        fold_audit
    )

    outer_registry_to_save.to_csv(
        OUTER_REGISTRY_FILE,
        index=False,
    )

    inner_registry_to_save.to_csv(
        INNER_REGISTRY_FILE,
        index=False,
    )

    row_assignments_to_save.to_csv(
        ROW_ASSIGNMENT_FILE,
        index=False,
    )

    fold_audit_to_save.to_csv(
        AUDIT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # 12.8 Print outer-fold summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OUTER FOLD SUMMARY")
    print("=" * 70)

    print(
        outer_registry_to_save[
            [
                "fold_id",
                "target_year",
                "train_issue_start",
                "train_issue_end",
                "embargo_issue_start",
                "embargo_issue_end",
                "test_issue_start",
                "test_issue_end",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("CODE 03 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Outer folds created : "
        f"{len(outer_registry)}"
    )

    print(
        f"Inner folds created : "
        f"{len(inner_registry)}"
    )

    print(
        f"\nOuter registry saved at:\n"
        f"{OUTER_REGISTRY_FILE}"
    )

    print(
        f"\nInner registry saved at:\n"
        f"{INNER_REGISTRY_FILE}"
    )

    print(
        f"\nRow assignments saved at:\n"
        f"{ROW_ASSIGNMENT_FILE}"
    )

    print(
        f"\nFold audit saved at:\n"
        f"{AUDIT_FILE}"
    )


# ============================================================
# 13. RUN THE PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
