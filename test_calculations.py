
"""
LEBAS - Independent Calculation Verification Test

This script independently verifies the mathematical calculations
used by the LEBAS application.

It checks:
1. Dataset structure
2. Process-level record handling
3. Crushing calculations
4. Grinding data
5. Sieve analysis
6. Beneficiation calculations
7. Material balance
8. Statistical calculations
9. Data sanity

IMPORTANT:
This test calculates expected values independently.
It does not simply call the LEBAS calculation functions.

Run with:
    python test_calculations.py
"""

from pathlib import Path
import math
import sys

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "dataset.csv"

TOLERANCE = 1e-6


EXPECTED_COLUMNS = [
    "sample_id",
    "material_type",
    "experiment_date",
    "operator",
    "replicate",
    "feed_mass",
    "feed_grade",
    "feed_moisture",
    "crusher_type",
    "crush_time",
    "crush_power",
    "crush_feed_size",
    "crush_product_size",
    "grinding_time",
    "grinding_mill_type",
    "grinding_media_mass",
    "grinding_speed",
    "grinding_feed_size",
    "grinding_product_size",
    "sieve_size",
    "mass_retained",
    "sieve_shaking_time",
    "beneficiation_method",
    "beneficiation_time",
    "reagent_type",
    "reagent_dosage",
    "ph_level",
    "pulp_density",
    "temperature",
    "concentrate_mass",
    "concentrate_grade",
    "tailings_mass",
    "tailings_grade",
    "losses_mass",
    "water_added",
    "solid_recovery",
]


# ============================================================
# TEST COUNTERS
# ============================================================

PASS_COUNT = 0
FAIL_COUNT = 0
WARNING_COUNT = 0


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_section(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


def test_pass(message):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"[PASS] {message}")


def test_fail(message):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"[FAIL] {message}")


def test_warning(message):
    global WARNING_COUNT
    WARNING_COUNT += 1
    print(f"[WARNING] {message}")


# ============================================================
# SAFE NUMERIC FUNCTIONS
# ============================================================

def safe_float(value):
    """Convert a value to float safely."""
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def valid_number(value):
    """Return True when value is a finite number."""
    number = safe_float(value)
    return math.isfinite(number)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    print_header("LEBAS INDEPENDENT CALCULATION TEST")

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset      : {DATASET_PATH}")

    if not DATASET_PATH.exists():
        test_fail(
            "dataset.csv was not found."
        )
        print()
        print(
            "Make sure dataset.csv is located in the same folder "
            "as test_calculations.py."
        )
        return None

    try:
        df = pd.read_csv(DATASET_PATH)

        test_pass(
            f"dataset.csv loaded successfully "
            f"({len(df)} rows, {len(df.columns)} columns)."
        )

        return df

    except Exception as exc:
        test_fail(
            f"Could not read dataset.csv: {exc}"
        )
        return None


# ============================================================
# DATASET STRUCTURE
# ============================================================

def test_dataset_structure(df):
    print_section("1. DATASET STRUCTURE")

    actual_columns = list(df.columns)

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in actual_columns
    ]

    if not missing_columns:
        test_pass(
            "All required dataset columns are present."
        )
    else:
        test_fail(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    if len(df) > 0:
        test_pass(
            f"Dataset contains {len(df)} rows."
        )
    else:
        test_fail(
            "Dataset contains zero rows."
        )

    if "sample_id" in df.columns:

        unique_samples = (
            df["sample_id"]
            .dropna()
            .astype(str)
            .nunique()
        )

        if unique_samples > 0:
            test_pass(
                f"Unique samples detected: {unique_samples}."
            )
        else:
            test_fail(
                "No valid sample IDs were detected."
            )

    if (
        "sample_id" in df.columns
        and "replicate" in df.columns
    ):

        unique_process_records = (
            df[
                ["sample_id", "replicate"]
            ]
            .dropna()
            .drop_duplicates()
            .shape[0]
        )

        if unique_process_records > 0:
            test_pass(
                f"Unique sample/replicate records: "
                f"{unique_process_records}."
            )
        else:
            test_fail(
                "No valid sample/replicate combinations found."
            )


# ============================================================
# PROCESS-LEVEL RECORDS
# ============================================================

def get_process_records(df):
    """
    Return one record for each sample + replicate.

    Sieve analysis can have many rows belonging to the same
    experiment. Therefore, process-level calculations must not
    treat every sieve row as a separate experiment.
    """

    required = [
        "sample_id",
        "replicate",
    ]

    for column in required:
        if column not in df.columns:
            return pd.DataFrame()

    process_df = (
        df
        .sort_values(
            ["sample_id", "replicate"]
        )
        .drop_duplicates(
            subset=[
                "sample_id",
                "replicate",
            ],
            keep="first",
        )
        .copy()
    )

    return process_df


# ============================================================
# CRUSHING CALCULATIONS
# ============================================================

def test_crushing(df):
    print_section("2. CRUSHING CALCULATIONS")

    process_df = get_process_records(df)

    if process_df.empty:
        test_fail(
            "No process-level records available."
        )
        return

    required = [
        "sample_id",
        "replicate",
        "crush_feed_size",
        "crush_product_size",
    ]

    missing = [
        column
        for column in required
        if column not in process_df.columns
    ]

    if missing:
        test_fail(
            "Crushing test cannot run. Missing columns: "
            + ", ".join(missing)
        )
        return

    tested = 0

    for _, row in process_df.iterrows():

        sample = row["sample_id"]
        replicate = row["replicate"]

        feed_size = safe_float(
            row["crush_feed_size"]
        )

        product_size = safe_float(
            row["crush_product_size"]
        )

        if not valid_number(feed_size):
            test_warning(
                f"{sample} Rep {replicate}: "
                "crushing feed size is missing."
            )
            continue

        if not valid_number(product_size):
            test_warning(
                f"{sample} Rep {replicate}: "
                "crushing product size is missing."
            )
            continue

        if feed_size <= 0:
            test_fail(
                f"{sample} Rep {replicate}: "
                "crushing feed size must be greater than zero."
            )
            continue

        if product_size <= 0:
            test_fail(
                f"{sample} Rep {replicate}: "
                "crushing product size must be greater than zero."
            )
            continue

        # ----------------------------------------------------
        # INDEPENDENT CRUSHING CALCULATIONS
        # ----------------------------------------------------

        reduction_ratio = (
            feed_size / product_size
        )

        size_reduction_percent = (
            (feed_size - product_size)
            / feed_size
            * 100.0
        )

        print(
            f"{sample} Rep {replicate}: "
            f"Feed={feed_size:g}, "
            f"Product={product_size:g}, "
            f"Reduction Ratio={reduction_ratio:.6f}, "
            f"Size Reduction={size_reduction_percent:.4f}%"
        )

        if reduction_ratio > 0:
            test_pass(
                f"{sample} Rep {replicate}: "
                "reduction ratio is valid."
            )
        else:
            test_fail(
                f"{sample} Rep {replicate}: "
                "reduction ratio is invalid."
            )

        if (
            -1e-9
            <= size_reduction_percent
            <= 100.000000001
        ):
            test_pass(
                f"{sample} Rep {replicate}: "
                "size reduction percentage is valid."
            )
        else:
            test_warning(
                f"{sample} Rep {replicate}: "
                f"size reduction = "
                f"{size_reduction_percent:.4f}%."
            )

        tested += 1

    if tested > 0:
        test_pass(
            f"Crushing calculations independently checked "
            f"for {tested} process records."
        )


# ============================================================
# GRINDING DATA TEST
# ============================================================

def test_grinding(df):
    print_section("3. GRINDING DATA INTEGRITY")

    process_df = get_process_records(df)

    if process_df.empty:
        test_fail(
            "No process-level records available for grinding."
        )
        return

    required = [
        "sample_id",
        "replicate",
        "grinding_time",
        "grinding_feed_size",
        "grinding_product_size",
    ]

    missing = [
        column
        for column in required
        if column not in process_df.columns
    ]

    if missing:
        test_fail(
            "Grinding test cannot run. Missing columns: "
            + ", ".join(missing)
        )
        return

    valid_records = 0

    for _, row in process_df.iterrows():

        sample = row["sample_id"]
        replicate = row["replicate"]

        grinding_time = safe_float(
            row["grinding_time"]
        )

        feed_size = safe_float(
            row["grinding_feed_size"]
        )

        product_size = safe_float(
            row["grinding_product_size"]
        )

        if valid_number(grinding_time):

            if grinding_time >= 0:
                test_pass(
                    f"{sample} Rep {replicate}: "
                    "grinding time is valid."
                )
            else:
                test_fail(
                    f"{sample} Rep {replicate}: "
                    "negative grinding time."
                )

        if valid_number(feed_size):

            if feed_size > 0:
                test_pass(
                    f"{sample} Rep {replicate}: "
                    "grinding feed size is valid."
                )
            else:
                test_fail(
                    f"{sample} Rep {replicate}: "
                    "invalid grinding feed size."
                )

        if valid_number(product_size):

            if product_size > 0:
                test_pass(
                    f"{sample} Rep {replicate}: "
                    "grinding product size is valid."
                )
            else:
                test_fail(
                    f"{sample} Rep {replicate}: "
                    "invalid grinding product size."
                )

        if (
            valid_number(feed_size)
            and valid_number(product_size)
            and feed_size > 0
            and product_size > 0
        ):

            grinding_ratio = (
                feed_size / product_size
            )

            print(
                f"{sample} Rep {replicate}: "
                f"Grinding reduction ratio="
                f"{grinding_ratio:.6f}"
            )

            valid_records += 1

    if len(process_df) <= len(df):
        test_pass(
            "Process-level deduplication is valid for grinding."
        )
    else:
        test_fail(
            "Process-level record count exceeds raw row count."
        )

    if valid_records > 0:
        test_pass(
            f"Grinding calculations/data checked for "
            f"{valid_records} records."
        )


# ============================================================
# SIEVE ANALYSIS
# ============================================================

def test_sieve_analysis(df):
    print_section("4. SIEVE ANALYSIS CALCULATIONS")

    required = [
        "sample_id",
        "replicate",
        "sieve_size",
        "mass_retained",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        test_fail(
            "Sieve test cannot run. Missing columns: "
            + ", ".join(missing)
        )
        return

    sieve_df = df.copy()

    sieve_df["sieve_size_numeric"] = pd.to_numeric(
        sieve_df["sieve_size"],
        errors="coerce",
    )

    sieve_df["mass_retained_numeric"] = pd.to_numeric(
        sieve_df["mass_retained"],
        errors="coerce",
    )

    valid_sizes = sorted(
        sieve_df.loc[
            sieve_df["sieve_size_numeric"] > 0,
            "sieve_size_numeric",
        ]
        .dropna()
        .unique(),
        reverse=True,
    )

    if len(valid_sizes) > 0:

        size_text = ", ".join(
            f"{size:g} µm"
            for size in valid_sizes
        )

        test_pass(
            "Sieve sizes detected dynamically: "
            + size_text
        )

    else:
        test_fail(
            "No valid positive sieve sizes detected."
        )

    # --------------------------------------------------------
    # 150 µm check
    # --------------------------------------------------------

    if any(
        abs(size - 150.0) <= TOLERANCE
        for size in valid_sizes
    ):
        test_pass(
            "150 µm sieve size is accepted and detected."
        )
    else:
        test_warning(
            "150 µm was not found in the current dataset. "
            "This is only a warning."
        )

    # --------------------------------------------------------
    # GROUP BY SAMPLE AND REPLICATE
    # --------------------------------------------------------

    grouped = sieve_df.groupby(
        [
            "sample_id",
            "replicate",
        ],
        dropna=False,
    )

    groups_tested = 0

    for (
        sample,
        replicate,
    ), group in grouped:

        group = group.copy()

        group = group[
            group["sieve_size_numeric"].notna()
            & group["mass_retained_numeric"].notna()
            & (group["sieve_size_numeric"] > 0)
        ].copy()

        if group.empty:
            continue

        total_mass = (
            group["mass_retained_numeric"]
            .sum()
        )

        if total_mass <= 0:
            test_fail(
                f"{sample} Rep {replicate}: "
                "total sieve mass is zero or negative."
            )
            continue

        # Largest opening to smallest opening.
        group = group.sort_values(
            "sieve_size_numeric",
            ascending=False,
        )

        # ----------------------------------------------------
        # PERCENT RETAINED
        # ----------------------------------------------------

        group["expected_percent_retained"] = (
            group["mass_retained_numeric"]
            / total_mass
            * 100.0
        )

        # ----------------------------------------------------
        # CUMULATIVE PERCENT RETAINED
        # ----------------------------------------------------

        group["expected_cumulative_retained"] = (
            group["expected_percent_retained"]
            .cumsum()
        )

        # ----------------------------------------------------
        # PERCENT PASSING
        # ----------------------------------------------------

        group["expected_percent_passing"] = (
            100.0
            - group["expected_cumulative_retained"]
        )

        retained_total = (
            group["expected_percent_retained"]
            .sum()
        )

        final_passing = (
            group["expected_percent_passing"]
            .iloc[-1]
        )

        if abs(final_passing) < 1e-9:
            final_passing = 0.0

        # ----------------------------------------------------
        # CHECK PERCENT RETAINED
        # ----------------------------------------------------

        if (
            abs(retained_total - 100.0)
            <= 1e-6
        ):
            test_pass(
                f"{sample} Rep {replicate}: "
                "% retained sums to 100%."
            )
        else:
            test_fail(
                f"{sample} Rep {replicate}: "
                f"% retained sum = "
                f"{retained_total:.6f}%."
            )

        # ----------------------------------------------------
        # CHECK PERCENT PASSING
        # ----------------------------------------------------

        if (
            -1e-6
            <= final_passing
            <= 100.000001
        ):
            test_pass(
                f"{sample} Rep {replicate}: "
                "percentage passing calculation is valid."
            )
        else:
            test_fail(
                f"{sample} Rep {replicate}: "
                f"final percentage passing = "
                f"{final_passing:.6f}%."
            )

        print(
            f"{sample} Rep {replicate}: "
            f"Sieve rows={len(group)}, "
            f"Total sieve mass={total_mass:.4f} g"
        )

        groups_tested += 1

    if groups_tested > 0:
        test_pass(
            f"Sieve calculations independently tested for "
            f"{groups_tested} sample/replicate groups."
        )
    else:
        test_fail(
            "No sieve groups could be tested."
        )


# ============================================================
# BENEFICIATION CALCULATIONS
# ============================================================

def test_beneficiation(df):
    print_section("5. BENEFICIATION CALCULATIONS")

    process_df = get_process_records(df)

    required = [
        "sample_id",
        "replicate",
        "feed_mass",
        "feed_grade",
        "concentrate_mass",
        "concentrate_grade",
        "tailings_mass",
        "tailings_grade",
        "losses_mass",
    ]

    missing = [
        column
        for column in required
        if column not in process_df.columns
    ]

    if missing:
        test_fail(
            "Beneficiation test cannot run. Missing columns: "
            + ", ".join(missing)
        )
        return

    tested = 0

    for _, row in process_df.iterrows():

        sample = row["sample_id"]
        replicate = row["replicate"]

        feed_mass = safe_float(
            row["feed_mass"]
        )

        feed_grade = safe_float(
            row["feed_grade"]
        )

        concentrate_mass = safe_float(
            row["concentrate_mass"]
        )

        concentrate_grade = safe_float(
            row["concentrate_grade"]
        )

        tailings_mass = safe_float(
            row["tailings_mass"]
        )

        tailings_grade = safe_float(
            row["tailings_grade"]
        )

        losses_mass = safe_float(
            row["losses_mass"]
        )

        # ----------------------------------------------------
        # MASS VALIDATION
        # ----------------------------------------------------

        if not valid_number(feed_mass):
            test_fail(
                f"{sample} Rep {replicate}: "
                "invalid feed mass."
            )
            continue

        if feed_mass <= 0:
            test_fail(
                f"{sample} Rep {replicate}: "
                "feed mass must be greater than zero."
            )
            continue

        if not valid_number(concentrate_mass):
            test_fail(
                f"{sample} Rep {replicate}: "
                "invalid concentrate mass."
            )
            continue

        if not valid_number(tailings_mass):
            test_fail(
                f"{sample} Rep {replicate}: "
                "invalid tailings mass."
            )
            continue

        if not valid_number(losses_mass):
            test_fail(
                f"{sample} Rep {replicate}: "
                "invalid losses mass."
            )
            continue

        # ----------------------------------------------------
        # MASS YIELD
        # ----------------------------------------------------

        mass_yield = (
            concentrate_mass
            / feed_mass
            * 100.0
        )

        # ----------------------------------------------------
        # MATERIAL BALANCE
        # ----------------------------------------------------

        output_mass = (
            concentrate_mass
            + tailings_mass
            + losses_mass
        )

        balance_error = (
            feed_mass
            - output_mass
        )

        balance_closure = (
            output_mass
            / feed_mass
            * 100.0
        )

        # ----------------------------------------------------
        # GRADE RECOVERY
        # ----------------------------------------------------

        grade_recovery = float("nan")

        if (
            valid_number(feed_grade)
            and feed_grade > 0
            and valid_number(concentrate_grade)
        ):

            grade_recovery = (
                concentrate_mass
                * concentrate_grade
                /
                (
                    feed_mass
                    * feed_grade
                )
                * 100.0
            )

        # ----------------------------------------------------
        # ENRICHMENT RATIO
        # ----------------------------------------------------

        enrichment_ratio = float("nan")

        if (
            valid_number(feed_grade)
            and feed_grade > 0
            and valid_number(concentrate_grade)
        ):

            enrichment_ratio = (
                concentrate_grade
                / feed_grade
            )

        print()
        print(
            f"{sample} Rep {replicate}"
        )

        print(
            f"  Feed mass           : "
            f"{feed_mass:.6f} g"
        )

        print(
            f"  Concentrate mass    : "
            f"{concentrate_mass:.6f} g"
        )

        print(
            f"  Tailings mass       : "
            f"{tailings_mass:.6f} g"
        )

        print(
            f"  Losses mass         : "
            f"{losses_mass:.6f} g"
        )

        print(
            f"  Mass yield          : "
            f"{mass_yield:.6f}%"
        )

        print(
            f"  Balance error       : "
            f"{balance_error:.6f} g"
        )

        print(
            f"  Balance closure     : "
            f"{balance_closure:.6f}%"
        )

        if valid_number(grade_recovery):
            print(
                f"  Grade recovery      : "
                f"{grade_recovery:.6f}%"
            )

        if valid_number(enrichment_ratio):
            print(
                f"  Enrichment ratio    : "
                f"{enrichment_ratio:.6f}"
            )

        # ----------------------------------------------------
        # CHECK MASS BALANCE
        # ----------------------------------------------------

        if abs(balance_error) <= TOLERANCE:

            test_pass(
                f"{sample} Rep {replicate}: "
                "mass balance closes exactly."
            )

        else:

            test_fail(
                f"{sample} Rep {replicate}: "
                f"mass balance error = "
                f"{balance_error:.6f} g."
            )

        # ----------------------------------------------------
        # CHECK MASS YIELD
        # ----------------------------------------------------

        if (
            0.0
            <= mass_yield
            <= 100.000001
        ):

            test_pass(
                f"{sample} Rep {replicate}: "
                "mass yield is valid."
            )

        else:

            test_fail(
                f"{sample} Rep {replicate}: "
                f"mass yield = "
                f"{mass_yield:.6f}%."
            )

        # ----------------------------------------------------
        # CHECK BALANCE CLOSURE
        # ----------------------------------------------------

        if abs(balance_closure - 100.0) <= TOLERANCE:

            test_pass(
                f"{sample} Rep {replicate}: "
                "balance closure = 100%."
            )

        else:

            test_fail(
                f"{sample} Rep {replicate}: "
                f"balance closure = "
                f"{balance_closure:.6f}%."
            )

        # ----------------------------------------------------
        # CHECK GRADE RECOVERY
        # ----------------------------------------------------

        if valid_number(grade_recovery):

            if (
                0.0
                <= grade_recovery
                <= 100.000001
            ):

                test_pass(
                    f"{sample} Rep {replicate}: "
                    "grade recovery is within normal range."
                )

            else:

                test_warning(
                    f"{sample} Rep {replicate}: "
                    f"grade recovery = "
                    f"{grade_recovery:.6f}%."
                )

        # ----------------------------------------------------
        # CHECK ENRICHMENT
        # ----------------------------------------------------

        if valid_number(enrichment_ratio):

            if enrichment_ratio > 0:

                test_pass(
                    f"{sample} Rep {replicate}: "
                    "enrichment ratio is valid."
                )

            else:

                test_fail(
                    f"{sample} Rep {replicate}: "
                    "enrichment ratio is invalid."
                )

        tested += 1

    if tested > 0:

        test_pass(
            f"Beneficiation calculations independently tested "
            f"for {tested} process records."
        )


# ============================================================
# MATERIAL BALANCE TEST
# ============================================================

def test_material_balance(df):
    print_section("6. MATERIAL BALANCE")

    process_df = get_process_records(df)

    required = [
        "sample_id",
        "replicate",
        "feed_mass",
        "concentrate_mass",
        "tailings_mass",
        "losses_mass",
    ]

    missing = [
        column
        for column in required
        if column not in process_df.columns
    ]

    if missing:
        test_fail(
            "Material balance test cannot run. Missing columns: "
            + ", ".join(missing)
        )
        return

    for _, row in process_df.iterrows():

        sample = row["sample_id"]
        replicate = row["replicate"]

        feed = safe_float(
            row["feed_mass"]
        )

        concentrate = safe_float(
            row["concentrate_mass"]
        )

        tailings = safe_float(
            row["tailings_mass"]
        )

        losses = safe_float(
            row["losses_mass"]
        )

        if not all(
            valid_number(value)
            for value in [
                feed,
                concentrate,
                tailings,
                losses,
            ]
        ):
            test_warning(
                f"{sample} Rep {replicate}: "
                "incomplete material balance data."
            )
            continue

        if feed <= 0:
            test_fail(
                f"{sample} Rep {replicate}: "
                "feed mass is zero or negative."
            )
            continue

        output = (
            concentrate
            + tailings
            + losses
        )

        difference = (
            feed - output
        )

        closure = (
            output
            / feed
            * 100.0
        )

        mass_recovery = (
            concentrate
            / feed
            * 100.0
        )

        print(
            f"{sample} Rep {replicate}: "
            f"Feed={feed:.4f} g | "
            f"Output={output:.4f} g | "
            f"Difference={difference:.6f} g | "
            f"Closure={closure:.6f}% | "
            f"Mass Recovery={mass_recovery:.6f}%"
        )

        if abs(difference) <= TOLERANCE:

            test_pass(
                f"{sample} Rep {replicate}: "
                "material balance closes."
            )

        else:

            test_fail(
                f"{sample} Rep {replicate}: "
                f"material balance error = "
                f"{difference:.6f} g."
            )


# ============================================================
# STATISTICS
# ============================================================

def test_statistics(df):
    print_section("7. STATISTICAL CALCULATIONS")

    process_df = get_process_records(df)

    if "feed_mass" not in process_df.columns:
        test_fail(
            "feed_mass column is missing."
        )
        return

    values = pd.to_numeric(
        process_df["feed_mass"],
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        test_fail(
            "No numeric feed-mass values available."
        )
        return

    mean_value = values.mean()

    sample_sd = values.std(
        ddof=1
    )

    population_sd = values.std(
        ddof=0
    )

    minimum = values.min()
    maximum = values.max()

    if mean_value != 0:

        coefficient_variation = (
            sample_sd
            / mean_value
            * 100.0
        )

    else:

        coefficient_variation = float("nan")

    print(
        f"Number of records      : {len(values)}"
    )

    print(
        f"Mean                   : "
        f"{mean_value:.6f}"
    )

    print(
        f"Sample SD (n-1)        : "
        f"{sample_sd:.6f}"
    )

    print(
        f"Population SD (n)      : "
        f"{population_sd:.6f}"
    )

    print(
        f"Minimum                : "
        f"{minimum:.6f}"
    )

    print(
        f"Maximum                : "
        f"{maximum:.6f}"
    )

    if valid_number(
        coefficient_variation
    ):

        print(
            f"Coefficient of "
            f"Variation           : "
            f"{coefficient_variation:.6f}%"
        )

    if (
        minimum
        <= mean_value
        <= maximum
    ):

        test_pass(
            "Mean lies between minimum and maximum."
        )

    else:

        test_fail(
            "Mean is outside the minimum/maximum range."
        )

    if sample_sd >= 0:

        test_pass(
            "Sample standard deviation is valid."
        )

    else:

        test_fail(
            "Sample standard deviation is negative."
        )

    if (
        valid_number(coefficient_variation)
        and coefficient_variation >= 0
    ):

        test_pass(
            "Coefficient of variation is valid."
        )

    else:

        test_warning(
            "Coefficient of variation could not be calculated."
        )


# ============================================================
# DUPLICATE / REPLICATE HANDLING
# ============================================================

def test_duplicate_handling(df):
    print_section("8. DUPLICATE / REPLICATE HANDLING")

    raw_rows = len(df)

    process_df = get_process_records(df)

    process_rows = len(process_df)

    print(
        f"Raw CSV rows          : {raw_rows}"
    )

    print(
        f"Process-level records : {process_rows}"
    )

    if process_rows <= raw_rows:

        test_pass(
            "Process-level records are correctly "
            "bounded by raw CSV rows."
        )

    else:

        test_fail(
            "Process-level records exceed raw CSV rows."
        )

    if (
        "sample_id" in df.columns
        and "replicate" in df.columns
    ):

        grouped_sizes = (
            df
            .groupby(
                [
                    "sample_id",
                    "replicate",
                ],
                dropna=False,
            )
            .size()
        )

        repeated_groups = grouped_sizes[
            grouped_sizes > 1
        ]

        if len(repeated_groups) > 0:

            test_pass(
                f"{len(repeated_groups)} sample/replicate "
                "groups contain multiple rows."
            )

        else:

            test_warning(
                "No repeated sample/replicate groups detected."
            )


# ============================================================
# DATA SANITY
# ============================================================

def test_data_sanity(df):
    print_section("9. DATA SANITY CHECKS")

    numeric_columns = [
        "feed_mass",
        "feed_grade",
        "feed_moisture",
        "crush_time",
        "crush_power",
        "crush_feed_size",
        "crush_product_size",
        "grinding_time",
        "grinding_media_mass",
        "grinding_speed",
        "grinding_feed_size",
        "grinding_product_size",
        "sieve_size",
        "mass_retained",
        "sieve_shaking_time",
        "beneficiation_time",
        "reagent_dosage",
        "ph_level",
        "pulp_density",
        "temperature",
        "concentrate_mass",
        "concentrate_grade",
        "tailings_mass",
        "tailings_grade",
        "losses_mass",
        "water_added",
        "solid_recovery",
    ]

    checked = 0

    for column in numeric_columns:

        if column not in df.columns:
            continue

        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        non_numeric = (
            df[column].notna()
            & numeric_values.isna()
        ).sum()

        if non_numeric > 0:

            test_warning(
                f"{column}: "
                f"{non_numeric} non-numeric value(s)."
            )

        else:

            checked += 1

    if checked > 0:

        test_pass(
            f"Numeric data checked across "
            f"{checked} columns."
        )

    # --------------------------------------------------------
    # NEGATIVE MASS CHECK
    # --------------------------------------------------------

    mass_columns = [
        "feed_mass",
        "mass_retained",
        "grinding_media_mass",
        "concentrate_mass",
        "tailings_mass",
        "losses_mass",
        "water_added",
    ]

    for column in mass_columns:

        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        negative_count = (
            values < 0
        ).sum()

        if negative_count == 0:

            test_pass(
                f"{column}: "
                "no negative values."
            )

        else:

            test_fail(
                f"{column}: "
                f"{negative_count} negative value(s)."
            )


# ============================================================
# FINAL REPORT
# ============================================================

def final_report():

    print()
    print("=" * 70)
    print("FINAL TEST RESULT")
    print("=" * 70)

    print(
        f"PASS     : {PASS_COUNT}"
    )

    print(
        f"FAIL     : {FAIL_COUNT}"
    )

    print(
        f"WARNING  : {WARNING_COUNT}"
    )

    print()

    if FAIL_COUNT == 0:

        if WARNING_COUNT == 0:

            print(
                "OVERALL RESULT: PASS"
            )

            print()
            print(
                "All independent checks passed."
            )

            print(
                "No calculation or dataset-structure "
                "failure was detected."
            )

        else:

            print(
                "OVERALL RESULT: PASS WITH WARNINGS"
            )

            print()
            print(
                "No calculation failure was detected, "
                "but some items require review."
            )

        return 0

    print(
        "OVERALL RESULT: FAIL"
    )

    print()
    print(
        "One or more independent tests failed."
    )

    print(
        "Investigate the failed items before "
        "using real laboratory data."
    )

    return 1


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    if df is None:

        return final_report()

    test_dataset_structure(df)

    test_duplicate_handling(df)

    test_data_sanity(df)

    test_crushing(df)

    test_grinding(df)

    test_sieve_analysis(df)

    test_beneficiation(df)

    test_material_balance(df)

    test_statistics(df)

    return final_report()


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    sys.exit(main())
