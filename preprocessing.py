"""
preprocessing.py
Bank Loan Approval Prediction - Shared Preprocessing Module

This module contains the complete data cleaning, missing value imputation,
banking feature engineering, log transformations, and inference pipeline
used by train.py, main.py, app.py, and unit tests.
"""

import sys
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger("Preprocessing")

# --- Constants ----------------------------------------------------------------

CATEGORICAL_FEATURES = [
    "Gender", "Married", "Dependents",
    "Education", "Self_Employed", "Property_Area",
]

NUMERICAL_FILL_COLS = ["LoanAmount", "Loan_Amount_Term", "Credit_History"]

# Columns subject to log1p transformation to reduce right skewness
LOG_TRANSFORM_COLS = ["ApplicantIncome", "LoanAmount", "TotalIncome", "EMI"]

# Default interest rate for EMI estimation if not provided (9.5% per annum)
DEFAULT_INTEREST_RATE = 9.5


# --- Banking Feature Helper Functions ---------------------------------------

def calculate_emi(loan_amount_k: float, term_months: float, annual_interest_rate: float = DEFAULT_INTEREST_RATE) -> float:
    """
    Calculate Monthly Equated Monthly Installment (EMI).

    Parameters:
    -----------
    loan_amount_k        : Loan amount in thousands (e.g. 128 means 128,000 INR).
    term_months          : Repayment term in months (e.g. 360).
    annual_interest_rate : Annual interest rate in percent (e.g. 9.5).

    Returns:
    --------
    Monthly EMI value.
    """
    principal = float(loan_amount_k) * 1000.0
    n = float(term_months) if term_months > 0 else 360.0

    if annual_interest_rate <= 0:
        return principal / n

    r = (annual_interest_rate / 100.0) / 12.0  # Monthly interest rate
    
    try:
        emi = principal * r * ((1 + r) ** n) / (((1 + r) ** n) - 1)
    except (ZeroDivisionError, OverflowError):
        emi = principal / n

    return max(0.0, emi)


# --- Data Loading -----------------------------------------------------------

def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw Kaggle Loan Prediction CSV dataset."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error(f"Dataset not found at '{path}'. Please place train.csv in the dataset/ folder.")
        sys.exit(1)

    logger.info(f"Loaded '{path}' -> {df.shape[0]} rows x {df.shape[1]} columns")

    if "Loan_Status" not in df.columns:
        logger.error("'Loan_Status' column not found in dataset. Please ensure Kaggle train.csv is used.")
        sys.exit(1)

    return df


# --- Full Preprocessing Pipeline ---------------------------------------------

def preprocess(df: pd.DataFrame, imputation_stats: dict = None, verbose: bool = True, annual_interest_rate: float = DEFAULT_INTEREST_RATE) -> tuple:
    """
    Run the complete preprocessing & feature engineering pipeline.

    Parameters
    ----------
    df                   : Raw DataFrame.
    imputation_stats     : Dictionary of medians/modes. If None, learns them from df.
    verbose              : Log progress if True.
    annual_interest_rate : Annual interest rate for EMI calculations.

    Returns
    -------
    (Cleaned DataFrame, imputation_stats)
    """
    df = df.copy()

    def log(msg):
        if verbose:
            logger.info(msg)

    # 1. Remove duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log(f"Duplicates removed: {before - len(df)} row(s) dropped")

    # 2. Drop Loan_ID identifier if present
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])
        log("Dropped 'Loan_ID' column")

    # 3. Learn Imputation Statistics (prevent data leakage during training)
    if imputation_stats is None:
        imputation_stats = {}
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                imputation_stats[col] = df[col].mode()[0]
        for col in NUMERICAL_FILL_COLS:
            if col in df.columns:
                imputation_stats[col] = df[col].median()
        log("Learned imputation statistics from training data")

    # 4. Apply Imputation
    for col in CATEGORICAL_FEATURES + NUMERICAL_FILL_COLS:
        if col in df.columns and df[col].isnull().any():
            n_missing = df[col].isnull().sum()
            df[col] = df[col].fillna(imputation_stats[col])
            log(f"Imputed '{col}' ({n_missing} missing) -> {imputation_stats[col]}")

    # 5. Clean Dependents ("3+" -> "3")
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace("3+", "3")
        log("Converted '3+' -> '3' in Dependents")

    # 6. Ensure categorical columns are strings for CatBoost compatibility
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # 7. Banking Feature Engineering (computed on raw numeric scale BEFORE log1p)
    if "ApplicantIncome" in df.columns and "CoapplicantIncome" in df.columns:
        df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
        log("Created 'TotalIncome' = ApplicantIncome + CoapplicantIncome")

    if all(col in df.columns for col in ["LoanAmount", "Loan_Amount_Term", "TotalIncome"]):
        # a. EMI Estimation
        df["EMI"] = df.apply(
            lambda row: calculate_emi(row["LoanAmount"], row["Loan_Amount_Term"], annual_interest_rate),
            axis=1
        )
        log("Created 'EMI' (Equated Monthly Installment)")

        # b. Debt-to-Income Ratio (DTI = EMI / TotalIncome)
        df["DTI"] = df["EMI"] / (df["TotalIncome"] + 1e-5)
        log("Created 'DTI' (Debt-to-Income Ratio)")

        # c. Loan Amount to Annual Income Ratio (Loan_To_Income)
        df["Loan_To_Income"] = (df["LoanAmount"] * 1000.0) / ((df["TotalIncome"] * 12.0) + 1e-5)
        log("Created 'Loan_To_Income' ratio")

        # d. Balance Income after EMI (Balance_Income = TotalIncome - EMI)
        df["Balance_Income"] = df["TotalIncome"] - df["EMI"]
        log("Created 'Balance_Income' = TotalIncome - EMI")

    # 8. Log1p Transformations for right-skewed variables
    for col in LOG_TRANSFORM_COLS:
        if col in df.columns:
            df[col] = np.log1p(np.maximum(0, df[col]))
            log(f"Applied log1p() transform to '{col}'")

    # 9. Target Encoding
    if "Loan_Status" in df.columns:
        df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
        counts = df["Loan_Status"].value_counts().to_dict()
        log(f"Encoded Loan_Status -> Approved(1)={counts.get(1, 0)}, Rejected(0)={counts.get(0, 0)}")

    log(f"Final preprocessed shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return df, imputation_stats


# --- Single-Row Inference Transform ------------------------------------------

def transform_input(raw: dict, annual_interest_rate: float = DEFAULT_INTEREST_RATE) -> dict:
    """
    Apply identical banking feature engineering and log transforms to a single
    raw input dictionary during inference.
    """
    d = dict(raw)

    # Clean categorical inputs
    d["Dependents"] = str(d.get("Dependents", "0")).replace("3+", "3")

    app_inc = float(d.get("ApplicantIncome", 0.0))
    co_inc  = float(d.get("CoapplicantIncome", 0.0))
    loan_amt = float(d.get("LoanAmount", 0.0))
    term    = float(d.get("Loan_Amount_Term", 360.0))

    # Raw banking metrics
    tot_inc = app_inc + co_inc
    emi     = calculate_emi(loan_amt, term, annual_interest_rate)
    dti     = emi / (tot_inc + 1e-5)
    lti     = (loan_amt * 1000.0) / ((tot_inc * 12.0) + 1e-5)
    bal_inc = tot_inc - emi

    d["TotalIncome"]     = tot_inc
    d["EMI"]             = emi
    d["DTI"]             = dti
    d["Loan_To_Income"]  = lti
    d["Balance_Income"]  = bal_inc

    # Apply log1p transforms identically to pipeline
    d["ApplicantIncome"] = float(np.log1p(max(0.0, app_inc)))
    d["LoanAmount"]      = float(np.log1p(max(0.0, loan_amt)))
    d["TotalIncome"]     = float(np.log1p(max(0.0, tot_inc)))
    d["EMI"]             = float(np.log1p(max(0.0, emi)))

    return d
