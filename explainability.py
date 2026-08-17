"""
explainability.py
Explainable AI (XAI) Module providing SHAP Waterfall Plots, Natural Language
Decision Reasons, and DiCE ML Counterfactual Explanations.
"""

import os
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.logger import get_logger

logger = get_logger("Explainability")

FEATURE_ENGLISH_MAP = {
    "Credit_History": "Credit History",
    "TotalIncome": "Total Income",
    "ApplicantIncome": "Applicant Income",
    "CoapplicantIncome": "Coapplicant Income",
    "LoanAmount": "Loan Amount",
    "Loan_Amount_Term": "Loan Repayment Term",
    "Property_Area": "Property Area",
    "Education": "Education Level",
    "Dependents": "Number of Dependents",
    "Self_Employed": "Employment Status",
    "EMI": "Monthly EMI",
    "DTI": "Debt-to-Income Ratio",
    "Loan_To_Income": "Loan-to-Income Ratio",
    "Balance_Income": "Disposable Balance Income",
}

POSITIVE_REASONS = {
    "Credit_History": "Good Credit History (No defaults)",
    "TotalIncome": "Sufficient Household Income",
    "ApplicantIncome": "Strong Primary Income",
    "CoapplicantIncome": "Supporting Co-applicant Income",
    "LoanAmount": "Reasonable Loan Amount relative to income",
    "Loan_Amount_Term": "Manageable Repayment Tenure",
    "Education": "Graduate Level Education",
    "Property_Area": "Favorable Property Location",
    "Self_Employed": "Stable Employment Status",
    "Dependents": "Favorable Family Dependents Ratio",
    "EMI": "Low Estimated Monthly EMI",
    "DTI": "Healthy Debt-to-Income Ratio",
    "Loan_To_Income": "Low Loan-to-Income Multiple",
    "Balance_Income": "Strong Post-EMI Disposable Income",
}

NEGATIVE_REASONS = {
    "Credit_History": "Poor or Missing Credit History",
    "TotalIncome": "Insufficient Combined Monthly Income",
    "ApplicantIncome": "Low Applicant Base Income",
    "LoanAmount": "High Requested Loan Amount",
    "Loan_Amount_Term": "Unfavorable Repayment Term",
    "Property_Area": "Property Location Risk Profile",
    "Education": "Non-Graduate Risk Category",
    "Dependents": "High Number of Dependents",
    "Self_Employed": "Self-Employed Variable Income Risk",
    "EMI": "High Estimated Monthly EMI Payment",
    "DTI": "Elevated Debt-to-Income (DTI) Ratio",
    "Loan_To_Income": "High Loan-to-Income Ratio",
    "Balance_Income": "Low Remaining Disposable Income after EMI",
}


def compute_individual_shap(model, X_single: pd.DataFrame) -> pd.DataFrame:
    """Compute individual SHAP feature contributions for a single applicant."""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_single)

    sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
    feature_names = list(X_single.columns)

    df_shap = pd.DataFrame({
        "Feature": feature_names,
        "English_Name": [FEATURE_ENGLISH_MAP.get(f, f) for f in feature_names],
        "Value": [X_single[f].values[0] for f in feature_names],
        "SHAP": sv,
        "Abs_SHAP": np.abs(sv),
    }).sort_values("Abs_SHAP", ascending=False).reset_index(drop=True)

    return df_shap


def save_shap_waterfall_plot(model, X_single: pd.DataFrame, save_path: str = "shap_waterfall.png"):
    """Generate and save a SHAP waterfall plot for an applicant."""
    try:
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X_single)

        # Handle binary classification slice
        if len(explanation.shape) == 3:
            exp_single = explanation[0, :, 1]
        else:
            exp_single = explanation[0]

        plt.figure(figsize=(9, 5))
        shap.plots.waterfall(exp_single, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved SHAP waterfall plot to '{save_path}'")
        return save_path
    except Exception as e:
        logger.warning(f"Could not generate SHAP waterfall plot: {e}")
        return None


def generate_natural_language_reasons(df_shap: pd.DataFrame) -> dict:
    """
    Generate natural language positive and negative decision factors.
    """
    pos_df = df_shap[df_shap["SHAP"] > 0]
    neg_df = df_shap[df_shap["SHAP"] < 0]

    pos_reasons = []
    for _, row in pos_df.iterrows():
        f = row["Feature"]
        desc = POSITIVE_REASONS.get(f, f"{FEATURE_ENGLISH_MAP.get(f, f)} positively contributed")
        pos_reasons.append(f"+ {desc}")

    neg_reasons = []
    for _, row in neg_df.iterrows():
        f = row["Feature"]
        desc = NEGATIVE_REASONS.get(f, f"{FEATURE_ENGLISH_MAP.get(f, f)} increased risk")
        neg_reasons.append(f"- {desc}")

    return {
        "positive": pos_reasons if pos_reasons else ["+ Meets basic lending criteria"],
        "negative": neg_reasons if neg_reasons else ["- No major risk flags identified"],
    }


def generate_counterfactual_explanation(model, X_train: pd.DataFrame, X_single: pd.DataFrame, target_column: str = "Loan_Status") -> dict:
    """
    Generate counterfactual explanation using DiCE ML or smart feature perturbation.

    Returns suggestions on what changes would flip or improve the loan decision.
    """
    cfs_list = []
    try:
        import dice_ml
        # Continuous and categorical features
        cat_features = [c for c in X_single.columns if X_single[c].dtype == "object" or str(X_single[c].dtype).startswith("str")]
        continuous_features = [c for c in X_single.columns if c not in cat_features]

        # Construct synthetic dataframe for DiCE Data container
        df_dice_train = X_train.copy()
        # Mock target if missing
        if target_column not in df_dice_train.columns:
            df_dice_train[target_column] = model.predict(X_train)

        d = dice_ml.Data(
            dataframe=df_dice_train,
            continuous_features=continuous_features,
            outcome_name=target_column
        )
        m = dice_ml.Model(model=model, backend="sklearn")
        exp = dice_ml.Dice(d, m, method="random")

        # Generate 1 counterfactual
        dice_exp = exp.generate_counterfactuals(X_single, total_CFs=1, desired_class="opposite")
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df

        if cf_df is not None and not cf_df.empty:
            diffs = []
            for col in X_single.columns:
                orig_val = X_single[col].values[0]
                cf_val   = cf_df[col].values[0]
                if orig_val != cf_val:
                    diffs.append(f"Change {FEATURE_ENGLISH_MAP.get(col, col)} from {orig_val} to {cf_val}")
            if diffs:
                return {"status": "success", "recommendations": diffs, "cf_dataframe": cf_df}

    except Exception as e:
        logger.info(f"DiCE ML generator fallback: {e}")

    # Robust Fallback Counterfactual Generator
    recommendations = []
    credit_val = X_single.get("Credit_History", pd.Series([1.0])).values[0]
    income_val = X_single.get("TotalIncome", pd.Series([0.0])).values[0]
    loan_val   = X_single.get("LoanAmount", pd.Series([0.0])).values[0]

    if credit_val == 0.0:
        recommendations.append("Improve Credit History score to 1.0 (clear outstanding default records).")
    
    recommendations.append("Increase monthly total household income by 15-25%.")
    recommendations.append("Decrease requested loan amount by 10-20% or extend loan repayment tenure.")

    return {
        "status": "fallback",
        "recommendations": recommendations,
        "summary": "If income increases or loan amount decreases, approval probability improves."
    }
