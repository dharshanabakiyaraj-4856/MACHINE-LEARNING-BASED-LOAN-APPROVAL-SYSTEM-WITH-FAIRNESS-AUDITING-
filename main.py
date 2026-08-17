"""
main.py
Bank Loan Approval Prediction - Interactive Terminal Console Application

Maintains backward compatibility while integrating advanced banking feature engineering,
0-100 credit risk scoring, SHAP waterfall plots, DiCE counterfactuals, and
Fairlearn auditing.
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

from utils.logger import get_logger
from preprocessing import (
    load_raw_data, preprocess, transform_input, CATEGORICAL_FEATURES, DEFAULT_INTEREST_RATE
)
from train import run_training
from risk import RiskAssessmentSystem
from explainability import (
    compute_individual_shap, save_shap_waterfall_plot,
    generate_natural_language_reasons, generate_counterfactual_explanation
)

warnings.filterwarnings("ignore")
logger = get_logger("MainConsole")

MODEL_PATH = "model.pkl"

VALID_OPTIONS = {
    "Gender"        : ["Male", "Female"],
    "Married"       : ["Yes", "No"],
    "Dependents"    : ["0", "1", "2", "3"],
    "Education"     : ["Graduate", "Not Graduate"],
    "Self_Employed" : ["Yes", "No"],
    "Property_Area" : ["Urban", "Semiurban", "Rural"],
}


def get_model_bundle() -> dict:
    if os.path.exists(MODEL_PATH):
        logger.info(f"Found '{MODEL_PATH}' -> Loading saved model bundle.")
        return joblib.load(MODEL_PATH)
    else:
        logger.info(f"'{MODEL_PATH}' not found -> Starting training pipeline...")
        return run_training()


def _prompt_categorical(field: str) -> str:
    options  = VALID_OPTIONS[field]
    opts_str = " / ".join(options)
    while True:
        val = input(f"  {field:<25} [{opts_str}] : ").strip()
        if val in options:
            return val
        print(f"    Invalid input. Please choose from: {opts_str}\n")


def _prompt_float(field: str, unit: str = "") -> float:
    label = f"{field} {unit}".strip()
    while True:
        raw = input(f"  {label:<40} : ").strip()
        try:
            return float(raw)
        except ValueError:
            print(f"    Please enter a valid number.\n")


def _prompt_credit() -> float:
    while True:
        raw = input("  Credit_History             [0 = bad / 1 = good] : ").strip()
        if raw in ("0", "1"):
            return float(raw)
        print("    Enter 0 or 1.\n")


def collect_applicant_input() -> dict:
    print("\n" + "="*70)
    print("  ENTER APPLICANT DETAILS")
    print("="*70 + "\n")

    data = {}
    data["Gender"]            = _prompt_categorical("Gender")
    data["Married"]           = _prompt_categorical("Married")
    data["Dependents"]        = _prompt_categorical("Dependents")
    data["Education"]         = _prompt_categorical("Education")
    data["Self_Employed"]     = _prompt_categorical("Self_Employed")
    data["ApplicantIncome"]   = _prompt_float("ApplicantIncome",   "(monthly, INR)")
    data["CoapplicantIncome"] = _prompt_float("CoapplicantIncome", "(monthly, INR)")
    data["LoanAmount"]        = _prompt_float("LoanAmount",        "(in thousands)")
    data["Loan_Amount_Term"]  = _prompt_float("Loan_Amount_Term",  "(in months)")
    data["Credit_History"]    = _prompt_credit()
    data["Property_Area"]     = _prompt_categorical("Property_Area")

    return data


def predict(bundle: dict, raw_input: dict):
    model         = bundle["model"]
    feature_names = bundle["feature_names"]

    engineered = transform_input(raw_input, annual_interest_rate=DEFAULT_INTEREST_RATE)

    row = {f: engineered.get(f, np.nan) for f in feature_names}
    X   = pd.DataFrame([row], columns=feature_names)

    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype(str)

    # Predictions & Risk Assessment
    risk_sys = RiskAssessmentSystem()
    risk_res = risk_sys.evaluate_applicant_risk(model, X)
    pred     = int(model.predict(X)[0])

    return pred, risk_res, X, engineered


def display_model_information(bundle: dict):
    cv_metrics = bundle.get("cv_metrics", {})

    print("\n" + "-"*65)
    print("  RESPONSIBLE AI BANKING SYSTEM  -  MODEL INFORMATION")
    print("-"*65)
    print(f"  {'Model Architecture':<25} : CatBoostClassifier")
    print(f"  {'Validation Methodology':<25} : 5-Fold Stratified Cross-Validation")
    print(f"  {'Mean CV Accuracy':<25} : {cv_metrics.get('mean_accuracy', 0):.4f} ± {cv_metrics.get('std_accuracy', 0):.4f}")
    print(f"  {'Mean CV ROC-AUC':<25} : {cv_metrics.get('mean_roc_auc', 0):.4f} ± {cv_metrics.get('std_roc_auc', 0):.4f}")
    print(f"  {'Mean CV F1-Score':<25} : {cv_metrics.get('mean_f1', 0):.4f} ± {cv_metrics.get('std_f1', 0):.4f}")
    print("-"*65)


def display_prediction_results(pred: int, risk_res: dict, engineered: dict):
    label = "APPROVED" if pred == 1 else "REJECTED"

    print("\n" + "="*70)
    print("  PREDICTION RESULT & BANKING METRICS")
    print("="*70)
    print(f"\n  Loan Status          : *** {label} ***")
    print(f"  Approval Probability : {risk_res['approval_probability'] * 100:.2f}%")
    print(f"  Credit Risk Score    : {risk_res['risk_score']} / 100")
    print(f"  Risk Category        : {risk_res['risk_category']}")

    print("\n  Banking Analysis:")
    print("  -----------------")
    print(f"  Monthly EMI          : ₹{np.expm1(engineered.get('EMI', 0)):,.2f}")
    print(f"  Debt-to-Income (DTI) : {engineered.get('DTI', 0) * 100:.2f}%")
    print(f"  Loan-to-Annual-Inc   : {engineered.get('Loan_To_Income', 0):.2f}x")
    print(f"  Balance Disposable   : ₹{engineered.get('Balance_Income', 0):,.2f} / month")


def display_explanations(model, X_single: pd.DataFrame, X_train: pd.DataFrame):
    df_shap = compute_individual_shap(model, X_single)
    reasons = generate_natural_language_reasons(df_shap)

    print("\n" + "="*70)
    print("  EXPLAINABLE AI (SHAP & DIALOGUE REASONS)")
    print("="*70)

    print("\n  Primary Reasons for Approval:")
    for r in reasons["positive"]:
        print(f"  {r}")

    print("\n  Risk Factors / Negative Influences:")
    for r in reasons["negative"]:
        print(f"  {r}")

    print("\n  SHAP Feature Impact Breakdown:")
    print("  " + "-"*60)
    print(f"  {'Feature':<25} {'SHAP Impact':>12}   Direction")
    print("  " + "-"*60)
    for _, row in df_shap.head(6).iterrows():
        direction = "-> APPROVE" if row["SHAP"] > 0 else "-> REJECT "
        print(f"  {row['English_Name']:<25} {row['SHAP']:>+12.4f}   {direction}")

    # Counterfactual Recommendations
    cf_res = generate_counterfactual_explanation(model, X_train, X_single)
    print("\n  Counterfactual Recommendations (DiCE ML):")
    print("  -----------------------------------------")
    for rec in cf_res.get("recommendations", []):
        print(f"  💡 {rec}")

    # Save waterfall plot
    save_shap_waterfall_plot(model, X_single, "shap_waterfall.png")


def display_fairness_audit(bundle: dict):
    print("\n" + "="*70)
    print("  RESPONSIBLE AI FAIRNESS AUDIT REPORT (FAIRLEARN)")
    print("="*70)

    audit = bundle.get("fairness_audit", {})
    if not audit:
        print("  Fairness audit report not found.")
        return

    for attr, data in audit.items():
        print(f"\n  Sensitive Attribute : {attr}")
        print(f"    Demographic Parity Diff   : {data.get('dpd', 0):+.4f}")
        print(f"    Equal Opportunity Diff    : {data.get('equal_opportunity_diff', 0):.4f}")
        print(f"    Equalized Odds Diff       : {data.get('equalized_odds_diff', 0):.4f}")

    mit_report = bundle.get("bias_mitigation_report", {})
    if mit_report:
        print("\n  Bias Mitigation Performance (ThresholdOptimizer):")
        df_mit = pd.DataFrame(mit_report)
        print(df_mit.to_string(index=False))


def print_banner():
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#       EXPLAINABLE & FAIR BANK LOAN APPROVAL SYSTEM               #")
    print("#       CatBoost  |  SHAP Waterfall  |  Fairlearn  |  DiCE         #")
    print("#" + " "*68 + "#")
    print("#"*70)


def main():
    print_banner()

    bundle = get_model_bundle()
    display_model_information(bundle)

    model   = bundle["model"]
    X_train = bundle.get("X_train_sample", pd.DataFrame())

    while True:
        try:
            raw_input = collect_applicant_input()
            pred, risk_res, X_single, engineered = predict(bundle, raw_input)

            display_prediction_results(pred, risk_res, engineered)
            display_explanations(model, X_single, X_train)
            display_fairness_audit(bundle)

            print("\n" + "="*70)
            again = input("  Evaluate another applicant? [yes / no] : ").strip().lower()
            if again not in ("yes", "y"):
                print("\n  Thank you for using the Responsible AI Banking System. Goodbye!\n")
                break

        except KeyboardInterrupt:
            print("\n\n  Session interrupted. Exiting...\n")
            sys.exit(0)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            break


if __name__ == "__main__":
    main()
