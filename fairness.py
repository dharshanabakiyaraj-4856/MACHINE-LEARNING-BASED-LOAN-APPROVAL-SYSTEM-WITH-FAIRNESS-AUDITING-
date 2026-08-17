"""
fairness.py
Advanced Fairness Auditing & Bias Mitigation Module using Fairlearn
"""

import numpy as np
import pandas as pd
from utils.logger import get_logger

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from fairlearn.metrics import (
    MetricFrame, selection_rate, true_positive_rate, false_positive_rate,
    demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference
)
from fairlearn.postprocessing import ThresholdOptimizer

logger = get_logger("FairnessAuditor")


def compute_equal_opportunity_difference(y_true, y_pred, sensitive_features) -> float:
    """Calculate Equal Opportunity Difference (difference in True Positive Rate across groups)."""
    mf = MetricFrame(
        metrics={"tpr": true_positive_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    tprs = mf.by_group["tpr"].values
    if len(tprs) < 2:
        return 0.0
    return float(np.max(tprs) - np.min(tprs))


def audit_fairness(model, X_test: pd.DataFrame, y_test: pd.Series, sens_df: pd.DataFrame) -> dict:
    """
    Perform comprehensive fairness auditing across sensitive attributes:
    Gender, Married, Education, and Intersectional group (Gender_Married).
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    results = {}

    # 1. Prepare sensitive features including intersectional feature
    sens_audit = sens_df.copy()
    if "Gender" in sens_audit.columns and "Married" in sens_audit.columns:
        sens_audit["Gender_Married"] = sens_audit["Gender"].astype(str) + "_" + sens_audit["Married"].astype(str)

    attributes_to_audit = [c for c in ["Gender", "Married", "Education", "Gender_Married"] if c in sens_audit.columns]

    for attr in attributes_to_audit:
        sens = sens_audit[attr]

        # MetricFrame for group selection rates and TPRs
        mf = MetricFrame(
            metrics={
                "selection_rate": selection_rate,
                "tpr": true_positive_rate,
                "fpr": false_positive_rate,
            },
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sens
        )

        dpd = float(demographic_parity_difference(y_test, y_pred, sensitive_features=sens))
        dpr = float(demographic_parity_ratio(y_test, y_pred, sensitive_features=sens))
        eod = compute_equal_opportunity_difference(y_test, y_pred, sensitive_features=sens)
        eq_odds = float(equalized_odds_difference(y_test, y_pred, sensitive_features=sens))

        results[attr] = {
            "dpd": dpd,
            "dpr": dpr,
            "equal_opportunity_diff": eod,
            "equalized_odds_diff": eq_odds,
            "selection_rates": mf.by_group["selection_rate"].to_dict(),
            "tpr_by_group": mf.by_group["tpr"].to_dict(),
            "fpr_by_group": mf.by_group["fpr"].to_dict(),
            "overall_selection_rate": float(mf.overall["selection_rate"]),
        }

    return results


def mitigate_bias(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, sens_train: pd.Series, sens_test: pd.Series, constraint: str = "demographic_parity") -> tuple:
    """
    Mitigate algorithmic bias using Fairlearn's ThresholdOptimizer.

    Returns:
    --------
    (mitigated_model, comparison_report_dict, comparison_df)
    """
    logger.info(f"Applying ThresholdOptimizer with constraint: {constraint}")

    # Baseline performance (before mitigation)
    y_pred_orig = model.predict(X_test)
    y_prob_orig = model.predict_proba(X_test)[:, 1]

    acc_orig = accuracy_score(y_test, y_pred_orig)
    auc_orig = roc_auc_score(y_test, y_prob_orig)
    dpd_orig = float(demographic_parity_difference(y_test, y_pred_orig, sensitive_features=sens_test))
    eod_orig = compute_equal_opportunity_difference(y_test, y_pred_orig, sensitive_features=sens_test)
    eq_orig  = float(equalized_odds_difference(y_test, y_pred_orig, sensitive_features=sens_test))

    # Fit ThresholdOptimizer
    opt = ThresholdOptimizer(
        estimator=model,
        constraints=constraint,
        predict_method="predict_proba",
        prefit=True
    )
    opt.fit(X_train, y_train, sensitive_features=sens_train)

    y_pred_mit = opt.predict(X_test, sensitive_features=sens_test)

    acc_mit = accuracy_score(y_test, y_pred_mit)
    # Note: ThresholdOptimizer outputs binary decisions, compute pseudo AUC or accuracy
    auc_mit = roc_auc_score(y_test, y_pred_mit)
    dpd_mit = float(demographic_parity_difference(y_test, y_pred_mit, sensitive_features=sens_test))
    eod_mit = compute_equal_opportunity_difference(y_test, y_pred_mit, sensitive_features=sens_test)
    eq_mit  = float(equalized_odds_difference(y_test, y_pred_mit, sensitive_features=sens_test))

    comparison = {
        "Metric": [
            "Accuracy",
            "ROC-AUC",
            "Demographic Parity Difference",
            "Equal Opportunity Difference",
            "Equalized Odds Difference"
        ],
        "Before Mitigation": [acc_orig, auc_orig, dpd_orig, eod_orig, eq_orig],
        "After Mitigation": [acc_mit, auc_mit, dpd_mit, eod_mit, eq_mit],
        "Improvement / Change": [
            acc_mit - acc_orig,
            auc_mit - auc_orig,
            abs(dpd_orig) - abs(dpd_mit),
            abs(eod_orig) - abs(eod_mit),
            abs(eq_orig) - abs(eq_mit),
        ]
    }

    df_comparison = pd.DataFrame(comparison)

    return opt, comparison, df_comparison
