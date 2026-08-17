"""
risk.py
Risk Assessment and Probability Calibration System for Bank Loan Approval
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from utils.logger import get_logger

logger = get_logger("RiskSystem")

class PlattCalibrator:
    """
    Platt Scaling Probability Calibrator.
    Fits a sigmoid logistic regression on raw model probability outputs.
    """
    def __init__(self):
        self.calibrator = LogisticRegression(C=1.0)
        self.is_fitted = False

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray):
        raw_probs = np.array(raw_probs).reshape(-1, 1)
        self.calibrator.fit(raw_probs, y_true)
        self.is_fitted = True
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return raw_probs
        raw_probs = np.array(raw_probs).reshape(-1, 1)
        return self.calibrator.predict_proba(raw_probs)[:, 1]


class RiskAssessmentSystem:
    """
    Computes calibrated approval probabilities, risk scores (0-100),
    and risk category buckets for loan applicants.
    """

    def __init__(self, high_risk_max: int = 40, medium_risk_max: int = 70):
        self.high_risk_max = high_risk_max
        self.medium_risk_max = medium_risk_max

    def calibrate_model(self, model, X_val: pd.DataFrame, y_val: pd.Series) -> PlattCalibrator:
        """
        Fit Platt scaling calibrator on validation probabilities.
        """
        raw_probs = model.predict_proba(X_val)[:, 1]
        calibrator = PlattCalibrator()
        calibrator.fit(raw_probs, y_val)
        logger.info("Successfully fitted Platt scaling probability calibrator.")
        return calibrator

    def compute_risk_score(self, prob_approve: float) -> tuple:
        """
        Calculate 0-100 Risk Score and determine Category.

        Parameters:
        -----------
        prob_approve: Calibrated probability of approval (0.0 to 1.0)

        Returns:
        --------
        (score: int, category: str, color_code: str)
        """
        prob_approve = float(np.clip(prob_approve, 0.0, 1.0))
        score = int(round(prob_approve * 100.0))

        if score <= self.high_risk_max:
            category = "High Risk"
            color = "#EF4444"  # Red
        elif score <= self.medium_risk_max:
            category = "Medium Risk"
            color = "#F59E0B"  # Yellow/Amber
        else:
            category = "Low Risk"
            color = "#10B981"  # Green

        return score, category, color

    def evaluate_applicant_risk(self, model, X_applicant: pd.DataFrame, calibrator=None) -> dict:
        """
        Evaluate an applicant row and return complete risk metrics.
        """
        probs = model.predict_proba(X_applicant)[0]
        raw_prob_approve = float(probs[1])

        if calibrator is not None and getattr(calibrator, "is_fitted", False):
            prob_approve = float(calibrator.predict_proba([raw_prob_approve])[0])
        else:
            prob_approve = raw_prob_approve

        prob_reject = 1.0 - prob_approve
        score, category, color = self.compute_risk_score(prob_approve)

        return {
            "approval_probability": prob_approve,
            "raw_probability": raw_prob_approve,
            "rejection_probability": prob_reject,
            "risk_score": score,
            "risk_category": category,
            "color_code": color,
        }
