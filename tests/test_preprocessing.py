"""
tests/test_preprocessing.py
Unit tests for data preprocessing and banking feature engineering.
"""

import unittest
import numpy as np
import pandas as pd

from preprocessing import (
    calculate_emi, preprocess, transform_input, CATEGORICAL_FEATURES
)

class TestPreprocessing(unittest.TestCase):

    def test_calculate_emi(self):
        # 100k loan, 360 months, 9.5% annual interest
        emi = calculate_emi(loan_amount_k=100, term_months=360, annual_interest_rate=9.5)
        self.assertGreater(emi, 0)
        # Expected EMI for 100,000 INR at 9.5% for 30 years is approx 840.85 INR/month
        self.assertAlmostEqual(emi, 840.85, delta=15.0)

    def test_preprocess_pipeline(self):
        raw_data = pd.DataFrame({
            "Loan_ID": ["LP001", "LP002"],
            "Gender": ["Male", np.nan],
            "Married": ["Yes", "No"],
            "Dependents": ["0", "3+"],
            "Education": ["Graduate", "Not Graduate"],
            "Self_Employed": ["No", "Yes"],
            "ApplicantIncome": [5000, 3000],
            "CoapplicantIncome": [1500, 0],
            "LoanAmount": [128, 100],
            "Loan_Amount_Term": [360, 180],
            "Credit_History": [1.0, np.nan],
            "Property_Area": ["Urban", "Rural"],
            "Loan_Status": ["Y", "N"]
        })

        df_clean, imp_stats = preprocess(raw_data, verbose=False)

        # Check missing values imputed
        self.assertEqual(df_clean.isnull().sum().sum(), 0)

        # Check new banking features created
        for col in ["TotalIncome", "EMI", "DTI", "Loan_To_Income", "Balance_Income"]:
            self.assertIn(col, df_clean.columns)

        # Check Dependents 3+ converted
        self.assertNotIn("3+", df_clean["Dependents"].values)
        self.assertEqual(df_clean["Loan_Status"].tolist(), [1, 0])

    def test_transform_input_consistency(self):
        raw_input = {
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "3+",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 6000.0,
            "CoapplicantIncome": 2000.0,
            "LoanAmount": 150.0,
            "Loan_Amount_Term": 360.0,
            "Credit_History": 1.0,
            "Property_Area": "Urban",
        }

        transformed = transform_input(raw_input, annual_interest_rate=9.5)

        self.assertEqual(transformed["Dependents"], "3")
        self.assertGreater(transformed["TotalIncome"], 0)
        self.assertIn("EMI", transformed)
        self.assertIn("DTI", transformed)
        self.assertIn("Loan_To_Income", transformed)
        self.assertIn("Balance_Income", transformed)

if __name__ == "__main__":
    unittest.main()
