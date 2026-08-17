"""
app.py
Industry-Level Responsible AI Banking System - Streamlit Web Dashboard

Features:
- Page 1: Applicant Input Form & Standalone Loan EMI Calculator
- Page 2: Loan Prediction Result & Calibrated Risk Assessment (0-100)
- Page 3: Explainable AI (SHAP Waterfall Plots + DiCE Counterfactuals)
- Page 4: Responsible AI Fairness & Bias Mitigation Dashboard (Fairlearn)
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from preprocessing import transform_input, CATEGORICAL_FEATURES, calculate_emi, DEFAULT_INTEREST_RATE
from risk import RiskAssessmentSystem
from explainability import (
    compute_individual_shap, save_shap_waterfall_plot,
    generate_natural_language_reasons, generate_counterfactual_explanation
)

# Set page configuration
st.set_page_config(
    page_title="Responsible AI Banking System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "model.pkl"

# Custom Styling (Glassmorphism & Vibrant Banking Theme)
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #4B5563;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .card {
        background: rgba(255, 255, 255, 0.85);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #E5E7EB;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1F2937;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_bundle():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def main():
    st.markdown('<div class="main-title">🏦 Responsible AI Bank Loan System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Explainable, Fair, and Calibrated Credit Risk Scoring Platform</div>', unsafe_allow_html=True)

    bundle = load_bundle()
    if bundle is None:
        st.error("⚠️ Model bundle (`model.pkl`) not found! Please run `python train.py` first to train the system.")
        return

    model = bundle["model"]
    feature_names = bundle["feature_names"]
    X_train = bundle.get("X_train_sample", pd.DataFrame())

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        [
            "📋 1. Applicant Form & EMI Calc",
            "📊 2. Loan Prediction & Risk",
            "🔍 3. Explainability & Counterfactuals",
            "⚖️ 4. Fairness & Bias Mitigation"
        ]
    )

    # -------------------------------------------------------------------------
    # PAGE 1: APPLICANT INPUT FORM & STANDALONE EMI CALCULATOR
    # -------------------------------------------------------------------------
    if page.startswith("📋"):
        st.header("📋 Applicant Information & Loan EMI Calculator")
        st.write("Fill in the applicant's details below to compute financial ratios and execute loan assessment.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Personal & Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_emp = st.selectbox("Self Employed", ["No", "Yes"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        with col2:
            st.subheader("Financial & Credit Details")
            app_income = st.number_input("Applicant Monthly Income (INR)", min_value=0.0, value=5000.0, step=500.0)
            co_income = st.number_input("Co-applicant Monthly Income (INR)", min_value=0.0, value=1500.0, step=500.0)
            loan_amt = st.number_input("Requested Loan Amount (in Thousands INR)", min_value=1.0, value=128.0, step=10.0)
            term_months = st.number_input("Loan Term (Months)", min_value=12.0, value=360.0, step=12.0)
            interest_rate = st.number_input("Annual Interest Rate (%)", min_value=1.0, value=9.5, step=0.25)
            credit_hist = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good (1.0)" if x == 1.0 else "Bad/Default (0.0)")

        # Real Banking EMI Calculator Block
        st.markdown("---")
        st.subheader("🧮 Real Banking EMI Calculator")

        raw_emi = calculate_emi(loan_amt, term_months, interest_rate)
        total_loan_p = loan_amt * 1000.0
        total_repayment = raw_emi * term_months
        total_interest = max(0.0, total_repayment - total_loan_p)

        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Loan Principal", f"₹{total_loan_p:,.0f}")
        ec2.metric("Monthly EMI", f"₹{raw_emi:,.2f}")
        ec3.metric("Total Interest", f"₹{total_interest:,.2f}")
        ec4.metric("Total Repayment", f"₹{total_repayment:,.2f}")

        # Store input state in Streamlit session_state
        raw_input = {
            "Gender": gender, "Married": married, "Dependents": dependents,
            "Education": education, "Self_Employed": self_emp,
            "ApplicantIncome": app_income, "CoapplicantIncome": co_income,
            "LoanAmount": loan_amt, "Loan_Amount_Term": term_months,
            "Credit_History": credit_hist, "Property_Area": property_area
        }
        st.session_state["raw_input"] = raw_input
        st.session_state["interest_rate"] = interest_rate

        st.success("✅ Applicant details submitted! Switch to **Page 2** to view the Prediction & Risk Assessment.")

    # -------------------------------------------------------------------------
    # PAGE 2: LOAN PREDICTION RESULT & RISK ASSESSMENT
    # -------------------------------------------------------------------------
    elif page.startswith("📊"):
        st.header("📊 Loan Prediction & Calibrated Credit Risk Score")

        raw_input = st.session_state.get("raw_input", None)
        interest_rate = st.session_state.get("interest_rate", DEFAULT_INTEREST_RATE)

        if raw_input is None:
            st.info("ℹ️ Please enter applicant details on **Page 1** first.")
            return

        engineered = transform_input(raw_input, annual_interest_rate=interest_rate)
        row = {f: engineered.get(f, np.nan) for f in feature_names}
        X_single = pd.DataFrame([row], columns=feature_names)
        for col in CATEGORICAL_FEATURES:
            if col in X_single.columns:
                X_single[col] = X_single[col].astype(str)

        # Risk System Evaluation
        risk_sys = RiskAssessmentSystem()
        risk_res = risk_sys.evaluate_applicant_risk(model, X_single)
        pred = int(model.predict(X_single)[0])

        st.session_state["X_single"] = X_single
        st.session_state["pred"] = pred
        st.session_state["risk_res"] = risk_res
        st.session_state["engineered"] = engineered

        # Status Banner
        if pred == 1:
            st.success(f"🎉 **LOAN APPROVED** — High likelihood of successful repayment.")
        else:
            st.error(f"❌ **LOAN REJECTED** — Applicant does not meet threshold criteria.")

        # Metric Cards
        m1, m2, m3 = st.columns(3)
        m1.metric("Approval Probability", f"{risk_res['approval_probability'] * 100:.2f}%")
        m2.metric("Credit Risk Score (0-100)", f"{risk_res['risk_score']} / 100")
        m3.metric("Risk Category", f"{risk_res['risk_category']}")

        st.markdown("### 🏦 Banking Ratio Analysis")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Estimated Monthly EMI", f"₹{np.expm1(engineered.get('EMI', 0)):,.2f}")
        b2.metric("Debt-to-Income (DTI)", f"{engineered.get('DTI', 0) * 100:.2f}%")
        b3.metric("Loan-to-Annual-Income", f"{engineered.get('Loan_To_Income', 0):.2f}x")
        b4.metric("Balance Disposable Income", f"₹{engineered.get('Balance_Income', 0):,.2f}")

    # -------------------------------------------------------------------------
    # PAGE 3: EXPLAINABLE AI (SHAP & COUNTERFACTUALS)
    # -------------------------------------------------------------------------
    elif page.startswith("🔍"):
        st.header("🔍 Explainable AI (SHAP & Counterfactual Analysis)")

        X_single = st.session_state.get("X_single", None)
        if X_single is None:
            st.info("ℹ️ Please run a prediction on **Page 1 & 2** first.")
            return

        df_shap = compute_individual_shap(model, X_single)
        reasons = generate_natural_language_reasons(df_shap)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🟢 Approval Drivers")
            for r in reasons["positive"]:
                st.markdown(f"**{r}**")

        with col2:
            st.subheader("🔴 Risk Factors")
            for r in reasons["negative"]:
                st.markdown(f"**{r}**")

        st.markdown("---")
        st.subheader("🌊 SHAP Waterfall Plot")
        waterfall_file = save_shap_waterfall_plot(model, X_single, "shap_waterfall.png")
        if waterfall_file and os.path.exists(waterfall_file):
            st.image(waterfall_file, caption="SHAP Waterfall Plot for Applicant", use_container_width=True)

        st.markdown("---")
        st.subheader("💡 Counterfactual Recommendations (DiCE ML)")
        cf_res = generate_counterfactual_explanation(model, X_train, X_single)
        for rec in cf_res.get("recommendations", []):
            st.info(f"👉 {rec}")

    # -------------------------------------------------------------------------
    # PAGE 4: FAIRNESS & BIAS MITIGATION DASHBOARD
    # -------------------------------------------------------------------------
    elif page.startswith("⚖️"):
        st.header("⚖️ Responsible AI Fairness Auditing & Bias Mitigation Dashboard")

        fairness_audit = bundle.get("fairness_audit", {})
        mit_report = bundle.get("bias_mitigation_report", {})

        st.subheader("1. Fairlearn Multi-Attribute Fairness Audit")
        if fairness_audit:
            audit_summary = []
            for attr, data in fairness_audit.items():
                audit_summary.append({
                    "Sensitive Attribute": attr,
                    "Demographic Parity Diff": f"{data.get('dpd', 0):+.4f}",
                    "Equal Opportunity Diff": f"{data.get('equal_opportunity_diff', 0):.4f}",
                    "Equalized Odds Diff": f"{data.get('equalized_odds_diff', 0):.4f}",
                })
            st.table(pd.DataFrame(audit_summary))

        st.markdown("---")
        st.subheader("2. Bias Mitigation Performance (Fairlearn ThresholdOptimizer)")
        if mit_report:
            df_mit = pd.DataFrame(mit_report)
            st.table(df_mit)

            # Chart Comparison
            fig, ax = plt.subplots(figsize=(8, 4))
            metrics = df_mit["Metric"]
            before = df_mit["Before Mitigation"]
            after = df_mit["After Mitigation"]

            x = np.arange(len(metrics))
            width = 0.35

            ax.bar(x - width/2, before, width, label="Before Mitigation", color="#EF4444")
            ax.bar(x + width/2, after, width, label="After Mitigation", color="#10B981")

            ax.set_ylabel("Metric Score")
            ax.set_title("Model Performance & Fairness Metrics Before vs After Mitigation")
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=20, ha="right")
            ax.legend()
            st.pyplot(fig)


if __name__ == "__main__":
    main()
