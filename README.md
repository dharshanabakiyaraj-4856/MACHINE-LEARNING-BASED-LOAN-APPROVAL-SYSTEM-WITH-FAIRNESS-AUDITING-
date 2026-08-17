# Industry-Level Responsible AI Bank Loan Approval Prediction System
### Powered by CatBoost, SHAP Waterfall Explanations, Fairlearn Bias Mitigation, DiCE Counterfactuals, and Streamlit Dashboard

---

## 🌟 Overview

An **industry-grade Responsible AI Banking System** built upon the Kaggle Loan Prediction dataset. Upgraded from a baseline classifier into a production-ready, transparent, and fair credit risk assessment engine adhering to global Responsible AI governance standards.

Key capabilities include:
- **CatBoost Classifier** — Native categorical handling with 5-fold Stratified Cross-Validation & optional Optuna tuning.
- **Advanced Banking Feature Engineering** — Debt-to-Income (DTI), Loan-to-Income ratios, EMI estimation, and Balance Disposable Income.
- **Credit Risk Scoring (0–100)** — Calibrated approval probabilities categorized into High Risk (0–40), Medium Risk (41–70), and Low Risk (71–100).
- **Explainable AI (SHAP & DiCE)** — Local SHAP waterfall plots, natural language positive/negative drivers, and counterfactual recommendations.
- **Fairness Auditing & Mitigation (Fairlearn)** — Multi-attribute fairness auditing (Gender, Married, Education, Intersectional groups) and automated post-processing bias mitigation (`ThresholdOptimizer`).
- **Interactive Streamlit Web Dashboard & CLI** — 4-page Streamlit application (`app.py`) plus backward-compatible terminal console app (`main.py`).

---

## 📂 Project Structure

```
intern_finall_project/
├── dataset/
│   └── train.csv                   # Kaggle Loan Prediction dataset
├── utils/
│   ├── __init__.py
│   └── logger.py                   # Centralized structured logging utility
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py       # Unit tests for preprocessing & banking features
├── preprocessing.py                # Preprocessing, banking ratio formulas, log1p & transform_input
├── risk.py                         # Probability calibration, 0-100 credit risk score & categories
├── fairness.py                     # Fairlearn multi-attribute auditing & ThresholdOptimizer bias mitigation
├── explainability.py               # SHAP waterfall generation, natural language reasons, DiCE ML counterfactuals
├── train.py                        # 5-Fold Stratified CV, Optuna tuning toggle, final model fit & bundle export
├── main.py                         # Interactive terminal console application (Backward Compatible)
├── app.py                          # Multi-page Streamlit web dashboard application
├── config.yaml                     # Centralized project configuration file
├── model.pkl                       # Trained model & evaluation bundle artifact
├── shap_summary.png                # Global SHAP feature importance plot
├── shap_waterfall.png              # Individual applicant SHAP waterfall plot
├── requirements.txt                # Python dependencies
└── README.md                       # Complete technical documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.9+
- pip

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1: Run Unit Tests
```bash
python -m unittest tests/test_preprocessing.py
```

### Step 2: Train & Audit Model
```bash
python train.py
```
This script will:
1. Load `dataset/train.csv`.
2. Preprocess data and compute banking features (EMI, DTI, Loan-to-Income, Balance Income).
3. Execute **5-Fold Stratified Cross-Validation** (reporting Mean & Std for Accuracy, ROC-AUC, F1-Score, Precision, Recall).
4. Run optional **Optuna hyperparameter tuning** if enabled in `config.yaml`.
5. Fit final CatBoost model on training data and calibrate probabilities.
6. Audit multi-attribute fairness using **Fairlearn** across `Gender`, `Married`, `Education`, and `Gender_Married`.
7. Apply **Fairlearn ThresholdOptimizer** for bias mitigation and output a Before vs. After comparison report.
8. Export the full model bundle to `model.pkl`.

### Step 3: Launch Streamlit Web Dashboard
```bash
streamlit run app.py
```
Dashboard contains 4 pages:
- **Page 1: Applicant Form & Real Banking EMI Calculator**
- **Page 2: Loan Prediction Result & 0–100 Credit Risk Score**
- **Page 3: Explainable AI (SHAP Waterfall Plots + DiCE Counterfactuals)**
- **Page 4: Responsible AI Fairness Auditing & Bias Mitigation Dashboard**

### Step 4: Run Terminal Console UI (Backward Compatible)
```bash
python main.py
```

---

## 📐 Banking Feature Formulas

1. **Equated Monthly Installment (EMI)**:
   $$EMI = P \times r \times \frac{(1+r)^n}{(1+r)^n - 1}$$
   *where $P = \text{LoanAmount} \times 1000$, $r = \text{Annual Interest Rate} / 1200$, $n = \text{Loan\_Amount\_Term}$.*
2. **Debt-to-Income Ratio (DTI)**:
   $$DTI = \frac{EMI}{\text{TotalIncome}}$$
3. **Loan-to-Annual-Income Ratio**:
   $$\text{Loan\_To\_Income} = \frac{\text{LoanAmount} \times 1000}{\text{TotalIncome} \times 12}$$
4. **Disposable Balance Income**:
   $$\text{Balance\_Income} = \text{TotalIncome} - EMI$$

---

## 📊 Credit Risk Scoring Buckets

- **0 – 40**: 🔴 **High Risk** (Rejection Recommended)
- **41 – 70**: 🟡 **Medium Risk** (Requires Manual Underwriting / Counterfactual Adjustment)
- **71 – 100**: 🟢 **Low Risk** (Approved for Automated Disbursement)

---

## ⚖️ Fairness Auditing & Bias Mitigation

The system evaluates:
- **Demographic Parity Difference** ($DPD \le 0.10$ Target)
- **Equal Opportunity Difference** ($EOD$ Target $0.00$)
- **Equalized Odds Difference** ($EQD$ Target $0.00$)

Post-processing bias mitigation is applied using **Fairlearn's `ThresholdOptimizer`**, comparing predictive accuracy and fairness metrics before and after mitigation.

---

## 🛠️ Configuration (`config.yaml`)

Edit `config.yaml` to adjust hyperparameters, Optuna trial count, interest rates, or dataset paths:
```yaml
optuna:
  enabled: false  # Set to true to enable automated hyperparameter search
  n_trials: 20

banking:
  default_annual_interest_rate: 9.5
```
