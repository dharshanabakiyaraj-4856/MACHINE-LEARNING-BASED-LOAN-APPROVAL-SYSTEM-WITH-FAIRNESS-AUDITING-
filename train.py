"""
train.py
Bank Loan Approval Prediction - Training & Validation Pipeline

Includes:
- 5-Fold Stratified Cross Validation
- Optuna-based Hyperparameter Tuning (Optional via config.yaml)
- Advanced Fairlearn Fairness Auditing & Bias Mitigation
- Global SHAP Feature Interpretability
- Probability Calibration & Final Model Bundle Export
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from catboost import CatBoostClassifier

from utils.logger import get_logger
from preprocessing import load_raw_data, preprocess, CATEGORICAL_FEATURES
from fairness import audit_fairness, mitigate_bias
from explainability import save_shap_waterfall_plot
from risk import RiskAssessmentSystem

warnings.filterwarnings("ignore")

logger = get_logger("TrainPipeline")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load settings from config.yaml safely without strict external dependencies."""
    if not os.path.exists(config_path):
        return {}
    try:
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        # Fallback default configuration if pyyaml is missing
        return {
            "data": {"train_path": os.path.join("dataset", "train.csv"), "model_path": "model.pkl"},
            "banking": {"default_annual_interest_rate": 9.5},
            "model": {"random_seed": 42, "catboost": {"iterations": 600, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3}},
            "validation": {"n_splits": 5},
            "optuna": {"enabled": False},
        }


def run_optuna_tuning(X: pd.DataFrame, y: pd.Series, cat_indices: list, n_trials: int = 15, seed: int = 42) -> dict:
    """Run Optuna hyperparameter optimization using 5-fold Stratified CV ROC-AUC as objective."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        logger.info(f"Starting Optuna hyperparameter search ({n_trials} trials)...")

        def objective(trial):
            params = {
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "iterations": trial.suggest_int("iterations", 300, 800),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "eval_metric": "AUC",
                "random_seed": seed,
                "verbose": 0,
            }

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            auc_scores = []

            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                m = CatBoostClassifier(**params, cat_features=cat_indices)
                m.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0, early_stopping_rounds=40)

                probs = m.predict_proba(X_val)[:, 1]
                auc_scores.append(roc_auc_score(y_val, probs))

            return np.mean(auc_scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        logger.info(f"Optuna Optimization Complete! Best Trial ROC-AUC: {study.best_value:.4f}")
        logger.info(f"Best Parameters: {study.best_params}")
        return study.best_params

    except Exception as e:
        logger.warning(f"Optuna tuning skipped: {e}. Using configured default parameters.")
        return {}


def run_training() -> dict:
    cfg = load_config()

    data_cfg   = cfg.get("data", {})
    model_cfg  = cfg.get("model", {})
    opt_cfg    = cfg.get("optuna", {})
    val_cfg    = cfg.get("validation", {})

    data_path  = data_cfg.get("train_path", os.path.join("dataset", "train.csv"))
    model_path = data_cfg.get("model_path", "model.pkl")
    shap_path  = data_cfg.get("shap_summary_path", "shap_summary.png")
    seed       = model_cfg.get("random_seed", 42)
    n_splits   = val_cfg.get("n_splits", 5)

    _banner("BANK LOAN APPROVAL PREDICTION  -  Industry Responsible AI Training Pipeline")

    # Step 1: Load Data
    _section("STEP 1 : Loading Raw Dataset")
    df_raw = load_raw_data(data_path)

    # Step 2: Split Train / Test to prevent data leakage during preprocessing
    _section("STEP 2 : Stratified Train/Test Split & Preprocessing")
    train_raw, test_raw = train_test_split(
        df_raw, test_size=0.2, random_state=seed, stratify=df_raw["Loan_Status"]
    )

    logger.info("Preprocessing Training Set (learning imputation statistics)...")
    train_df, imp_stats = preprocess(train_raw, verbose=False)

    logger.info("Preprocessing Test Set (using learned training statistics)...")
    test_df, _ = preprocess(test_raw, imputation_stats=imp_stats, verbose=False)

    X_train = train_df.drop(columns=["Loan_Status"])
    y_train = train_df["Loan_Status"]
    X_test  = test_df.drop(columns=["Loan_Status"])
    y_test  = test_df["Loan_Status"]

    # Preserve sensitive attributes for auditing & bias mitigation before dropping
    sens_train = train_df[["Gender", "Married", "Education"]].copy()
    sens_test  = test_df[["Gender", "Married", "Education"]].copy()

    # Drop direct sensitive features from predictive model features (prevent disparate treatment)
    features_to_drop = [c for c in ["Gender", "Married"] if c in X_train.columns]
    X_train = X_train.drop(columns=features_to_drop)
    X_test  = X_test.drop(columns=features_to_drop)

    cat_features_model = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    cat_indices = [i for i, c in enumerate(X_train.columns) if c in cat_features_model]
    cat_names   = [X_train.columns[i] for i in cat_indices]

    # Step 3: Hyperparameter Optimization (Optional)
    _section("STEP 3 : Hyperparameter Optimization")
    cb_params = model_cfg.get("catboost", {
        "iterations": 600, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3
    })

    if opt_cfg.get("enabled", False):
        n_trials = opt_cfg.get("n_trials", 15)
        best_params = run_optuna_tuning(X_train, y_train, cat_indices, n_trials=n_trials, seed=seed)
        if best_params:
            cb_params.update(best_params)

    # Step 4: 5-Fold Stratified Cross Validation Evaluation
    _section(f"STEP 4 : {n_splits}-Fold Stratified Cross-Validation")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    cv_acc, cv_auc, cv_f1, cv_prec, cv_rec = [], [], [], [], []

    for fold, (t_idx, v_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr_f, X_val_f = X_train.iloc[t_idx], X_train.iloc[v_idx]
        y_tr_f, y_val_f = y_train.iloc[t_idx], y_train.iloc[v_idx]

        fold_model = CatBoostClassifier(
            **cb_params,
            eval_metric="AUC",
            random_seed=seed,
            verbose=0,
            cat_features=cat_indices
        )
        fold_model.fit(X_tr_f, y_tr_f, eval_set=(X_val_f, y_val_f), verbose=0, early_stopping_rounds=40)

        preds = fold_model.predict(X_val_f)
        probs = fold_model.predict_proba(X_val_f)[:, 1]

        cv_acc.append(accuracy_score(y_val_f, preds))
        cv_auc.append(roc_auc_score(y_val_f, probs))
        cv_f1.append(f1_score(y_val_f, preds, zero_division=0))
        cv_prec.append(precision_score(y_val_f, preds, zero_division=0))
        cv_rec.append(recall_score(y_val_f, preds, zero_division=0))

    cv_metrics = {
        "mean_accuracy": float(np.mean(cv_acc)), "std_accuracy": float(np.std(cv_acc)),
        "mean_roc_auc": float(np.mean(cv_auc)),  "std_roc_auc": float(np.std(cv_auc)),
        "mean_f1": float(np.mean(cv_f1)),        "std_f1": float(np.std(cv_f1)),
        "mean_precision": float(np.mean(cv_prec)), "std_precision": float(np.std(cv_prec)),
        "mean_recall": float(np.mean(cv_rec)),   "std_recall": float(np.std(cv_rec)),
    }

    print("\n" + "-"*65)
    print(f"  {n_splits}-FOLD CROSS-VALIDATION RESULTS")
    print("-"*65)
    print(f"  Accuracy  : {cv_metrics['mean_accuracy']:.4f} ± {cv_metrics['std_accuracy']:.4f}")
    print(f"  ROC-AUC   : {cv_metrics['mean_roc_auc']:.4f} ± {cv_metrics['std_roc_auc']:.4f}")
    print(f"  F1-Score  : {cv_metrics['mean_f1']:.4f} ± {cv_metrics['std_f1']:.4f}")
    print(f"  Precision : {cv_metrics['mean_precision']:.4f} ± {cv_metrics['std_precision']:.4f}")
    print(f"  Recall    : {cv_metrics['mean_recall']:.4f} ± {cv_metrics['std_recall']:.4f}")

    # Step 5: Final Model Training & Calibration
    _section("STEP 5 : Final Model Training on Full Training Dataset")
    final_model = CatBoostClassifier(
        **cb_params,
        eval_metric="AUC",
        random_seed=seed,
        verbose=100,
        cat_features=cat_indices,
        early_stopping_rounds=50,
    )
    final_model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

    risk_sys = RiskAssessmentSystem()
    calibrated_model = risk_sys.calibrate_model(final_model, X_test, y_test)

    # Step 6: Advanced Fairness Auditing & Bias Mitigation
    _section("STEP 6 : Advanced Fairlearn Auditing & Bias Mitigation")
    fairness_results = audit_fairness(final_model, X_test, y_test, sens_test)

    # Mitigate bias using ThresholdOptimizer on Gender
    sens_tr_gender = sens_train["Gender"]
    sens_te_gender = sens_test["Gender"]
    mitigated_model, comparison_dict, df_comparison = mitigate_bias(
        final_model, X_train, y_train, X_test, y_test, sens_tr_gender, sens_te_gender
    )

    print("\n  Fairness Audit Results (Gender):")
    g_res = fairness_results.get("Gender", {})
    print(f"    Demographic Parity Difference : {g_res.get('dpd', 0):+.4f}")
    print(f"    Equal Opportunity Difference  : {g_res.get('equal_opportunity_diff', 0):.4f}")
    print(f"    Equalized Odds Difference     : {g_res.get('equalized_odds_diff', 0):.4f}")

    print("\n  Bias Mitigation Comparison (Before vs After):")
    print(df_comparison.to_string(index=False))

    # Save SHAP Summary Plot
    try:
        import shap
        explainer = shap.TreeExplainer(final_model)
        sv = explainer.shap_values(X_test)
        vals = sv[1] if isinstance(sv, list) else sv
        plt.figure(figsize=(9, 5))
        shap.summary_plot(vals, X_test, show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(shap_path, dpi=120)
        plt.close()
        logger.info(f"Saved SHAP summary plot to '{shap_path}'")
    except Exception as e:
        logger.warning(f"Could not save SHAP summary plot: {e}")

    # Export complete bundle
    bundle = {
        "model"                 : final_model,
        "calibrated_model"      : calibrated_model,
        "mitigated_model"       : mitigated_model,
        "feature_names"         : list(X_train.columns),
        "cat_feature_indices"   : cat_indices,
        "cat_feature_names"     : cat_names,
        "imputation_stats"      : imp_stats,
        "cv_metrics"            : cv_metrics,
        "fairness_audit"        : fairness_results,
        "bias_mitigation_report": comparison_dict,
        "config"                : cfg,
        "X_train_sample"        : X_train.head(100),
    }

    joblib.dump(bundle, model_path)
    logger.info(f"Complete Model Bundle saved to '{model_path}'")

    return bundle


def _section(title: str):
    print("\n" + "="*65)
    print(f"  {title}")
    print("="*65)


def _banner(title: str):
    print("\n" + "#"*65)
    print(f"  {title}")
    print("#"*65)


if __name__ == "__main__":
    run_training()
