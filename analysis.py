"""
Machine learning for maternal health risk stratification using IoT-collected
clinical datasets from Bangladesh: a two-cohort comparative analysis

Authors: Md Rafin Rahman, Jafren Iqbal Rose, Md Rofiqur Rahman
Institution: Institute for Developing Science and Health Initiatives (ideSHi), Dhaka, Bangladesh
Corresponding author: mrahman@ideshi.org | ORCID: 0009-0007-5366-0175

Datasets:
  Cohort 1 — UCI Maternal Health Risk Dataset (DOI: 10.24432/C5DP5D)
  Cohort 2 — Mendeley Maternal Health Risk Assessment Dataset (DOI: 10.17632/p5w98dvbbk.1)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, f1_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42
DPI = 300

# ─────────────────────────────────────────────
# COHORT 1: UCI Dataset (3-class, 6 features)
# ─────────────────────────────────────────────

def run_cohort1(filepath: str, output_dir: str = "figures_cohort1"):
    import os; os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df.loc[df['HeartRate'] == 7, 'HeartRate'] = 70  # Implausible outlier correction

    label_map = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
    df['RiskEncoded'] = df['RiskLevel'].map(label_map)
    X = df.drop(['RiskLevel', 'RiskEncoded'], axis=1)
    y = df['RiskEncoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=200, random_state=RANDOM_SEED, eval_metric='mlogloss', verbosity=0),
        'LightGBM': lgb.LGBMClassifier(n_estimators=200, random_state=RANDOM_SEED, verbose=-1),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_SEED),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        'SVM': SVC(probability=True, random_state=RANDOM_SEED),
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    results = {}

    for name, model in models.items():
        cv_res = cross_validate(model, X, y, cv=cv,
                                scoring=['accuracy', 'f1_macro', 'roc_auc_ovr_weighted'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        auc_score = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='weighted')
        results[name] = {
            'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
            'accuracy': acc, 'f1_macro': f1, 'auc': auc_score,
            'cv_acc_mean': cv_res['test_accuracy'].mean(),
            'cv_acc_std': cv_res['test_accuracy'].std(),
            'cv_auc_mean': cv_res['test_roc_auc_ovr_weighted'].mean(),
            'cv_auc_std': cv_res['test_roc_auc_ovr_weighted'].std(),
        }
        print(f"[Cohort 1] {name}: Acc={acc:.4f} F1={f1:.4f} AUC={auc_score:.4f}")

    # SHAP for XGBoost
    explainer = shap.TreeExplainer(results['XGBoost']['model'])
    shap_values = explainer.shap_values(X_test)

    return results, X, y, X_test, y_test, shap_values


# ─────────────────────────────────────────────
# COHORT 2: Mendeley Dataset (binary, 11 features)
# ─────────────────────────────────────────────

def run_cohort2(filepath: str, output_dir: str = "figures_cohort2"):
    import os; os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(filepath)
    df = df[df['Risk Level'].notna()].copy()
    df = df[df['Age'] < 100].copy()
    df.loc[df['BMI'] == 0, 'BMI'] = np.nan
    df['RiskEncoded'] = (df['Risk Level'] == 'High').astype(int)

    feat_cols = ['Age', 'Systolic BP', 'Diastolic', 'BS', 'Body Temp', 'BMI',
                 'Previous Complications', 'Preexisting Diabetes',
                 'Gestational Diabetes', 'Mental Health', 'Heart Rate']

    X = df[feat_cols].copy()
    y = df['RiskEncoded'].values

    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feat_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=200, random_state=RANDOM_SEED, eval_metric='logloss', verbosity=0),
        'LightGBM': lgb.LGBMClassifier(n_estimators=200, random_state=RANDOM_SEED, verbose=-1),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_SEED),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        'SVM': SVC(probability=True, random_state=RANDOM_SEED),
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    results = {}

    for name, model in models.items():
        cv_res = cross_validate(model, X_imp, y, cv=cv,
                                scoring=['accuracy', 'f1', 'roc_auc'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_prob)
        results[name] = {
            'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
            'accuracy': acc, 'f1': f1, 'auc': auc_score,
            'cv_acc_mean': cv_res['test_accuracy'].mean(),
            'cv_acc_std': cv_res['test_accuracy'].std(),
            'cv_auc_mean': cv_res['test_roc_auc'].mean(),
            'cv_auc_std': cv_res['test_roc_auc'].std(),
        }
        print(f"[Cohort 2] {name}: Acc={acc:.4f} F1={f1:.4f} AUC={auc_score:.4f}")

    explainer = shap.TreeExplainer(results['XGBoost']['model'])
    shap_values = explainer.shap_values(X_test)

    return results, X_imp, y, X_test, y_test, shap_values, feat_cols


if __name__ == "__main__":
    print("=== COHORT 1: UCI ===")
    c1_results, *_ = run_cohort1("data/maternal_health_risk_uci.csv")

    print("\n=== COHORT 2: MENDELEY ===")
    c2_results, *_ = run_cohort2("data/maternal_health_risk_mendeley.csv")

    print("\nAnalysis complete.")
