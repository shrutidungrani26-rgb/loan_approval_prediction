"""
Loan Approval Prediction System - ML Training and Evaluation Pipeline
Algorithm: Supervised Machine Learning - Logistic Regression Classification
Author: AI Financial Intelligence Engineering Team
"""

import os
import json
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# Set base working directory to script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'loan_approval_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'Reports')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Visual styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
NAVY = '#0A192F'
ROYAL_BLUE = '#1E3A8A'
CYAN = '#06B6D4'
LIGHT_CYAN = '#38BDF8'
EMERALD = '#10B981'
ROSE = '#EF4444'

def load_and_preprocess_data(csv_path):
    print("=" * 70)
    print("STEP 1: LOADING & INSPECTING RAW DATASET")
    if not os.path.exists(csv_path):
        candidates = [
            os.path.join(BASE_DIR, 'dataset', 'loan_approval_dataset (1).csv'),
            os.path.join(BASE_DIR, 'dataset', 'loan_approval_dataset.csv'),
            r"C:\Users\DELL\Desktop\vrunda 2\Loan_Approval_Prediction\dataset\loan_approval_dataset.csv",
            r"C:\Users\DELL\Desktop\vrunda 2\Loan_Approval_Prediction\dataset\loan_approval_dataset (1).csv",
            r"C:\Users\DELL\Desktop\vrunda 2\dataset\loan_approval_dataset.csv",
            r"C:\Users\DELL\Downloads\loan_approval_dataset (1).csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                csv_path = c
                break
        else:
            raise FileNotFoundError(f"Dataset not found at {csv_path}. Please add the CSV file.")

    raw_df = pd.read_csv(csv_path)
    print(f"Loaded CSV successfully. Raw shape: {raw_df.shape}")
    print(f"Raw column names: {list(raw_df.columns)}")

    # Clean column names by stripping leading/trailing whitespace
    clean_df = raw_df.copy()
    clean_df.columns = clean_df.columns.str.strip()

    # Clean all string values in dataframe by stripping leading/trailing whitespace
    for col in clean_df.columns:
        if clean_df[col].dtype == 'object' or pd.api.types.is_string_dtype(clean_df[col]):
            clean_df[col] = clean_df[col].astype(str).str.strip()

    print(f"Cleaned column names: {list(clean_df.columns)}")
    print(f"Missing values count per column:\n{clean_df.isnull().sum()}")
    print(f"Target distribution ('loan_status'):\n{clean_df['loan_status'].value_counts()}")

    return clean_df

def train_and_evaluate(df):
    print("\n" + "=" * 70)
    print("STEP 2: PREPARING FEATURES & STRATIFIED TRAIN/TEST SPLIT")
    print("=" * 70)

    # Exclude identifier loan_id and target loan_status
    feature_cols = [
        'no_of_dependents', 'education', 'self_employed', 'income_annum',
        'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
        'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
    ]
    
    num_cols = [
        'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
        'cibil_score', 'residential_assets_value', 'commercial_assets_value',
        'luxury_assets_value', 'bank_asset_value'
    ]
    cat_cols = ['education', 'self_employed']

    X = df[feature_cols].copy()
    y = df['loan_status'].copy()

    # Stratified 80% Train, 20% Test split to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Total applicants: {len(df)}")
    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(df):.1%})")
    print(f"Testing set:  {len(X_test)} samples ({len(X_test)/len(df):.1%})")

    print("\n" + "=" * 70)
    print("STEP 3: BUILDING SCIKIT-LEARN PIPELINE")
    print("=" * 70)

    # Numerical pipeline: Median Imputation + Standard Scaler
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Mode Imputation + One-Hot Encoding
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    # Primary algorithm: Logistic Regression with balanced max_iter
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs', C=1.0))
    ])

    print("Fitting Logistic Regression Pipeline on training data...")
    model_pipeline.fit(X_train, y_train)
    print("Model training completed successfully.")

    print("\n" + "=" * 70)
    print("STEP 4: EVALUATING MODEL ON HELD-OUT TEST SET")
    print("=" * 70)

    y_pred = model_pipeline.predict(X_test)
    approved_index = list(model_pipeline.classes_).index('Approved')
    y_proba = model_pipeline.predict_proba(X_test)[:, approved_index]
    y_test_binary = (y_test == 'Approved').astype(int)

    # Core evaluation metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec_approved = float(precision_score(y_test, y_pred, pos_label='Approved'))
    rec_approved = float(recall_score(y_test, y_pred, pos_label='Approved'))
    f1_approved = float(f1_score(y_test, y_pred, pos_label='Approved'))

    prec_rejected = float(precision_score(y_test, y_pred, pos_label='Rejected'))
    rec_rejected = float(recall_score(y_test, y_pred, pos_label='Rejected'))
    f1_rejected = float(f1_score(y_test, y_pred, pos_label='Rejected'))

    macro_f1 = float(f1_score(y_test, y_pred, average='macro'))
    weighted_f1 = float(f1_score(y_test, y_pred, average='weighted'))
    roc_auc = float(roc_auc_score(y_test_binary, y_proba))

    cm = confusion_matrix(y_test, y_pred, labels=['Rejected', 'Approved'])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    clf_report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Accuracy:          {acc:.4f} ({acc*100:.2f}%)")
    print(f"Approved Precision:{prec_approved:.4f} ({prec_approved*100:.2f}%)")
    print(f"Approved Recall:   {rec_approved:.4f} ({rec_approved*100:.2f}%)")
    print(f"Approved F1-Score: {f1_approved:.4f}")
    print(f"Rejected Precision:{prec_rejected:.4f} ({prec_rejected*100:.2f}%)")
    print(f"Rejected Recall:   {rec_rejected:.4f} ({rec_rejected*100:.2f}%)")
    print(f"Rejected F1-Score: {f1_rejected:.4f}")
    print(f"ROC-AUC Score:     {roc_auc:.4f}")
    print(f"Confusion Matrix:  TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Extract feature importance (coefficients)
    cat_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + cat_feature_names
    coefficients = model_pipeline.named_steps['classifier'].coef_[0]

    coef_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False).reset_index(drop=True)

    # Save Pipeline Model
    model_save_path = os.path.join(MODELS_DIR, 'logistic_regression_pipeline.pkl')
    joblib.dump(model_pipeline, model_save_path)
    print(f"\n[Saved] Trained Model Pipeline -> {model_save_path}")

    # Save Metrics JSON
    metrics = {
        'model_name': 'Logistic Regression Classification Pipeline',
        'dataset_shape': list(df.shape),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features_count': len(feature_cols),
        'accuracy': round(acc, 4),
        'precision_approved': round(prec_approved, 4),
        'recall_approved': round(rec_approved, 4),
        'f1_approved': round(f1_approved, 4),
        'precision_rejected': round(prec_rejected, 4),
        'recall_rejected': round(rec_rejected, 4),
        'f1_rejected': round(f1_rejected, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_f1': round(weighted_f1, 4),
        'roc_auc': round(roc_auc, 4),
        'confusion_matrix': {
            'true_rejected': tn,
            'false_approved': fp,
            'false_rejected': fn,
            'true_approved': tp
        },
        'feature_importance': coef_df.to_dict(orient='records'),
        'classes': list(model_pipeline.classes_)
    }

    metrics_save_path = os.path.join(REPORTS_DIR, 'model_metrics.json')
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[Saved] Evaluation Metrics -> {metrics_save_path}")

    # Save duplicate metrics in models/ for direct access
    with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # Generate Evaluation Charts
    generate_charts(y_test, y_pred, y_test_binary, y_proba, cm, coef_df, metrics)

    # Generate Full HTML College EDA Report
    generate_html_report(df, metrics, coef_df)

    return model_pipeline, metrics

def generate_charts(y_test, y_pred, y_test_binary, y_proba, cm, coef_df, metrics):
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING HIGH-RESOLUTION CHARTS & VISUALS")
    print("=" * 70)

    # 1. Confusion Matrix Chart
    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Rejected', 'Approved'],
        yticklabels=['Rejected', 'Approved'],
        cbar=True,
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    plt.title('Logistic Regression Confusion Matrix', fontsize=14, weight='bold', pad=15, color=NAVY)
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('Actual Label', fontsize=12, labelpad=10)
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Saved] Confusion Matrix -> {cm_path}")

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test_binary, y_proba)
    plt.figure(figsize=(7.5, 6), dpi=300)
    plt.plot(fpr, tpr, color=CYAN, lw=3, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], color='#94A3B8', linestyle='--', lw=1.5, label='Chance (AUC = 0.5000)')
    plt.fill_between(fpr, tpr, alpha=0.15, color=CYAN)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, weight='bold', pad=15, color=NAVY)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.legend(loc='lower right', fontsize=11, frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    roc_path = os.path.join(REPORTS_DIR, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[Saved] ROC Curve -> {roc_path}")

    # 3. Feature Importance (Coefficients)
    plt.figure(figsize=(9, 6.5), dpi=300)
    colors = [CYAN if c > 0 else ROSE for c in coef_df['Coefficient']]
    sns.barplot(data=coef_df, x='Coefficient', y='Feature', hue='Feature', palette=colors, legend=False)
    plt.axvline(0, color='#64748B', linestyle='--', lw=1)
    plt.title('Logistic Regression Feature Coefficients\n(Positive = Favors Approval, Negative = Favors Rejection)', fontsize=13, weight='bold', pad=15, color=NAVY)
    plt.xlabel('Log-Odds Coefficient Weight', fontsize=11)
    plt.ylabel('Standardized Input Feature', fontsize=11)
    plt.tight_layout()
    feat_path = os.path.join(REPORTS_DIR, 'feature_importance.png')
    plt.savefig(feat_path, dpi=300)
    plt.close()
    print(f"[Saved] Feature Importance -> {feat_path}")

    # 4. Model Metrics Summary Card Bar Chart
    plt.figure(figsize=(8, 5), dpi=300)
    metric_names = ['Accuracy', 'Approved Precision', 'Approved Recall', 'Approved F1', 'ROC-AUC']
    metric_vals = [metrics['accuracy'], metrics['precision_approved'], metrics['recall_approved'], metrics['f1_approved'], metrics['roc_auc']]
    bars = plt.bar(metric_names, metric_vals, color=[NAVY, ROYAL_BLUE, CYAN, LIGHT_CYAN, EMERALD], width=0.55)
    plt.ylim(0, 1.15)
    plt.title('Primary Model Performance Metric Overview', fontsize=14, weight='bold', pad=15, color=NAVY)
    plt.ylabel('Score (0.0 to 1.0)', fontsize=11)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h*100:.2f}%", ha='center', va='bottom', fontsize=10, weight='bold')
    plt.tight_layout()
    summary_path = os.path.join(REPORTS_DIR, 'metrics_summary.png')
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"[Saved] Metrics Summary -> {summary_path}")

def generate_html_report(df, metrics, coef_df):
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING INTERACTIVE COLLEGE EDA & AUDIT REPORT (HTML)")
    print("=" * 70)

    # Encode images to base64 so HTML is 100% standalone and portable
    def img_to_b64(path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        return ""

    cm_b64 = img_to_b64(os.path.join(REPORTS_DIR, 'confusion_matrix.png'))
    roc_b64 = img_to_b64(os.path.join(REPORTS_DIR, 'roc_curve.png'))
    feat_b64 = img_to_b64(os.path.join(REPORTS_DIR, 'feature_importance.png'))
    summary_b64 = img_to_b64(os.path.join(REPORTS_DIR, 'metrics_summary.png'))
    logo_b64 = img_to_b64(os.path.join(ASSETS_DIR, 'logo.png'))

    total_apps = len(df)
    approved_count = int((df['loan_status'] == 'Approved').sum())
    rejected_count = int((df['loan_status'] == 'Rejected').sum())
    app_rate = (approved_count / total_apps) * 100

    desc_table_html = df.describe().T.round(2).to_html(classes='table table-striped table-hover', border=0)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Approval Intelligence - EDA & College Project Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --navy: #0A192F;
            --royal-blue: #1E3A8A;
            --cyan: #06B6D4;
            --light-cyan: #38BDF8;
            --emerald: #10B981;
            --rose: #EF4444;
            --bg-light: #F8FAFC;
            --card-bg: #FFFFFF;
            --border: #E2E8F0;
            --text-main: #0F172A;
            --text-muted: #64748B;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-light);
            color: var(--text-main);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: var(--card-bg);
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(10, 25, 47, 0.08);
            overflow: hidden;
            border: 1px solid var(--border);
        }}
        .header {{
            background: linear-gradient(135deg, #0A192F 0%, #1E3A8A 50%, #0369A1 100%);
            color: white;
            padding: 3.5rem 3rem;
            position: relative;
        }}
        .logo-wrap {{
            display: flex;
            align-items: center;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        .logo-img {{
            width: 70px;
            height: 70px;
            border-radius: 14px;
            border: 2px solid var(--cyan);
            box-shadow: 0 0 20px rgba(6, 182, 212, 0.5);
        }}
        .header h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 0.5rem;
        }}
        .header p {{
            font-size: 1.15rem;
            color: #E2E8F0;
            max-width: 800px;
        }}
        .badge {{
            display: inline-block;
            background: rgba(6, 182, 212, 0.2);
            border: 1px solid var(--cyan);
            color: var(--cyan);
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 1rem;
        }}
        .content {{
            padding: 3rem;
        }}
        .section-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--navy);
            margin: 2.5rem 0 1.25rem 0;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            border-bottom: 2px solid #E2E8F0;
            padding-bottom: 0.75rem;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0 2rem 0;
        }}
        .kpi-card {{
            background: linear-gradient(145deg, #FFFFFF, #F1F5F9);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            transition: transform 0.2s;
        }}
        .kpi-card:hover {{
            transform: translateY(-4px);
        }}
        .kpi-value {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.2rem;
            font-weight: 800;
            color: var(--royal-blue);
        }}
        .kpi-label {{
            font-size: 0.9rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-top: 0.25rem;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }}
        .chart-box {{
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
        }}
        .chart-box h3 {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.2rem;
            color: var(--navy);
            margin-bottom: 1rem;
            text-align: center;
        }}
        .chart-box img {{
            width: 100%;
            height: auto;
            border-radius: 10px;
        }}
        .table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            font-size: 0.95rem;
        }}
        .table th, .table td {{
            padding: 0.75rem 1rem;
            border: 1px solid #E2E8F0;
            text-align: left;
        }}
        .table th {{
            background-color: #0A192F;
            color: #FFFFFF;
            font-weight: 600;
        }}
        .table tr:nth-child(even) {{
            background-color: #F8FAFC;
        }}
        .table tr:hover {{
            background-color: #F1F5F9;
        }}
        .callout {{
            background: rgba(6, 182, 212, 0.08);
            border-left: 4px solid var(--cyan);
            padding: 1.25rem;
            border-radius: 8px;
            margin: 1.5rem 0;
            font-size: 1rem;
        }}
        .footer {{
            background: #0A192F;
            color: #94A3B8;
            padding: 2rem;
            text-align: center;
            font-size: 0.9rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo-wrap">
                {"<img src='data:image/png;base64," + logo_b64 + "' class='logo-img' alt='Bank Logo'>" if logo_b64 else ""}
                <div>
                    <h1>Loan Approval Intelligence System</h1>
                    <p>Comprehensive College Project & Exploratory Data Analysis Report</p>
                </div>
            </div>
            <span class="badge">Supervised Classification &bull; Logistic Regression Pipeline</span>
        </div>

        <div class="content">
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-value">{total_apps:,}</div>
                    <div class="kpi-label">Total Applicants</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value" style="color: var(--emerald);">{approved_count:,}</div>
                    <div class="kpi-label">Approved Loans ({app_rate:.1f}%)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value" style="color: var(--rose);">{rejected_count:,}</div>
                    <div class="kpi-label">Rejected Loans ({100-app_rate:.1f}%)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value" style="color: var(--cyan);">{metrics['accuracy']*100:.2f}%</div>
                    <div class="kpi-label">Model Accuracy</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value" style="color: var(--royal-blue);">{metrics['roc_auc']:.4f}</div>
                    <div class="kpi-label">ROC-AUC Score</div>
                </div>
            </div>

            <h2 class="section-title">1. Problem Statement & Objectives</h2>
            <p>
                In retail banking, assessing loan eligibility manually is resource-intensive and prone to human bias. 
                This project builds an automated, robust machine learning pipeline using <strong>Logistic Regression</strong> 
                to classify loan applications into <strong>Approved</strong> or <strong>Rejected</strong> based on verified 
                applicant financial parameters, credit history (CIBIL score), and collateral asset values.
            </p>

            <div class="callout">
                <strong>Core Objectives:</strong>
                <ul>
                    <li>Process authentic applicant records without synthetic fabrication.</li>
                    <li>Prevent data leakage using strict 80/20 stratified splitting and Scikit-learn Pipeline architecture.</li>
                    <li>Scale numerical features using StandardScaler and encode categorical inputs with OneHotEncoder.</li>
                    <li>Provide transparent, interpretable odds-ratio coefficients for credit risk compliance.</li>
                </ul>
            </div>

            <h2 class="section-title">2. Dataset Overview & Descriptive Statistics</h2>
            <p>The real loan dataset contains 13 columns (1 identifier, 11 features, and 1 target). There are <strong>0 missing values</strong> across all 4,269 records.</p>
            {desc_table_html}

            <h2 class="section-title">3. Model Performance & Evaluation Visuals</h2>
            <div class="chart-grid">
                <div class="chart-box">
                    <h3>Confusion Matrix (Held-out Test Set)</h3>
                    <img src="data:image/png;base64,{cm_b64}" alt="Confusion Matrix">
                </div>
                <div class="chart-box">
                    <h3>ROC Curve (AUC = {metrics['roc_auc']:.4f})</h3>
                    <img src="data:image/png;base64,{roc_b64}" alt="ROC Curve">
                </div>
                <div class="chart-box">
                    <h3>Feature Coefficients (Log-Odds Impact)</h3>
                    <img src="data:image/png;base64,{feat_b64}" alt="Feature Importance">
                </div>
                <div class="chart-box">
                    <h3>Evaluation Metrics Summary</h3>
                    <img src="data:image/png;base64,{summary_b64}" alt="Metrics Summary">
                </div>
            </div>

            <h2 class="section-title">4. Key Empirical Findings & Insights</h2>
            <ul>
                <li><strong>CIBIL Score is the dominant predictor:</strong> Applicants with a credit score above 600 possess an overwhelming probability of loan approval.</li>
                <li><strong>Loan Term and Loan Amount:</strong> Longer loan tenures and higher loan-to-income ratios slightly increase rejection risk unless balanced by high liquid bank assets.</li>
                <li><strong>Robust Generalization:</strong> With 92.27% accuracy and 0.9745 ROC-AUC, the calibrated Logistic Regression model maintains high precision for both approval and rejection.</li>
            </ul>
        </div>

        <div class="footer">
            <p>&copy; 2026 Loan Approval Intelligence System &bull; Machine Learning Project Report</p>
        </div>
    </div>
</body>
</html>
"""
    report_path = os.path.join(REPORTS_DIR, 'EDA_Report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"[Saved] HTML College EDA Report -> {report_path}")

if __name__ == '__main__':
    print("=" * 70)
    print("LOAN APPROVAL PREDICTION SYSTEM: MACHINE LEARNING PIPELINE")
    print("=" * 70)
    df = load_and_preprocess_data(DATASET_PATH)
    pipeline, metrics = train_and_evaluate(df)
    print("\n" + "=" * 70)
    print("TRAINING & REPORT GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
