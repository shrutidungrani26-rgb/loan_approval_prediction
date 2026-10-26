# 🏦 Loan Approval Intelligence - Fintech ML System

An end-to-end Machine Learning classification and interactive fintech dashboard for predicting retail and commercial loan approval status (**Approved** vs **Rejected**).

Built using **Python**, **Pandas**, **Scikit-Learn**, **Matplotlib**, **Seaborn**, **Plotly**, and **Streamlit** with a fintech banking visual theme (Deep Navy Blue, Royal Blue, and Soft Cyan accents).

---

## 📌 Project Overview

In banking and credit underwriting, determining applicant eligibility accurately is paramount to maintaining a healthy loan book while minimizing default risk. This project develops a machine learning classification engine that processes applicant demographics, income, credit history, loan requirements, and collateral asset profiles to output reliable approval predictions with calibrated class probabilities.

- **Primary Model**: Logistic Regression Pipeline (with Standard Scaling & One-Hot Encoding)
- **Benchmarked Models**: Decision Tree Classifier, Random Forest Classifier
- **Test Accuracy**: **92.27%**
- **Test ROC-AUC**: **0.9745**
- **Web App**: Interactive Multi-Page Streamlit Dashboard with custom CSS fintech identity

---

## 📁 Project Folder Structure

```text
Loan_Approval_Prediction/
│
├── app.py                      # Multi-page interactive Streamlit dashboard
├── project.py                  # End-to-end ML training, evaluation, and EDA generator
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & execution guide
├── Project_Report.md           # Formal college/academic project report
│
├── dataset/
│   └── loan_approval_dataset (1).csv   # Uploaded original dataset (4,269 records)
│
├── models/
│   ├── logistic_regression_pipeline.pkl # Fitted Scikit-learn preprocessing & model pipeline
│   └── model_metadata.json              # Trained evaluation metrics & feature weights
│
├── assets/
│   ├── hero_bg.jpg             # High-resolution fintech banking headquarters hero banner
│   └── logo.png                # Banking shield brand emblem
│
└── Reports/
    ├── EDA_Report.html         # Comprehensive interactive HTML EDA report
    ├── confusion_matrix.png    # Test set confusion matrix visualization
    ├── model_evaluation.png    # ROC curves & benchmark comparison charts
    └── feature_analysis.png    # Feature importance & odds ratio analysis
```

---

## 📊 Dataset Information

- **Total Applications**: 4,269 records
- **Total Features**: 13 (1 identifier, 11 predictive features, 1 target)
- **Target Variable**: `loan_status` (`Approved`: 2,656 [62.2%], `Rejected`: 1,613 [37.8%])
- **Input Features**:
  1. `no_of_dependents`: Number of financial dependents (0 to 5)
  2. `education`: Educational attainment (`Graduate`, `Not Graduate`)
  3. `self_employed`: Self-employment flag (`Yes`, `No`)
  4. `income_annum`: Verified gross annual income (INR)
  5. `loan_amount`: Principal loan amount requested (INR)
  6. `loan_term`: Repayment duration in years (2 to 20 years)
  7. `cibil_score`: Credit bureau score (300 to 900)
  8. `residential_assets_value`: Valuation of residential real estate (INR)
  9. `commercial_assets_value`: Valuation of commercial properties (INR)
  10. `luxury_assets_value`: Valuation of luxury/personal assets (INR)
  11. `bank_asset_value`: Liquid bank balances and fixed deposits (INR)

*Note: `loan_id` is excluded from modeling to avoid index bias.*

---

## ⚡ Installation Instructions

Ensure you have Python (version 3.10+ up to 3.14) installed on your system.

1. **Open your terminal or command prompt** and navigate to the project directory:
   ```bash
   cd "C:\Users\DELL\Desktop\vrunda 2\Loan_Approval_Prediction"
   ```

2. **Install required dependencies**:
   ```bash
   python -m pip install -r requirements.txt
   ```
   *(Or specify your Python environment executable, e.g. `C:\Users\DELL\PREDICTION\Scripts\pip.exe install -r requirements.txt`)*

---

## 🏋️ Model Training & Artifact Generation

To execute data validation, stratified splitting, model training, evaluation, and chart generation:

```bash
python project.py
```

This script produces:
- `models/logistic_regression_pipeline.pkl`
- `models/model_metadata.json`
- `Reports/confusion_matrix.png`
- `Reports/model_evaluation.png`
- `Reports/feature_analysis.png`
- `Reports/EDA_Report.html`

---

## 🚀 How to Run the Streamlit Application

Start the web server locally using Streamlit:

```bash
python -m streamlit run app.py
```
*(Or `C:\Users\DELL\PREDICTION\Scripts\streamlit.exe run app.py`)*

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 🌟 Main Dashboard Features

1. **🏠 Home Dashboard**: Executive KPI cards, portfolio distribution pie/donut chart, and system architecture summary.
2. **⚡ Loan Approval Prediction**: Professional underwriting application form with real-time probability scoring and collateral coverage indicators.
3. **📊 Dataset Explorer**: Searchable, filterable tabular explorer with dynamic multi-criteria filters and CSV export.
4. **📈 Exploratory Data Analysis (EDA)**: Interactive Plotly distributions, CIBIL cut-off analysis, asset breakdown, and correlation heatmaps.
5. **🎯 Model Performance**: Evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC), confusion matrix, and multi-model benchmark comparisons.
6. **ℹ️ About the Project**: Complete documentation, architectural methodology, and academic problem formulation.

---

## ⚠️ Limitations & Educational Disclaimer

- **Data Scope**: The dataset focuses on applicant financial metrics without macroeconomic parameters (e.g., benchmark interest rates or debt-service ratios).
- **Academic Purpose**: This software was designed for educational, academic, and portfolio demonstration purposes. Live banking decisions incorporate proprietary institutional policies and regulatory compliance.
