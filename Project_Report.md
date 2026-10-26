# A MAJOR PROJECT REPORT ON
# **LOAN APPROVAL PREDICTION SYSTEM USING MACHINE LEARNING**

---

### **A Project Report Submitted in Partial Fulfillment of the Requirements for the Degree of Bachelor of Technology / Science in Computer Science and Engineering / Data Science**

**Submitted By:**  
**Student Name:** Vrunda  
**Roll No / Enrollment No:** [Your Roll Number Here]  
**Department:** Department of Computer Science & Engineering  
**Academic Year:** 2026 – 2027  

**Under the Guidance of:**  
**Project Guide / Mentor:** [Internal Guide Name]  
**Designation:** Assistant Professor / Associate Professor  

---

## 📜 CERTIFICATE OF APPROVAL

This is to certify that the project entitled **"Loan Approval Prediction System Using Machine Learning"** submitted by **Vrunda** in partial fulfillment of the requirements for the award of the degree is a bona fide record of work carried out under my supervision and guidance.

The project work embodies the results of original research and has reached the standard necessary for submission. To the best of my knowledge, this work has not been submitted elsewhere for the award of any degree or diploma.

<br><br>

___________________________ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ___________________________  
**Internal Project Guide** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Head of Department (HOD)**  
Department of Computer Science & Engineering &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Department of Computer Science & Engineering  

---

## 📑 TABLE OF CONTENTS

1. [Problem Statement](#1-problem-statement)
2. [Project Objectives](#2-project-objectives)
3. [Introduction](#3-introduction)
4. [Dataset Description](#4-dataset-description)
5. [Input Features and Target Variable](#5-input-features-and-target-variable)
6. [Data Preprocessing Pipeline](#6-data-preprocessing-pipeline)
7. [Exploratory Data Analysis (EDA)](#7-exploratory-data-analysis-eda)
8. [Machine Learning Algorithm: Logistic Regression](#8-machine-learning-algorithm-logistic-regression)
9. [Model Training and Testing](#9-model-training-and-testing)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Results Based on Actual Execution](#11-results-based-on-actual-execution)
12. [Comparative Benchmark Analysis](#12-comparative-benchmark-analysis)
13. [Web Application Architecture & User Interface](#13-web-application-architecture--user-interface)
14. [Limitations](#14-limitations)
15. [Conclusion](#15-conclusion)
16. [Future Scope](#16-future-scope)
17. [References](#17-references)

---

## 1. PROBLEM STATEMENT

In commercial and retail banking institutions, the underwriting of loans represents the core driver of revenue and the primary source of operational credit risk. A conventional credit assessment relies upon manual document verification, subjective rule-based heuristics, and lengthy committee approvals. This approach presents several challenges:
- **Delayed Processing Times**: Turnaround times frequently extend across days or weeks.
- **Inconsistent Decision Boundaries**: Human assessors may exhibit subjective variance or cognitive bias.
- **Credit Default Risk**: Failure to identify subprime credit indicators leads directly to Non-Performing Assets (NPAs).

The objective is to formulate an intelligent, automated credit underwriting decisioning engine that utilizes machine learning classification to predict whether a prospective borrower's application is likely to be **Approved** or **Rejected** based on verified socio-economic, financial, and collateral asset profiles.

---

## 2. PROJECT OBJECTIVES

The specific milestones established for this project comprise:
1. **Data Ingestion and Cleansing**: Perform rigorous sanitization of the provided `loan_approval_dataset (1).csv` dataset (addressing whitespace issues in column headers and string values).
2. **Exploratory Data Analysis (EDA)**: Statistically audit distribution patterns, credit bureau score cut-offs, income-to-loan ratios, and collateral coverage across classes.
3. **Robust Preprocessing Pipeline**: Build a Scikit-Learn `ColumnTransformer` pipeline that performs median imputation, feature standard scaling (`StandardScaler`), and categorical one-hot encoding without data leakage.
4. **Primary Model Training**: Train and calibrate a **Logistic Regression** classifier with fixed random states for strict reproducibility.
5. **Algorithmic Benchmarking**: Compare Logistic Regression with Decision Tree and Random Forest classifiers on the same stratified held-out test split.
6. **Automated Documentation and Reporting**: Generate standalone HTML and graphical visual reports (`EDA_Report.html`, `confusion_matrix.png`, `model_evaluation.png`, `feature_analysis.png`).
7. **Production Web Interface**: Deploy an interactive, multi-page **Streamlit** dashboard featuring a custom banking design aesthetic (Navy, Royal Blue, and Soft Cyan).

---

## 3. INTRODUCTION

Retail lending is a foundational pillar of modern financial systems. Loan products—including personal loans, home mortgages, and small business credit lines—empower economic mobility while generating interest yield for banks. However, default occurrences threaten institutional solvency and systemic economic stability.

With the advent of financial technology (Fintech) and accessible statistical learning libraries, machine learning has transitioned from an exploratory paradigm into mission-critical banking infrastructure. By analyzing multi-dimensional historical records, classification algorithms discern subtle non-linear dependencies between applicant attributes (such as credit history, debt burden, and asset liquidity) and repayment success.

This project delivers an end-to-end implementation of such a system, demonstrating the complete data science lifecycle from exploratory investigation to production-grade model deployment.

---

## 4. DATASET DESCRIPTION

The system utilizes the verified empirical loan dataset stored at `dataset/loan_approval_dataset (1).csv`. 

### Key Dataset Dimensions:
- **Total Record Count ($N$):** 4,269 rows
- **Total Attribute Count ($D$):** 13 columns (1 identifier, 11 predictor features, 1 binary target)
- **Missing Value Count:** 0 (0.00% missing values across all columns)
- **Duplicate Records:** 0 exact duplicate instances
- **Target Distribution:**
  - `Approved`: 2,656 instances (62.22%)
  - `Rejected`: 1,613 instances (37.78%)

---

## 5. INPUT FEATURES AND TARGET VARIABLE

| Feature Column Name | Data Type | Domain / Range | Description |
| :--- | :--- | :--- | :--- |
| `loan_id` | Integer | 1 to 4,269 | Unique application identifier (Excluded from predictive modeling) |
| `no_of_dependents` | Integer | 0 to 5 | Number of financial dependents supported by the applicant |
| `education` | Categorical | `Graduate`, `Not Graduate` | Highest formal educational qualification |
| `self_employed` | Categorical | `Yes`, `No` | Self-employment status / business entrepreneurship flag |
| `income_annum` | Numerical | ₹200,000 to ₹9,900,000 | Total annual verified income (INR) |
| `loan_amount` | Numerical | ₹300,000 to ₹39,500,000 | Total loan principal requested (INR) |
| `loan_term` | Numerical | 2 to 20 years | Agreed repayment period in years (even intervals) |
| `cibil_score` | Numerical | 300 to 900 | Credit Information Bureau (India) Limited credit score |
| `residential_assets_value` | Numerical | -₹100,000 to ₹29,100,000 | Appraised market valuation of residential property |
| `commercial_assets_value` | Numerical | ₹0 to ₹19,400,000 | Appraised market valuation of commercial property |
| `luxury_assets_value` | Numerical | ₹300,000 to ₹39,200,000 | Market value of vehicles, luxury goods, and jewelry |
| `bank_asset_value` | Numerical | ₹0 to ₹14,700,000 | Total liquid cash deposits, fixed deposits, and bank balances |
| **`loan_status` (Target)** | **Categorical** | **`Approved`, `Rejected`** | **Final underwriting decision outcome** |

---

## 6. DATA PREPROCESSING PIPELINE

Data leakage is a critical vulnerability in applied machine learning, occurring when information from the evaluation partition inadvertently influences training transformations. To eliminate leakage, all preprocessing operations were encapsulated inside a Scikit-Learn `Pipeline` and `ColumnTransformer` fitted solely on the training partition ($80\%$).

```text
Raw Features X
     │
     ├─── [Numerical Features] ───> SimpleImputer(median) ───> StandardScaler() ──────┐
     │                                                                                ├───> Logistic Regression
     └─── [Categorical Features] ─> SimpleImputer(mode) ────> OneHotEncoder(drop=1st) ┘
```

1. **Header & Value Sanitization**: Cleaned column headers via `df.columns = df.columns.str.strip()` and trimmed white spaces from string records.
2. **Feature Stratification**: Segregated features ($X$) and ground truth ($y$). Excluded `loan_id` to prevent artificial index memorization.
3. **Partitioning**: Applied stratified $80/20$ train-test splitting with `random_state=42`, yielding:
   - **Training Set**: 3,415 records ($2,125$ Approved, $1,290$ Rejected)
   - **Test Set**: 854 records ($531$ Approved, $323$ Rejected)
4. **Standardization**: Standardized continuous numerical variables to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$):
   $$z = \frac{x - \mu}{\sigma}$$
5. **One-Hot Encoding**: Transformed binary categorical features into dummy variables with `drop='first'` to avoid multicollinearity.

---

## 7. EXPLORATORY DATA ANALYSIS (EDA)

Exploratory analysis produced several empirical findings:

1. **The CIBIL Score Threshold**: The credit bureau score is the single most decisive factor in determining approval. A sharp decision cut-off exists near **550**. Applicants with scores above 550 achieve near-universal approval, whereas applicants with scores below 550 suffer heavy rejection rates.
2. **Income vs. Loan Principal**: A strong positive correlation exists between annual income and approved loan amounts ($r \approx 0.93$). Banks enforce strict Debt-to-Income (DTI) thresholds.
3. **Asset Portfolios**: Total collateral valuation (the sum of residential, commercial, luxury, and bank deposit valuations) provides strong secondary validation, acting as security against default.

*(Detailed graphical visualizations can be viewed in `Reports/EDA_Report.html` and `Reports/feature_analysis.png`.)*

---

## 8. MACHINE LEARNING ALGORITHM: LOGISTIC REGRESSION

Logistic Regression is a foundational linear classification algorithm that models the probability of a binary outcome using the sigmoid (logistic) link function:

$$P(Y = 1 | X) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Where $z$ is the linear combination of weighted features and intercept:
$$z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_k x_k$$

The model is optimized using maximum likelihood estimation (MLE) with cross-entropy log-loss:
$$\mathcal{L}(\beta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$

### Fitted Feature Importance (Log-Odds & Odds Ratio):

| Rank | Feature | Model Coefficient ($\beta$) | Odds Ratio ($e^\beta$) | Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **`cibil_score`** | **+4.1041** | **60.59** | Highest positive driver. +1 std increase multiplies odds of approval by ~60x |
| 2 | **`income_annum`** | **-1.6471** | **0.19** | Higher baseline income adjusts capacity expectations |
| 3 | **`loan_amount`** | **+1.2511** | **3.49** | Positive scaling factor with qualified collateral |
| 4 | **`loan_term`** | **-0.8496** | **0.43** | Longer tenures elevate cumulative default exposure, dampening odds |
| 5 | **`luxury_assets_value`** | **+0.3186** | **1.38** | Liquid high-value assets bolster applicant security profile |
| 6 | **`self_employed_Yes`** | **+0.2118** | **1.24** | Self-employed profiles show slightly higher commercial demand |
| 7 | **`education_Not Graduate`**| **-0.1078** | **0.90** | Slight downward pressure compared to graduate peers |
| 8 | **`bank_asset_value`** | **+0.1036** | **1.11** | Liquid bank balances positively enhance repayment capacity |
| 9 | **`commercial_assets_value`**| **+0.0813** | **1.08** | Commercial holdings provide collateral backing |
| 10 | **`no_of_dependents`** | **-0.0389** | **0.96** | Higher dependent count slightly increases household expenditure |
| 11 | **`residential_assets_value`**| **+0.0277** | **1.03** | Real estate holdings support net worth calculations |

---

## 9. MODEL TRAINING AND TESTING

Training and evaluation were conducted using the environment setup script `project.py`. The model was trained over 3,415 samples and evaluated on the held-out test partition of 854 samples.

### Test Set Confusion Matrix Breakdown:
- **True Approved (Sensitivity hit):** 505 applicants correctly classified as Approved
- **True Rejected (Specificity hit):** 283 applicants correctly classified as Rejected
- **False Approved (Type I error):** 40 applicants incorrectly classified as Approved
- **False Rejected (Type II error):** 26 applicants incorrectly classified as Rejected

```text
                      PREDICTED APPROVED    PREDICTED REJECTED
ACTUAL APPROVED              505                   26
ACTUAL REJECTED               40                  283
```

---

## 10. EVALUATION METRICS

Model efficacy was evaluated using standard classification metrics:

1. **Accuracy**: Overall proportion of correct predictions:
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{505 + 283}{854} = \mathbf{92.27\%}$$

2. **Precision (Positive Predictive Value for Approval)**:
   $$\text{Precision} = \frac{TP}{TP + FP} = \frac{505}{505 + 40} = \mathbf{92.66\%}$$

3. **Recall (Sensitivity / True Positive Rate)**:
   $$\text{Recall} = \frac{TP}{TP + FN} = \frac{505}{505 + 26} = \mathbf{95.10\%}$$

4. **F1-Score**: Harmonic mean of Precision and Recall:
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \mathbf{0.9387}$$

5. **Area Under the ROC Curve (ROC-AUC)**: Measures class separability across all classification thresholds:
   $$\text{ROC-AUC} = \mathbf{0.9745}$$

---

## 11. RESULTS BASED ON ACTUAL EXECUTION

### Primary Model (Logistic Regression) Metrics:
- **Training Set Accuracy:** 91.48%
- **Testing Set Accuracy:** **92.27%**
- **Test Set Precision:** 0.9266 (92.66%)
- **Test Set Recall:** 0.9510 (95.10%)
- **Test Set F1-Score:** 0.9387 (93.87%)
- **Test Set ROC-AUC:** 0.9745 (97.45%)

The negligible disparity between training accuracy (91.48%) and test accuracy (92.27%) confirms that the model generalizes well without overfitting or variance degradation.

---

## 12. COMPARATIVE BENCHMARK ANALYSIS

To evaluate Logistic Regression relative to alternative supervised architectures, Decision Tree and Random Forest classifiers were trained on the identical pipeline:

| Model Architecture | Train Accuracy | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Primary)** | **91.48%** | **92.27%** | **0.9266** | **0.9510** | **0.9387** | **0.9745** |
| Decision Tree Classifier | 97.86% | 97.19% | 0.9617 | 0.9944 | 0.9778 | 0.9937 |
| Random Forest Classifier | 99.77% | 97.78% | 0.9758 | 0.9887 | 0.9822 | 0.9964 |

### Architectural Trade-off Discussion:
While tree ensemble models achieve slightly higher nominal test accuracy (97.78%), **Logistic Regression remains the primary model of choice** in banking and credit underwriting due to regulatory audit requirements. Under guidelines such as the Equal Credit Opportunity Act (ECOA) and Basel III standards, credit institutions must provide clear, linear, explainable rationales for credit denials. Logistic Regression offers transparent coefficients and odds ratios that satisfy these regulatory expectations.

---

## 13. WEB APPLICATION ARCHITECTURE & USER INTERFACE

The web dashboard is developed in **Streamlit** using custom CSS and HTML styling.

### Visual Identity:
- **Hero Banner:** Full-width digital banking skyscraper asset with dark navy glassmorphic overlay.
- **Color Palette:** Deep Navy (`#0f172a`), Royal Blue (`#1e3a8a`), Soft Cyan (`#0284c7`, `#38bdf8`), and White Cards (`#ffffff`).
- **Typography:** Modern sans-serif typography (`Plus Jakarta Sans`).
- **Interactive Visualizations:** Plotly interactive donut charts, boxplots, histograms, and scatter plots.

### Dashboard Pages:
1. **🏠 Home Dashboard:** Displays live KPI cards, portfolio distribution donut chart, and operational workflow steps.
2. **⚡ Loan Approval Prediction:** Real-time form supporting dynamic input sliders, currency formatting, and instant classification output with probability scoring.
3. **📊 Dataset Explorer:** Data exploration table equipped with multi-attribute filtering and CSV export functionality.
4. **📈 Exploratory Data Analysis (EDA):** Visual gallery examining CIBIL score cut-offs, income vs loan trends, and asset portfolio distributions.
5. **🎯 Model Performance:** Visual displays of the confusion matrix, ROC curves, feature impact weights, and benchmark tables.
6. **ℹ️ About the Project:** Full architectural documentation, schema descriptions, and references.

---

## 14. LIMITATIONS

1. **Synthetic/Standardized Data Limitations**: The dataset does not incorporate temporal repayment trajectories, macroeconomic parameters (e.g., inflation rates or benchmark repo rates), or existing loan count.
2. **Static Thresholds**: Credit dynamics can shift during economic downturns; periodic model retraining is necessary.
3. **Categorical Breadth**: The demographic features (`education`, `self_employed`) are binary and do not account for industry classification or years in employment.

---

## 15. CONCLUSION

The **Loan Approval Prediction System** successfully automates credit decisioning with an accuracy of **92.27%** and an ROC-AUC of **0.9745** on held-out test data. By combining an end-to-end preprocessing pipeline with a Logistic Regression classifier, the system maintains high predictive fidelity while retaining the interpretability required in financial risk management. The accompanying Streamlit web dashboard translates these statistical predictions into an accessible, responsive decision-support interface.

---

## 16. FUTURE SCOPE

1. **Credit Bureau API Integration**: Direct integration with credit bureau APIs (CIBIL, Experian, Equifax) for automated score fetching.
2. **Explainable AI (XAI)**: Implementation of SHAP (SHapley Additive exPlanations) and LIME to generate individualized adverse action notices for denied applicants.
3. **Deep Learning & Gradient Boosting**: Benchmarking with XGBoost, LightGBM, and TabNet architectures for institutional portfolio sizing.
4. **Cloud Containerization**: Packaging via Docker and deployment onto cloud platforms (AWS ECS / Azure App Service / GCP Cloud Run).

---

## 17. REFERENCES

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer Science & Business Media.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
3. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 51-56.
4. Streamlit Documentation: Modern Web Applications for Machine Learning. https://docs.streamlit.io
5. Reserve Bank of India (RBI) Master Direction on Non-Banking Financial Company - Systemically Important Non-Deposit taking Company and Deposit taking Company (Reserve Bank) Directions, 2016 (Credit Underwriting Guidelines).
