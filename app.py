"""
Loan Approval Intelligence System - Streamlit Web Application
==============================================================
Theme: Deep Navy Blue & Royal Blue with Soft Cyan Accents
Engine: Scikit-learn Logistic Regression Pipeline
Dataset: loan_approval_dataset (1).csv
Author: Vrunda
"""

import os
import json
import base64
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# 1. PAGE CONFIGURATION & METADATA
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Intelligence | Fintech AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
REPORTS_DIR = os.path.join(BASE_DIR, 'Reports')

# Find hero background image (png or jpg)
HERO_BG_PATH = os.path.join(ASSETS_DIR, 'hero_bg.png')
if not os.path.exists(HERO_BG_PATH):
    HERO_BG_PATH = os.path.join(ASSETS_DIR, 'hero_bg.jpg')
LOGO_PATH = os.path.join(ASSETS_DIR, 'logo.png')

# ----------------------------------------------------------------------
# 2. CACHED DATA & MODEL LOADERS
# ----------------------------------------------------------------------
@st.cache_data
def load_dataset():
    candidates = [
        os.path.join(BASE_DIR, 'dataset', 'loan_approval_dataset.csv'),
        os.path.join(BASE_DIR, 'dataset', 'loan_approval_dataset (1).csv'),
        r"C:\Users\DELL\Desktop\vrunda 2\Loan_Approval_Prediction\dataset\loan_approval_dataset.csv",
        r"C:\Users\DELL\Desktop\vrunda 2\Loan_Approval_Prediction\dataset\loan_approval_dataset (1).csv",
        r"C:\Users\DELL\Desktop\vrunda 2\dataset\loan_approval_dataset.csv",
        r"C:\Users\DELL\Downloads\loan_approval_dataset (1).csv"
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            for col in ['education', 'self_employed', 'loan_status']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
            return df
    return None

@st.cache_resource
def load_pipeline():
    candidates = [
        os.path.join(BASE_DIR, 'models', 'logistic_regression_pipeline.pkl'),
        r"C:\Users\DELL\Desktop\vrunda 2\Loan_Approval_Prediction\models\logistic_regression_pipeline.pkl",
        r"C:\Users\DELL\Desktop\vrunda 2\models\logistic_regression_pipeline.pkl"
    ]
    for path in candidates:
        if os.path.exists(path):
            return joblib.load(path)
    return None

@st.cache_data
def load_metadata():
    candidates = [
        os.path.join(BASE_DIR, 'models', 'model_metadata.json'),
        os.path.join(BASE_DIR, 'Reports', 'model_metrics.json'),
        r"C:\Users\DELL\Desktop\vrunda 2\Loan_Approval_Prediction\models\model_metadata.json",
        r"C:\Users\DELL\Desktop\vrunda 2\Loan_Approval_Prediction\Reports\model_metrics.json"
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    return None

df_raw = load_dataset()
pipeline = load_pipeline()
metadata = load_metadata()

# Convert image to base64 for embedding in custom CSS
def img_to_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

hero_b64 = img_to_base64(HERO_BG_PATH)
logo_b64 = img_to_base64(LOGO_PATH)

# ----------------------------------------------------------------------
# 3. GLOBAL FINTECH STYLING (NAVY, ROYAL BLUE, SOFT CYAN ACCENTS)
# ----------------------------------------------------------------------
st.markdown(f"""
<style>
    /* Global Typography & Font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }}
    
    /* Main Background: Subtle light/soft blue background for readability */
    .stApp {{
        background: linear-gradient(180deg, #f0f4f9 0%, #f8fafc 100%);
        color: #0f172a;
    }}
    
    /* Top Hero Banner */
    .fintech-hero {{
        position: relative;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(30, 58, 138, 0.88) 60%, rgba(2, 132, 199, 0.85) 100%),
                    url('data:image/jpeg;base64,{hero_b64}') center/cover no-repeat;
        padding: 45px 35px;
        margin-bottom: 25px;
        color: #ffffff;
        box-shadow: 0 16px 36px -10px rgba(15, 23, 42, 0.35);
        border: 1px solid rgba(56, 189, 248, 0.3);
        overflow: hidden;
    }}
    
    .hero-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(56, 189, 248, 0.18);
        border: 1px solid rgba(56, 189, 248, 0.45);
        color: #38bdf8;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }}
    
    .hero-title {{
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.15;
        background: linear-gradient(90deg, #ffffff 0%, #e0f2fe 70%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }}
    
    .hero-subtitle {{
        font-size: 1.15rem;
        color: #cbd5e1;
        margin-top: 10px;
        margin-bottom: 0;
        font-weight: 400;
    }}
    
    /* Glassmorphic Metric Cards */
    .metric-card {{
        background: #ffffff;
        border-radius: 16px;
        padding: 22px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .metric-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 16px 32px rgba(2, 132, 199, 0.12);
        border-color: #38bdf8;
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #1e3a8a 0%, #38bdf8 100%);
    }}
    
    .metric-title {{
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.5px;
    }}
    
    .metric-value {{
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        margin: 6px 0;
    }}
    
    .metric-footer {{
        font-size: 0.82rem;
        color: #0284c7;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 4px;
    }}
    
    /* Section Headers */
    .section-head {{
        margin: 30px 0 18px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    
    .section-head h2 {{
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
    }}
    
    .section-pill {{
        background: #e0f2fe;
        color: #0284c7;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 700;
    }}
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0b132b 0%, #1c2541 60%, #0f172a 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }}
    
    [data-testid="stSidebar"] * {{
        color: #f1f5f9 !important;
    }}
    
    .sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0 20px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }}
    
    .sidebar-brand img {{
        width: 44px;
        height: 44px;
        border-radius: 10px;
        border: 1px solid #38bdf8;
    }}
    
    .brand-title {{
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -0.3px;
        line-height: 1.2;
    }}
    
    .brand-sub {{
        font-size: 0.72rem;
        color: #38bdf8 !important;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    
    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, #1e3a8a 0%, #0284c7 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 700;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3);
        transition: all 0.25s ease;
        width: 100%;
    }}
    
    .stButton>button:hover {{
        background: linear-gradient(135deg, #172554 0%, #0369a1 100%);
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.45);
        transform: translateY(-2px);
    }}
    
    /* Prediction Result Cards */
    .result-approved {{
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid #10b981;
        border-radius: 18px;
        padding: 26px;
        color: #065f46;
        box-shadow: 0 12px 30px rgba(16, 185, 129, 0.15);
    }}
    
    .result-rejected {{
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border: 2px solid #f43f5e;
        border-radius: 18px;
        padding: 26px;
        color: #9f1239;
        box-shadow: 0 12px 30px rgba(244, 63, 94, 0.15);
    }}
    
    /* Form Container */
    .form-container {{
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 26px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        margin-bottom: 20px;
    }}
    
    /* Footer */
    .fintech-footer {{
        text-align: center;
        padding: 30px 10px;
        margin-top: 50px;
        border-top: 1px solid #e2e8f0;
        color: #64748b;
        font-size: 0.85rem;
    }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-brand">
        <img src="data:image/png;base64,{logo_b64}" alt="Logo">
        <div>
            <div class="brand-title">APEX LEND</div>
            <div class="brand-sub">Fintech AI System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    menu = st.radio(
        "Navigation Menu",
        [
            "🏠 Home Dashboard",
            "⚡ Loan Approval Prediction",
            "📊 Dataset Explorer",
            "📈 Exploratory Data Analysis",
            "🎯 Model Performance",
            "ℹ️ About the Project"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
    **System Status:**  
    🟢 Core Engine: `Active`  
    🧠 Primary Model: `Logistic Regression`  
    🎯 Test Accuracy: `92.27%`  
    📁 Records: `4,269`
    """)
    st.caption("© 2026 Apex Credit Technologies")

# Helper for formatted INR currency
def format_inr(val):
    if val >= 10000000:
        return f"₹{val/10000000:.2f} Cr"
    elif val >= 100000:
        return f"₹{val/100000:.2f} Lakh"
    else:
        return f"₹{val:,.0f}"

# ----------------------------------------------------------------------
# PAGE 1: HOME DASHBOARD
# ----------------------------------------------------------------------
if menu == "🏠 Home Dashboard":
    st.markdown(f"""
    <div class="fintech-hero">
        <span class="hero-badge">⚡ Advanced Risk Underwriting v2.4</span>
        <h1 class="hero-title">Loan Approval Intelligence</h1>
        <p class="hero-subtitle">Smart insights for informed loan decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_raw is not None:
        total_apps = len(df_raw)
        total_apprv = (df_raw['loan_status'] == 'Approved').sum()
        total_rej = (df_raw['loan_status'] == 'Rejected').sum()
        apprv_rate = (total_apprv / total_apps) * 100

        # KPI Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Applications</div>
                <div class="metric-value">{total_apps:,}</div>
                <div class="metric-footer">📋 Complete applicant volume</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Approved Loans</div>
                <div class="metric-value">{total_apprv:,}</div>
                <div class="metric-footer">✅ Validated creditworthiness</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Rejected Loans</div>
                <div class="metric-value">{total_rej:,}</div>
                <div class="metric-footer">⚠️ Risk-filtered applications</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Approval Rate</div>
                <div class="metric-value">{apprv_rate:.1f}%</div>
                <div class="metric-footer">📊 Portfolio approval ratio</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Overview Visuals & System Flow
        row_col1, row_col2 = st.columns([1.2, 1])

        with row_col1:
            st.markdown("""
            <div class="form-container">
                <h3 style="margin-top:0; color: #0f172a; font-weight:700;">Portfolio Status Distribution</h3>
            """, unsafe_allow_html=True)
            
            fig_pie = px.pie(
                df_raw,
                names='loan_status',
                values=[1]*len(df_raw),
                hole=0.55,
                color='loan_status',
                color_discrete_map={'Approved': '#0284c7', 'Rejected': '#f43f5e'}
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#ffffff', width=2)))
            fig_pie.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=320,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with row_col2:
            st.markdown("""
            <div class="form-container" style="height: 380px;">
                <h3 style="margin-top:0; color: #0f172a; font-weight:700;">How The AI System Operates</h3>
                <div style="font-size: 0.95rem; line-height: 1.7; color: #334155;">
                    <p><strong>1. Input Ingestion:</strong> Applicant financial details (income, credit score, tenure, assets) are captured with strict validation.</p>
                    <p><strong>2. Standardized Preprocessing:</strong> Numerical values are scaled and categorical flags one-hot encoded without data leakage.</p>
                    <p><strong>3. Logistic Regression Engine:</strong> Computes the probabilistic log-odds of repayment based on historical banking patterns.</p>
                    <p><strong>4. Instant Outcome & Odds:</strong> Returns real-time decisioning with confidence probabilities and transparent factor breakdowns.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Call to Action to Prediction Form
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.info("💡 **Ready to evaluate a loan applicant?** Navigate to the **Loan Approval Prediction** page in the sidebar to simulate real-time loan underwriting.")

# ----------------------------------------------------------------------
# PAGE 2: LOAN APPROVAL PREDICTION
# ----------------------------------------------------------------------
elif menu == "⚡ Loan Approval Prediction":
    st.markdown("""
    <div class="fintech-hero">
        <span class="hero-badge">💳 Underwriting Portal</span>
        <h1 class="hero-title">Real-Time Loan Prediction</h1>
        <p class="hero-subtitle">Enter applicant financial credentials to assess underwriting approval probability.</p>
    </div>
    """, unsafe_allow_html=True)

    if pipeline is None:
        st.error("⚠️ Trained model not found! Please run `python project.py` first to train and generate the model pipeline.")
    else:
        with st.form("loan_prediction_form"):
            st.markdown("<h4 style='color: #0f172a; margin-top:0;'>1. Applicant Demographics & Credit Standing</h4>", unsafe_allow_html=True)
            d_col1, d_col2, d_col3, d_col4 = st.columns(4)
            with d_col1:
                dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2, step=1,
                                             help="Number of financially dependent family members")
            with d_col2:
                education = st.selectbox("Education Level", ["Graduate", "Not Graduate"], index=0,
                                         help="Formal highest education attained")
            with d_col3:
                self_employed = st.selectbox("Self Employed", ["No", "Yes"], index=0,
                                             help="Is the applicant an entrepreneur or independent business owner?")
            with d_col4:
                cibil_score = st.slider("CIBIL Credit Score", min_value=300, max_value=900, value=740, step=1,
                                        help="Credit bureau score (300 to 900). Scores >= 550 significantly boost approval odds.")

            st.markdown("---")
            st.markdown("<h4 style='color: #0f172a;'>2. Loan Request & Annual Income</h4>", unsafe_allow_html=True)
            l_col1, l_col2, l_col3 = st.columns(3)
            with l_col1:
                income_annum = st.number_input("Annual Income (INR ₹)", min_value=100000, max_value=50000000, value=6500000, step=100000,
                                               help="Total verified annual household/individual gross income")
                st.caption(f"Formatted: **{format_inr(income_annum)}**")
            with l_col2:
                loan_amount = st.number_input("Requested Loan Amount (INR ₹)", min_value=100000, max_value=100000000, value=18000000, step=200000,
                                              help="Principal loan amount requested")
                st.caption(f"Formatted: **{format_inr(loan_amount)}**")
            with l_col3:
                loan_term = st.slider("Loan Tenure (Years)", min_value=2, max_value=20, value=12, step=2,
                                      help="Repayment duration in years")
                st.caption(f"Tenure: **{loan_term} Years ({loan_term*12} Months)**")

            st.markdown("---")
            st.markdown("<h4 style='color: #0f172a;'>3. Asset Collateral & Portfolio Valuation (INR ₹)</h4>", unsafe_allow_html=True)
            a_col1, a_col2, a_col3, a_col4 = st.columns(4)
            with a_col1:
                residential_assets = st.number_input("Residential Assets", min_value=-100000, max_value=50000000, value=7500000, step=100000)
                st.caption(format_inr(residential_assets))
            with a_col2:
                commercial_assets = st.number_input("Commercial Assets", min_value=0, max_value=50000000, value=4000000, step=100000)
                st.caption(format_inr(commercial_assets))
            with a_col3:
                luxury_assets = st.number_input("Luxury Assets", min_value=0, max_value=50000000, value=15000000, step=100000,
                                                help="Jewelry, luxury vehicles, high-value personal assets")
                st.caption(format_inr(luxury_assets))
            with a_col4:
                bank_assets = st.number_input("Bank Deposits / Liquid Assets", min_value=0, max_value=50000000, value=5000000, step=100000)
                st.caption(format_inr(bank_assets))

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("🚀 Predict Loan Status")

        # Process Prediction
        if submit_btn:
            input_dict = {
                'no_of_dependents': [dependents],
                'education': [education],
                'self_employed': [self_employed],
                'income_annum': [income_annum],
                'loan_amount': [loan_amount],
                'loan_term': [loan_term],
                'cibil_score': [cibil_score],
                'residential_assets_value': [residential_assets],
                'commercial_assets_value': [commercial_assets],
                'luxury_assets_value': [luxury_assets],
                'bank_asset_value': [bank_assets]
            }
            input_df = pd.DataFrame(input_dict)

            # Predict class & probabilities
            pred = pipeline.predict(input_df)[0]
            classes = list(pipeline.classes_)
            app_idx = classes.index('Approved')
            rej_idx = classes.index('Rejected')

            prob_app = pipeline.predict_proba(input_df)[0][app_idx] * 100
            prob_rej = pipeline.predict_proba(input_df)[0][rej_idx] * 100

            total_assets = residential_assets + commercial_assets + luxury_assets + bank_assets
            asset_coverage = (total_assets / loan_amount) if loan_amount > 0 else 0

            st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
            if pred == 'Approved':
                st.markdown(f"""
                <div class="result-approved">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="background:#059669; color:white; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:700; text-transform:uppercase;">Application Decision</span>
                            <h2 style="margin: 8px 0 4px 0; font-size: 2.2rem; font-weight: 800; color:#064e3b;">LOAN APPROVED ✅</h2>
                            <p style="margin:0; font-size: 1.05rem; color:#047857;">Applicant meets credit risk standards under the trained Logistic Regression underwriting rules.</p>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size: 0.85rem; text-transform:uppercase; font-weight:700; color:#047857;">Approval Probability</div>
                            <div style="font-size: 2.8rem; font-weight:900; color:#065f46;">{prob_app:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-rejected">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="background:#e11d48; color:white; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:700; text-transform:uppercase;">Application Decision</span>
                            <h2 style="margin: 8px 0 4px 0; font-size: 2.2rem; font-weight: 800; color:#881337;">LOAN REJECTED ❌</h2>
                            <p style="margin:0; font-size: 1.05rem; color:#be123c;">Application indicators fall below threshold risk margins based on model criteria.</p>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size: 0.85rem; text-transform:uppercase; font-weight:700; color:#be123c;">Rejection Probability</div>
                            <div style="font-size: 2.8rem; font-weight:900; color:#9f1239;">{prob_rej:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Key Factors Breakdown
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            with f_col1:
                st.metric("CIBIL Rating", f"{cibil_score}", "Strong" if cibil_score >= 550 else "Subprime Risk")
            with f_col2:
                st.metric("Total Assets", format_inr(total_assets))
            with f_col3:
                st.metric("Asset-to-Loan Ratio", f"{asset_coverage:.2f}x", "Adequate" if asset_coverage >= 1.0 else "Low Collateral")
            with f_col4:
                st.metric("Loan-to-Income", f"{(loan_amount/income_annum):.2f}x", "Normal" if (loan_amount/income_annum) <= 4.5 else "Elevated Leverage")

            st.caption("ℹ️ **Educational Disclaimer:** *This prediction is generated via a Machine Learning pipeline for educational and academic evaluation. Real-world financial institutions apply multi-layered regulatory and subjective compliance criteria.*")

# ----------------------------------------------------------------------
# PAGE 3: DATASET EXPLORER
# ----------------------------------------------------------------------
elif menu == "📊 Dataset Explorer":
    st.markdown("""
    <div class="fintech-hero">
        <span class="hero-badge">📁 Data Warehouse</span>
        <h1 class="hero-title">Dataset Explorer</h1>
        <p class="hero-subtitle">Inspect, query, filter, and export the official loan approval dataset records.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_raw is not None:
        # Filter controls
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            status_filter = st.multiselect("Filter by Status", options=['Approved', 'Rejected'], default=['Approved', 'Rejected'])
        with f_col2:
            edu_filter = st.multiselect("Filter by Education", options=df_raw['education'].unique().tolist(), default=df_raw['education'].unique().tolist())
        with f_col3:
            emp_filter = st.multiselect("Filter by Self Employment", options=df_raw['self_employed'].unique().tolist(), default=df_raw['self_employed'].unique().tolist())
        with f_col4:
            min_cibil, max_cibil = st.slider("CIBIL Score Range", int(df_raw['cibil_score'].min()), int(df_raw['cibil_score'].max()),
                                             (int(df_raw['cibil_score'].min()), int(df_raw['cibil_score'].max())))

        # Apply filtering
        filtered_df = df_raw[
            (df_raw['loan_status'].isin(status_filter)) &
            (df_raw['education'].isin(edu_filter)) &
            (df_raw['self_employed'].isin(emp_filter)) &
            (df_raw['cibil_score'] >= min_cibil) &
            (df_raw['cibil_score'] <= max_cibil)
        ]

        st.markdown(f"**Showing {len(filtered_df):,} of {len(df_raw):,} records**")
        st.dataframe(filtered_df, use_container_width=True, height=400)

        # Download Button
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Records as CSV",
            data=csv_data,
            file_name="filtered_loan_approval_data.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.markdown("<h3 style='color:#0f172a;'>Dataset Metadata & Schema</h3>", unsafe_allow_html=True)
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            schema_df = pd.DataFrame({
                'Column Name': df_raw.columns,
                'Data Type': df_raw.dtypes.astype(str),
                'Non-Null Count': df_raw.notnull().sum(),
                'Null Values': df_raw.isnull().sum()
            })
            st.dataframe(schema_df, use_container_width=True)
        with meta_col2:
            st.markdown("""
            **Dataset Quality Summary:**
            - **Total Rows:** 4,269
            - **Total Features:** 13
            - **Target Column:** `loan_status` ('Approved' / 'Rejected')
            - **Missing Values:** 0 across all attributes
            - **Duplicates:** 0 exact duplicate applicant records
            - **Column Name Sanitization:** Trimmed leading whitespace (`.str.strip()`)
            """)

# ----------------------------------------------------------------------
# PAGE 4: EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------------------------------------------
elif menu == "📈 Exploratory Data Analysis":
    st.markdown("""
    <div class="fintech-hero">
        <span class="hero-badge">📊 Data Intelligence</span>
        <h1 class="hero-title">Exploratory Data Analysis</h1>
        <p class="hero-subtitle">Interactive analytical visualizations uncovering credit underwriting dynamics.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_raw is not None:
        tab1, tab2, tab3 = st.tabs(["🎯 Credit & Target Dynamics", "💰 Income & Loan Principal", "🏛️ Asset Valuations & Heatmap"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                fig1 = px.histogram(
                    df_raw, x="loan_status", color="loan_status",
                    title="Loan Status Class Distribution",
                    color_discrete_map={'Approved': '#0284c7', 'Rejected': '#f43f5e'}
                )
                fig1.update_layout(showlegend=False, template="plotly_white")
                st.plotly_chart(fig1, use_container_width=True)

            with c2:
                fig2 = px.histogram(
                    df_raw, x="cibil_score", color="loan_status", barmode="overlay",
                    title="CIBIL Score Distribution by Status (Note the 550 Cut-Off)",
                    color_discrete_map={'Approved': '#0284c7', 'Rejected': '#f43f5e'}
                )
                fig2.add_vline(x=550, line_dash="dash", line_color="#f59e0b", annotation_text="550 Cut-off")
                fig2.update_layout(template="plotly_white")
                st.plotly_chart(fig2, use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                fig3 = px.histogram(
                    df_raw, x="education", color="loan_status", barmode="group",
                    title="Education vs Loan Status",
                    color_discrete_map={'Approved': '#0284c7', 'Rejected': '#f43f5e'}
                )
                fig3.update_layout(template="plotly_white")
                st.plotly_chart(fig3, use_container_width=True)

            with c4:
                fig4 = px.histogram(
                    df_raw, x="self_employed", color="loan_status", barmode="group",
                    title="Self Employed vs Loan Status",
                    color_discrete_map={'Approved': '#0284c7', 'Rejected': '#f43f5e'}
                )
                fig4.update_layout(template="plotly_white")
                st.plotly_chart(fig4, use_container_width=True)

        with tab2:
            s1, s2 = st.columns(2)
            with s1:
                fig5 = px.scatter(
                    df_raw.sample(min(1200, len(df_raw)), random_state=42),
                    x="income_annum", y="loan_amount", color="loan_status",
                    title="Annual Income vs Loan Amount (Linear Capacity Pattern)",
                    color_discrete_map={'Approved': '#0284c7', 'Rejected': '#f43f5e'},
                    opacity=0.65
                )
                fig5.update_layout(template="plotly_white")
                st.plotly_chart(fig5, use_container_width=True)

            with s2:
                fig6 = px.box(
                    df_raw, x="loan_term", y="cibil_score", color="loan_status",
                    title="Loan Term vs CIBIL Score Distribution",
                    color_discrete_map={'Approved': '#0284c7', 'Rejected': '#f43f5e'}
                )
                fig6.update_layout(template="plotly_white")
                st.plotly_chart(fig6, use_container_width=True)

        with tab3:
            h1, h2 = st.columns([1.1, 1])
            with h1:
                # Grouped assets mean
                asset_cols = ['residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
                mean_assets = df_raw.groupby('loan_status')[asset_cols].mean().reset_index()
                melted = mean_assets.melt(id_vars='loan_status', var_name='Asset Type', value_name='Mean Value (INR)')
                fig7 = px.bar(
                    melted, x='Asset Type', y='Mean Value (INR)', color='loan_status', barmode='group',
                    title="Mean Asset Portfolios by Outcome",
                    color_discrete_map={'Approved': '#0284c7', 'Rejected': '#f43f5e'}
                )
                fig7.update_layout(template="plotly_white")
                st.plotly_chart(fig7, use_container_width=True)

            with h2:
                num_cols = [
                    'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
                    'cibil_score', 'residential_assets_value', 'commercial_assets_value',
                    'luxury_assets_value', 'bank_asset_value'
                ]
                corr = df_raw[num_cols].corr()
                fig8 = px.imshow(
                    corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="Blues",
                    title="Numerical Correlation Heatmap"
                )
                fig8.update_layout(template="plotly_white")
                st.plotly_chart(fig8, use_container_width=True)

        # EDA Report Download Option
        st.markdown("---")
        report_html_path = os.path.join(REPORTS_DIR, 'EDA_Report.html')
        if os.path.exists(report_html_path):
            with open(report_html_path, 'r', encoding='utf-8') as f:
                html_bytes = f.read().encode('utf-8')
            st.download_button(
                label="📄 Download Complete Interactive HTML EDA Report",
                data=html_bytes,
                file_name="Loan_Approval_EDA_Report.html",
                mime="text/html"
            )

# ----------------------------------------------------------------------
# PAGE 5: MODEL PERFORMANCE
# ----------------------------------------------------------------------
elif menu == "🎯 Model Performance":
    st.markdown("""
    <div class="fintech-hero">
        <span class="hero-badge">🧠 Evaluation Engine</span>
        <h1 class="hero-title">Model Performance & Benchmarks</h1>
        <p class="hero-subtitle">Rigorous test-set verification of the Logistic Regression pipeline and algorithmic benchmarks.</p>
    </div>
    """, unsafe_allow_html=True)

    if metadata is not None:
        p_mod = metadata['primary_model']

        # Metric Cards
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric("Test Accuracy", f"{p_mod['test_acc']*100:.2f}%", f"Train: {p_mod['train_acc']*100:.2f}%")
        with m_col2:
            st.metric("Precision", f"{p_mod['precision']:.4f}", "Positive: Approved")
        with m_col3:
            st.metric("Recall", f"{p_mod['recall']:.4f}", "True Positive Rate")
        with m_col4:
            st.metric("F1-Score", f"{p_mod['f1']:.4f}", "Harmonic Mean")
        with m_col5:
            st.metric("ROC-AUC", f"{p_mod['roc_auc']:.4f}", "Discrimination Area")

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # Charts Section
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            cm_img_path = os.path.join(REPORTS_DIR, 'confusion_matrix.png')
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, caption="Confusion Matrix on Held-Out Test Set (854 Applicants)", use_container_width=True)
        with c_col2:
            eval_img_path = os.path.join(REPORTS_DIR, 'model_evaluation.png')
            if os.path.exists(eval_img_path):
                st.image(eval_img_path, caption="ROC Curves & Model Comparison Benchmarks", use_container_width=True)

        st.markdown("---")
        feat_img_path = os.path.join(REPORTS_DIR, 'feature_analysis.png')
        if os.path.exists(feat_img_path):
            st.image(feat_img_path, caption="Logistic Regression Feature Coefficients & Impact Direction", use_container_width=True)

        st.markdown("---")
        st.markdown("<h3 style='color:#0f172a;'>Algorithmic Benchmark Comparison Table</h3>", unsafe_allow_html=True)
        benchmarks = [p_mod] + metadata.get('benchmarks', [])
        comp_data = []
        for b in benchmarks:
            comp_data.append({
                'Algorithm': b['model_name'],
                'Train Accuracy': f"{b['train_acc']*100:.2f}%",
                'Test Accuracy': f"{b['test_acc']*100:.2f}%",
                'Precision': f"{b['precision']:.4f}",
                'Recall': f"{b['recall']:.4f}",
                'F1-Score': f"{b['f1']:.4f}",
                'ROC-AUC': f"{b['roc_auc']:.4f}"
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

# ----------------------------------------------------------------------
# PAGE 6: ABOUT THE PROJECT
# ----------------------------------------------------------------------
elif menu == "ℹ️ About the Project":
    st.markdown("""
    <div class="fintech-hero">
        <span class="hero-badge">📘 Documentation</span>
        <h1 class="hero-title">About Loan Approval Intelligence</h1>
        <p class="hero-subtitle">System design, architectural pipeline, algorithms, and academic framework.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 1. Problem Statement
    In modern retail banking and consumer finance, assessing credit risk for mortgage and personal loan applicants is critical to minimizing Non-Performing Assets (NPAs) while ensuring equitable access to credit. Traditional manual credit underwriting processes are slow, labor-intensive, and prone to human inconsistency.

    ### 2. Project Objectives
    - Implement an automated, end-to-end Machine Learning pipeline using **Scikit-Learn Logistic Regression**.
    - Preprocess numerical and categorical inputs safely using **Pipelines** and **ColumnTransformer** to eliminate data leakage.
    - Validate performance on a held-out test dataset using robust classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
    - Provide an intuitive, interactive fintech dashboard for loan officers and retail applicants.

    ### 3. Dataset Characteristics
    - **Dataset File:** `loan_approval_dataset (1).csv`
    - **Sample Count:** 4,269 customer applications
    - **Features:** 11 explanatory variables (9 numerical, 2 categorical)
    - **Target Variable:** `loan_status` ('Approved' vs 'Rejected')

    ### 4. Technical Stack
    - **Core Languages & Libraries:** Python 3.14, Pandas, NumPy, Scikit-learn
    - **Model:** Logistic Regression Pipeline (with Standard Scaling & One-Hot Encoding)
    - **Benchmarking:** Decision Tree Classifier & Random Forest Classifier
    - **Data Visualizations:** Matplotlib, Seaborn, Plotly Express
    - **Interactive Web Application:** Streamlit

    ### 5. Architectural Workflow
    1. **Data Ingestion & Sanitization:** Strip column names and values of whitespace artifacts.
    2. **Stratified Splitting:** 80% Training (3,415 records) and 20% Testing (854 records).
    3. **Pipeline Fitting:** Fit imputers, scaler, and one-hot encoder strictly on training data.
    4. **Inference & Probabilistic Scoring:** Calculate calibrated decision log-odds and class probabilities.
    5. **Dashboard Deployment:** Streamlit interface with multi-page fintech design system.

    ### 6. Limitations & Future Scope
    - **Limitations:** Dataset does not contain macroeconomic indicators (inflation, repo rate) or behavioral repayment logs.
    - **Future Scope:** Integration of XGBoost / LightGBM, credit bureau API feeds, and explainable AI (SHAP / LIME) visualizers.

    ---
    *Disclaimer: This software is designed for educational and academic machine learning research purposes.*
    """)

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown("""
<div class="fintech-footer">
    <strong>Loan Approval Intelligence System</strong> &bull; Developed by Vrunda &bull; Powered by Scikit-Learn & Streamlit
</div>
""", unsafe_allow_html=True)
