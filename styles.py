# styles.py
import streamlit as st

def apply_custom_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #0e1117;
        color: #ffffff;
    }

    /* Glassmorphism sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(17, 25, 40, 0.75);
        backdrop-filter: blur(16px) saturate(180%);
        border-right: 1px solid rgba(255, 255, 255, 0.125);
    }

    /* Custom Card Style */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }

    .disease-title {
        color: #00d2ff;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .risk-score {
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        color: #888;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.4);
    }

    /* Inputs */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }

    /* Dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def render_result_card(disease, risk_score):
    st.markdown(f"""
    <div class="result-card">
        <div class="metric-label">Predicted Disease Outbreak</div>
        <div class="disease-title">{disease}</div>
        <hr>
        <div class="metric-label">Likelihood / Risk Score</div>
        <div class="risk-score">{risk_score}%</div>
    </div>
    """, unsafe_allow_html=True)
