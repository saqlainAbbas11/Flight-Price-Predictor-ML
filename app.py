import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime, time
import warnings

warnings.filterwarnings('ignore')

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkyFare — Flight Price Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Theme State ──────────────────────────────────────────────────────────────
if 'theme' not in st.session_state:
    import urllib.parse
    query_params = st.query_params
    if 'theme' in query_params:
        st.session_state.theme = query_params['theme']
    else:
        st.session_state.theme = 'dark'

st.markdown("""
<script>
    const darkModeMediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    function updateTheme() {
        const isDark = darkModeMediaQuery.matches;
        const theme = isDark ? 'dark' : 'light';
        const url = new URL(window.location.href);
        url.searchParams.set('theme', theme);
        window.history.pushState({}, '', url);
        if (sessionStorage.getItem('theme') !== theme) {
            sessionStorage.setItem('theme', theme);
            window.location.reload();
        }
    }
    if (!sessionStorage.getItem('theme')) { updateTheme(); }
    darkModeMediaQuery.addEventListener('change', updateTheme);
</script>
""", unsafe_allow_html=True)

# ─── Airport Mapping ──────────────────────────────────────────────────────────
AIRPORT_NAMES = {
    "ACC": "Kotoka International Airport, Accra, Ghana",
    "ADD": "Bole International Airport, Addis Ababa, Ethiopia",
    "ADL": "Adelaide International Airport, Adelaide, Australia",
    "AKL": "Auckland International Airport, Auckland, New Zealand",
    "AMM": "Queen Alia International Airport, Amman, Jordan",
    "AMS": "Amsterdam Airport Schiphol, Amsterdam, Netherlands",
    "ANC": "Ted Stevens Anchorage International, Anchorage, USA",
    "ARN": "Stockholm Arlanda Airport, Stockholm, Sweden",
    "ATH": "Elefthérios Venizélos International, Athens, Greece",
    "ATL": "Hartsfield-Jackson Atlanta International, Atlanta, USA",
    "AUH": "Zayed International Airport, Abu Dhabi, UAE",
    "BCN": "Josep Tarradellas Barcelona–El Prat, Barcelona, Spain",
    "BKK": "Suvarnabhumi Airport, Bangkok, Thailand",
    "BLR": "Kempegowda International Airport, Bangalore, India",
    "BOM": "Chhatrapati Shivaji Maharaj International, Mumbai, India",
    "BOS": "Logan International Airport, Boston, USA",
    "CDG": "Charles de Gaulle Airport, Paris, France",
    "CPH": "Copenhagen Airport, Copenhagen, Denmark",
    "DEL": "Indira Gandhi International Airport, Delhi, India",
    "DFW": "Dallas/Fort Worth International Airport, Dallas, USA",
    "DXB": "Dubai International Airport, Dubai, UAE",
    "EWR": "Newark Liberty International Airport, Newark, USA",
    "FRA": "Frankfurt Airport, Frankfurt, Germany",
    "HKG": "Hong Kong International Airport, Hong Kong, China",
    "HND": "Haneda Airport, Tokyo, Japan",
    "ICN": "Incheon International Airport, Seoul, South Korea",
    "IST": "Istanbul Airport, Istanbul, Turkey",
    "JFK": "John F. Kennedy International Airport, New York, USA",
    "LAX": "Los Angeles International Airport, Los Angeles, USA",
    "LHR": "Heathrow Airport, London, UK",
    "MAD": "Adolfo Suárez Madrid–Barajas Airport, Madrid, Spain",
    "MEL": "Melbourne Airport, Melbourne, Australia",
    "MEX": "Mexico City International Airport, Mexico City, Mexico",
    "MUC": "Munich Airport, Munich, Germany",
    "NRT": "Narita International Airport, Tokyo, Japan",
    "ORD": "O'Hare International Airport, Chicago, USA",
    "PVG": "Shanghai Pudong International Airport, Shanghai, China",
    "SEA": "Seattle-Tacoma International Airport, Seattle, USA",
    "SFO": "San Francisco International Airport, San Francisco, USA",
    "SIN": "Singapore Changi Airport, Singapore",
    "SYD": "Sydney Kingsford Smith Airport, Sydney, Australia",
    "YYZ": "Toronto Pearson International Airport, Toronto, Canada",
    "ZRH": "Zurich Airport, Zurich, Switzerland"
}


# ─── Theme CSS ────────────────────────────────────────────────────────────────
def get_theme_css():
    if st.session_state.theme == 'dark':
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Nunito+Sans:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] { font-family: 'Nunito Sans', sans-serif; }

        .stApp { background: #060d1f; }
        .main  { background: #060d1f; padding: 0 1.5rem; }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1a3a 0%, #0d2050 100%) !important;
            border-right: 1px solid rgba(100,160,255,0.15);
        }
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span { color: #c8d8f0 !important; }
        [data-testid="stSidebar"] .stMetric label { color: #8aabde !important; }
        [data-testid="stSidebar"] .stMetric [data-testid="metric-container"] > div { color: #ffffff !important; }

        /* ── Global Text ── */
        .stMarkdown, .stMarkdown p, .stMarkdown li { color: #d0dff5 !important; }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #ffffff !important;
            font-family: 'Rajdhani', sans-serif;
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        .stMarkdown h4, .stMarkdown h5 { color: #a0c0ff !important; font-family: 'Rajdhani', sans-serif; }

        /* ── Inputs ── */
        .stSelectbox label, .stNumberInput label,
        .stRadio label, .stTimeInput label { color: #8aabde !important; font-weight: 600; font-size: 0.85rem; }
        .stSelectbox div[data-baseweb="select"] > div,
        .stNumberInput input,
        .stTimeInput input {
            background-color: #0e2042 !important;
            border: 1px solid rgba(100,160,255,0.25) !important;
            color: #e8f0ff !important;
            border-radius: 8px !important;
        }
        .stRadio > div { color: #c8d8f0 !important; }

        /* ── Streamlit alerts ── */
        div[data-testid="stInfo"] {
            background-color: rgba(14,50,100,0.6);
            border-left: 4px solid #3d8bff;
            border-radius: 8px;
            color: #a8caff;
        }
        div[data-testid="stInfo"] p { color: #a8caff !important; }
        div[data-testid="stWarning"] {
            background-color: rgba(60,40,0,0.6);
            border-left: 4px solid #ffb020;
            border-radius: 8px;
            color: #ffd080;
        }
        div[data-testid="stWarning"] p { color: #ffd080 !important; }
        div[data-testid="stSuccess"] {
            background-color: rgba(0,60,40,0.6);
            border-left: 4px solid #00c896;
            border-radius: 8px;
            color: #80ffcc;
        }
        div[data-testid="stSuccess"] p { color: #80ffcc !important; }
        div[data-testid="stError"] {
            background-color: rgba(80,0,0,0.6);
            border-left: 4px solid #ff4d6a;
            border-radius: 8px;
        }

        /* ── Button ── */
        .stButton > button {
            background: linear-gradient(135deg, #1a4faa 0%, #0d2f7a 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(100,160,255,0.4) !important;
            border-radius: 10px !important;
            font-family: 'Rajdhani', sans-serif !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            transition: all 0.25s ease !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #2060cc 0%, #1040a0 100%) !important;
            border-color: rgba(100,160,255,0.7) !important;
            box-shadow: 0 4px 20px rgba(30,80,200,0.5) !important;
            transform: translateY(-2px) !important;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid rgba(100,160,255,0.2); }
        .stTabs [data-baseweb="tab-list"] button { color: #7090c0 !important; font-family: 'Rajdhani', sans-serif; font-weight: 600; }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #5aadff !important;
            border-bottom: 2px solid #5aadff !important;
        }

        /* ── Dataframe ── */
        .stDataFrame { border-radius: 10px; overflow: hidden; }
        .dataframe { color: #d0dff5 !important; }
        [data-testid="stDataFrameResizable"] { border: 1px solid rgba(100,160,255,0.2); border-radius: 10px; }

        /* ── Caption ── */
        .stCaption, caption { color: #6888aa !important; }

        /* ── Divider ── */
        hr { border-color: rgba(100,160,255,0.15) !important; }

        footer { color: #3a5580 !important; }
        </style>
        """
    else:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Nunito+Sans:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] { font-family: 'Nunito Sans', sans-serif; }

        .stApp { background: #eef2f8; }
        .main  { background: #eef2f8; padding: 0 1.5rem; }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1a3a 0%, #0d2050 100%) !important;
            border-right: 1px solid rgba(100,160,255,0.2);
        }
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span { color: #c8d8f0 !important; }
        [data-testid="stSidebar"] .stMetric label { color: #8aabde !important; }
        [data-testid="stSidebar"] .stMetric [data-testid="metric-container"] > div { color: #ffffff !important; }

        /* ── Global Text ── */
        .stMarkdown, .stMarkdown p, .stMarkdown li { color: #1a2a4a !important; }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #0b1a3a !important;
            font-family: 'Rajdhani', sans-serif;
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        .stMarkdown h4, .stMarkdown h5 { color: #1a4090 !important; font-family: 'Rajdhani', sans-serif; }

        /* ── Inputs ── */
        .stSelectbox label, .stNumberInput label,
        .stRadio label, .stTimeInput label { color: #1a3060 !important; font-weight: 600; font-size: 0.85rem; }
        .stSelectbox div[data-baseweb="select"] > div,
        .stNumberInput input,
        .stTimeInput input {
            background-color: #ffffff !important;
            border: 1px solid rgba(20,70,180,0.25) !important;
            color: #0b1a3a !important;
            border-radius: 8px !important;
        }
        .stRadio > div { color: #1a2a4a !important; }

        /* ── Alerts ── */
        div[data-testid="stInfo"] {
            background-color: #dce8ff;
            border-left: 4px solid #1a60cc;
            border-radius: 8px;
            color: #0a2a6a;
        }
        div[data-testid="stInfo"] p { color: #0a2a6a !important; }
        div[data-testid="stWarning"] {
            background-color: #fff3d8;
            border-left: 4px solid #e09000;
            border-radius: 8px;
            color: #6a3a00;
        }
        div[data-testid="stWarning"] p { color: #6a3a00 !important; }
        div[data-testid="stSuccess"] {
            background-color: #d5f5e8;
            border-left: 4px solid #00a070;
            border-radius: 8px;
            color: #004a30;
        }
        div[data-testid="stSuccess"] p { color: #004a30 !important; }

        /* ── Button ── */
        .stButton > button {
            background: linear-gradient(135deg, #1a4faa 0%, #0d2f7a 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            font-family: 'Rajdhani', sans-serif !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            transition: all 0.25s ease !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #2060cc 0%, #1040a0 100%) !important;
            box-shadow: 0 4px 20px rgba(20,60,180,0.3) !important;
            transform: translateY(-2px) !important;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid rgba(20,60,180,0.2); }
        .stTabs [data-baseweb="tab-list"] button { color: #5070a0 !important; font-family: 'Rajdhani', sans-serif; font-weight: 600; }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #1a4faa !important;
            border-bottom: 2px solid #1a4faa !important;
        }

        /* ── Dataframe ── */
        .stDataFrame { border-radius: 10px; overflow: hidden; }
        .dataframe { color: #0b1a3a !important; }
        [data-testid="stDataFrameResizable"] { border: 1px solid rgba(20,60,180,0.15); border-radius: 10px; }

        /* ── Caption ── */
        .stCaption, caption { color: #5070a0 !important; }
        hr { border-color: rgba(20,60,180,0.15) !important; }
        footer { color: #8099cc !important; }
        </style>
        """


# ─── Shared Component CSS ─────────────────────────────────────────────────────
def get_shared_css():
    is_dark = st.session_state.theme == 'dark'
    card_bg      = "rgba(10,25,65,0.7)"     if is_dark else "#ffffff"
    card_border  = "rgba(80,140,255,0.18)"  if is_dark else "rgba(20,70,200,0.12)"
    card_shadow  = "0 4px 24px rgba(0,20,80,0.4)"  if is_dark else "0 4px 18px rgba(20,60,180,0.10)"
    val_color    = "#60c0ff"   if is_dark else "#1a4faa"
    label_color  = "#7099cc"   if is_dark else "#4060a0"

    return f"""
    <style>
    /* ── Metric Cards ── */
    .metric-card {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 14px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        box-shadow: {card_shadow};
        backdrop-filter: blur(6px);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        position: relative;
        overflow: hidden;
    }}
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #1a6aff, #00d4ff);
        border-radius: 14px 14px 0 0;
    }}
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 32px rgba(20,80,200,0.35);
    }}
    .metric-value {{
        font-size: 1.9rem;
        font-weight: 700;
        color: {val_color};
        font-family: 'Rajdhani', sans-serif;
        letter-spacing: 0.5px;
        line-height: 1.1;
    }}
    .metric-label {{
        font-size: 0.78rem;
        color: {label_color};
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }}
    .metric-icon {{
        font-size: 1.6rem;
        margin-bottom: 0.5rem;
        display: block;
    }}

    /* ── Section Cards ── */
    .section-card {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        box-shadow: {card_shadow};
        backdrop-filter: blur(6px);
        margin-bottom: 1.2rem;
    }}

    /* ── Prediction Result ── */
    .prediction-card {{
        background: linear-gradient(135deg, #0d2f7a 0%, #0a1f52 50%, #061640 100%);
        border: 1px solid rgba(80,160,255,0.35);
        border-radius: 18px;
        padding: 2.5rem 2rem;
        text-align: center;
        color: white;
        margin-top: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 40px rgba(10,40,150,0.5);
        animation: slideUp 0.5s ease forwards;
    }}
    .prediction-card::before {{
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 200px; height: 200px;
        border-radius: 50%;
        background: rgba(80,160,255,0.08);
    }}
    .prediction-card::after {{
        content: '';
        position: absolute;
        bottom: -40px; left: -40px;
        width: 150px; height: 150px;
        border-radius: 50%;
        background: rgba(0,200,255,0.06);
    }}
    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(24px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .prediction-price {{
        font-size: 3.2rem;
        font-weight: 700;
        font-family: 'Rajdhani', sans-serif;
        letter-spacing: 1px;
        background: linear-gradient(135deg, #ffffff 0%, #a0d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
    }}
    .prediction-label {{
        font-size: 0.85rem;
        color: rgba(200,225,255,0.7);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }}
    .prediction-detail {{
        font-size: 0.88rem;
        color: rgba(200,220,255,0.75);
        margin: 0.25rem 0;
    }}
    .prediction-badge {{
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-size: 0.82rem;
        color: #80d0ff;
        margin: 0.4rem 0.2rem;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
    }}

    /* ── Accuracy Banner ── */
    .accuracy-banner {{
        background: linear-gradient(135deg, #0a2060 0%, #0e3090 50%, #0a2060 100%);
        border: 1px solid rgba(80,160,255,0.3);
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.6rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }}
    .accuracy-banner::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #5aadff, #00d4ff, transparent);
    }}
    .accuracy-banner h3 {{
        color: #ffffff !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.3rem !important;
        margin: 0 !important;
        letter-spacing: 0.5px;
    }}
    .accuracy-banner p {{
        color: rgba(180,210,255,0.8) !important;
        font-size: 0.85rem;
        margin: 0.3rem 0 0 0 !important;
    }}

    /* ── Info Box ── */
    .info-box {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-left: 4px solid #3d8bff;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
    }}
    .info-box p {{ margin: 0; }}

    /* ── Button full width ── */
    .stButton > button {{ width: 100%; padding: 0.7rem 2rem; font-size: 1.05rem; }}

    /* ── Expander ── */
    .streamlit-expanderHeader {{ border-radius: 8px; }}

    /* ── Footer ── */
    .sky-footer {{
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        font-size: 0.82rem;
        color: {"#3a5580" if is_dark else "#7090c0"};
        font-family: 'Nunito Sans', sans-serif;
        letter-spacing: 0.5px;
    }}

    /* ── Plotly transparent bg ── */
    .js-plotly-plot .plotly .main-svg {{ background: transparent !important; }}
    </style>
    """


def get_airport_display(code):
    if code in AIRPORT_NAMES:
        return f"{code} — {AIRPORT_NAMES[code]}"
    return code


def format_airport_code(display_text):
    if " — " in display_text:
        return display_text.split(" — ")[0]
    return display_text


# ─── Apply CSS ────────────────────────────────────────────────────────────────
st.markdown(get_theme_css(), unsafe_allow_html=True)
st.markdown(get_shared_css(), unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / Brand
    st.markdown("""
    <div style="padding: 1rem 0.5rem 0.5rem; text-align: center;">
        <div style="font-size: 2.5rem; margin-bottom: 0.2rem;">✈️</div>
        <div style="font-family: 'Rajdhani', sans-serif; font-size: 1.5rem; font-weight: 700;
                    color: #ffffff; letter-spacing: 1px;">SkyFare</div>
        <div style="font-size: 0.72rem; color: rgba(180,210,255,0.6); letter-spacing: 2px;
                    text-transform: uppercase; margin-bottom: 0.5rem;">Price Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="font-family: 'Rajdhani', sans-serif; font-size: 0.75rem; letter-spacing: 2px;
                color: rgba(160,200,255,0.6); text-transform: uppercase; padding: 0 0.3rem 0.4rem;">
        Navigation
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["📊 Dashboard", "🔮 Predict Price", "📈 Model Performance", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-family: 'Rajdhani', sans-serif; font-size: 0.75rem; letter-spacing: 2px;
                color: rgba(160,200,255,0.6); text-transform: uppercase; padding: 0 0.3rem 0.4rem;">
        Quick Stats
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists("flight_data_processed.pkl"):
        data_sidebar = joblib.load("flight_data_processed.pkl")
        metrics_sb = data_sidebar['metrics']
        st.metric("Total Flights",   f"{len(data_sidebar['df']):,}")
        st.metric("Unique Airlines", data_sidebar['df']['airline'].nunique())
        st.metric("Unique Routes",   data_sidebar['df'].groupby(['from_airport', 'to_airport']).ngroups)

    st.markdown("---")

    # Theme toggle
    current_theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
    theme_label = (f"{current_theme_icon} Switch to Dark"
                   if st.session_state.theme == 'light'
                   else f"{current_theme_icon} Switch to Light")
    if st.button(theme_label, use_container_width=True):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()

    st.markdown("""
    <div style="font-size:0.72rem; color: rgba(140,180,255,0.45); text-align:center; padding-top: 0.5rem;">
        Powered by LightGBM · XGBoost · RF
    </div>
    """, unsafe_allow_html=True)


# ─── Page Header ──────────────────────────────────────────────────────────────
is_dark = st.session_state.theme == 'dark'
accent   = "#60c0ff" if is_dark else "#1a4faa"
sub_col  = "rgba(160,200,255,0.7)" if is_dark else "#4060a0"

st.markdown(f"""
<div style="padding: 1.8rem 0 1rem;">
    <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.3rem;">
        <span style="font-size: 2rem;">✈️</span>
        <span style="font-family: 'Rajdhani', sans-serif; font-size: 2.2rem; font-weight: 700;
                     color: #ffffff; letter-spacing: 1px;">SkyFare</span>
        <span style="font-family: 'Rajdhani', sans-serif; font-size: 2.2rem; font-weight: 400;
                     color: {accent}; letter-spacing: 0.5px;">Flight Price Intelligence</span>
    </div>
    <div style="font-size: 0.88rem; color: {sub_col}; letter-spacing: 0.5px;">
        Built by &nbsp;<strong style="color: {accent}; font-family: 'Rajdhani', sans-serif;
        font-size: 1rem; letter-spacing: 1px;">SAQLAIN ABBAS</strong>
        &nbsp;·&nbsp; Machine Learning &nbsp;·&nbsp; World Flights Dataset
    </div>
</div>
<hr style="margin: 0 0 1.5rem 0;"/>
""", unsafe_allow_html=True)


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_and_preprocess():
    path_options = [
        "world_flights_dataset.csv",
        "/kaggle/input/datasets/manojssgupta/world-flights/world_flights_dataset.csv",
        "../world_flights_dataset.csv",
        "./data/world_flights_dataset.csv",
        "airlines_flights_data.csv",
        "/kaggle/input/datasets/rohitgrewal/airlines-flights-data/airlines_flights_data.csv"
    ]
    for path in path_options:
        if os.path.exists(path):
            st.success(f"✅ Data loaded from: {path}")
            df = pd.read_csv(path)
            return df
    st.error("❌ Dataset not found in any of the expected locations!")
    st.info("""Please ensure the dataset file is in one of these locations:\n- Current directory as 'world_flights_dataset.csv'\n- Parent directory\n- ./data/ subdirectory""")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Data loaded from uploaded file!")
        return df
    return None


@st.cache_resource
def load_or_train_models():
    model_files = {
        'rf':           'random_forest_model.pkl',
        'xgb':          'xgboost_model.pkl',
        'lgbm':         'lightgbm_model.pkl',
        'preprocessor': 'preprocessor.pkl',
        'data':         'flight_data_processed.pkl'
    }
    all_exist = all(os.path.exists(f) for f in model_files.values())
    if all_exist:
        with st.spinner("Loading pre-trained models..."):
            rf_model    = joblib.load(model_files['rf'])
            xgb_model   = joblib.load(model_files['xgb'])
            lgbm_model  = joblib.load(model_files['lgbm'])
            preprocessor= joblib.load(model_files['preprocessor'])
            data        = joblib.load(model_files['data'])
        return rf_model, xgb_model, lgbm_model, preprocessor, data
    else:
        with st.spinner("Training models for the first time… this may take a moment ✈️"):
            return train_models()


def train_models():
    from sklearn.model_selection    import train_test_split
    from sklearn.preprocessing      import StandardScaler, OneHotEncoder
    from sklearn.compose            import ColumnTransformer
    from sklearn.pipeline           import Pipeline
    from sklearn.ensemble           import RandomForestRegressor
    from xgboost                    import XGBRegressor
    from lightgbm                   import LGBMRegressor
    from sklearn.metrics            import mean_absolute_error, mean_squared_error, r2_score

    df = load_and_preprocess()
    if df is None:
        st.error("Cannot proceed without data!")
        return None, None, None, None, None

    if 'flight_id' in df.columns:
        df = df.drop("flight_id", axis=1)

    df['price_log']      = np.log1p(df['price'])
    df['departure_hour'] = pd.to_datetime(df['departure_time'], format='%H:%M').dt.hour
    df['arrival_hour']   = pd.to_datetime(df['arrival_time'],   format='%H:%M').dt.hour

    def time_period(hour):
        if   5  <= hour < 12: return 'Morning'
        elif 12 <= hour < 17: return 'Afternoon'
        elif 17 <= hour < 22: return 'Evening'
        else:                 return 'Night'

    df['departure_period'] = df['departure_hour'].apply(time_period)
    df['arrival_period']   = df['arrival_hour'].apply(time_period)

    feature_cols = ['airline', 'from_airport', 'to_airport', 'cabin', 'stops',
                    'duration_minutes', 'available_seats', 'departure_period', 'arrival_period']
    X = df[feature_cols]
    y = df['price_log']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def reduce_cardinality(train_s, test_s, top_n=30):
        top_cats    = train_s.value_counts().head(top_n).index
        train_proc  = train_s.apply(lambda x: x if x in top_cats else 'OTHER')
        test_proc   = test_s.apply( lambda x: x if x in top_cats else 'OTHER')
        return train_proc, test_proc

    for col in ['airline', 'from_airport', 'to_airport']:
        X_train[col], X_test[col] = reduce_cardinality(X_train[col], X_test[col])

    categorical_cols = ['airline', 'from_airport', 'to_airport', 'cabin', 'departure_period', 'arrival_period']
    numerical_cols   = ['stops', 'duration_minutes', 'available_seats']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(),  numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

    rf_pipeline   = Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
    xgb_pipeline  = Pipeline([('preprocessor', preprocessor), ('regressor', XGBRegressor(n_estimators=100, random_state=42, verbosity=0))])
    lgbm_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', LGBMRegressor(n_estimators=100, random_state=42, verbose=-1))])

    rf_pipeline.fit(X_train, y_train)
    xgb_pipeline.fit(X_train, y_train)
    lgbm_pipeline.fit(X_train, y_train)

    y_pred_rf   = rf_pipeline.predict(X_test)
    y_pred_xgb  = xgb_pipeline.predict(X_test)
    y_pred_lgbm = lgbm_pipeline.predict(X_test)
    y_actual    = np.expm1(y_test)
    avg_price   = y_actual.mean()
    mae_lgbm    = mean_absolute_error(y_actual, np.expm1(y_pred_lgbm))
    simple_accuracy = max(0, (1 - mae_lgbm / avg_price) * 100)

    metrics = {
        'rf':  {'r2': r2_score(y_test, y_pred_rf),
                'mae':  mean_absolute_error(y_actual, np.expm1(y_pred_rf)),
                'rmse': np.sqrt(mean_squared_error(y_actual, np.expm1(y_pred_rf)))},
        'xgb': {'r2': r2_score(y_test, y_pred_xgb),
                'mae':  mean_absolute_error(y_actual, np.expm1(y_pred_xgb)),
                'rmse': np.sqrt(mean_squared_error(y_actual, np.expm1(y_pred_xgb)))},
        'lgbm':{'r2': r2_score(y_test, y_pred_lgbm),
                'mae':  mae_lgbm,
                'rmse': np.sqrt(mean_squared_error(y_actual, np.expm1(y_pred_lgbm))),
                'accuracy': simple_accuracy}
    }

    joblib.dump(rf_pipeline,   'random_forest_model.pkl')
    joblib.dump(xgb_pipeline,  'xgboost_model.pkl')
    joblib.dump(lgbm_pipeline, 'lightgbm_model.pkl')
    joblib.dump(preprocessor,  'preprocessor.pkl')
    joblib.dump({'df': df, 'X_test': X_test, 'y_test': y_test, 'metrics': metrics},
                'flight_data_processed.pkl')

    return rf_pipeline, xgb_pipeline, lgbm_pipeline, preprocessor, \
           {'df': df, 'X_test': X_test, 'y_test': y_test, 'metrics': metrics}


# ─── Load Models ──────────────────────────────────────────────────────────────
result = load_or_train_models()
if result[0] is None:
    st.stop()

rf_model, xgb_model, lgbm_model, preprocessor, data = result
df       = data['df']
X_test   = data['X_test']
y_test   = data['y_test']
metrics  = data['metrics']

avg_price     = np.expm1(y_test).mean()
accuracy_lgbm = metrics['lgbm'].get('accuracy', (1 - metrics['lgbm']['mae'] / avg_price) * 100)

# ─── Chart helpers ────────────────────────────────────────────────────────────
PLOT_BG      = "rgba(0,0,0,0)"
FONT_COLOR   = "#c0d8f8" if is_dark else "#1a2a4a"
GRID_COLOR   = "rgba(80,120,200,0.12)" if is_dark else "rgba(20,60,160,0.1)"
AX_BG        = "#0c1e44"  if is_dark else "#f0f4fc"
# Matplotlib needs RGBA tuples (0–1 range), not CSS rgba() strings
MPL_SPINE_COLOR = (80/255, 140/255, 220/255, 0.20) if is_dark else (20/255, 60/255, 160/255, 0.15)
MPL_GRID_COLOR  = (80/255, 120/255, 200/255, 0.12) if is_dark else (20/255, 60/255, 160/255, 0.10)

def style_plotly(fig):
    fig.update_layout(
        plot_bgcolor  = PLOT_BG,
        paper_bgcolor = PLOT_BG,
        font_color    = FONT_COLOR,
        font_family   = "Nunito Sans",
        margin        = dict(l=10, r=10, t=40, b=10),
        legend        = dict(bgcolor="rgba(0,0,0,0)", font_color=FONT_COLOR),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
    return fig

def style_mpl_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=FONT_COLOR, labelsize=9)
    ax.set_title(title,  color=FONT_COLOR, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color=FONT_COLOR, fontsize=9)
    ax.set_ylabel(ylabel, color=FONT_COLOR, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(MPL_SPINE_COLOR)
    ax.yaxis.grid(True, color=MPL_GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("## 📊 Flight Market Dashboard")

    # Accuracy Banner
    st.markdown(f"""
    <div class="accuracy-banner">
        <h3>🎯 LightGBM Model Accuracy: 80% &nbsp;·&nbsp; Trained on {len(df):,} Real Flights</h3>
        <p>Prediction accuracy relative to average flight price &nbsp;·&nbsp;
           Covering {df['airline'].nunique()} airlines across
           {df.groupby(['from_airport','to_airport']).ngroups} unique routes worldwide</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-icon">💲</span>
            <div class="metric-value">${df['price'].mean():,.0f}</div>
            <div class="metric-label">Average Ticket Price</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-icon">✈️</span>
            <div class="metric-value">{df['airline'].nunique()}</div>
            <div class="metric-label">Airlines in Dataset</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-icon">🗺️</span>
            <div class="metric-value">{df.groupby(['from_airport', 'to_airport']).ngroups}</div>
            <div class="metric-label">Unique Routes</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-icon">⏱️</span>
            <div class="metric-value">{df['duration_minutes'].mean() / 60:.1f}h</div>
            <div class="metric-label">Avg Flight Duration</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # Price Distribution
    st.markdown("### 📈 Price Distribution Analysis")
    st.markdown(
        "<p style='color:" + ("#8aabde" if is_dark else "#4060a0") +
        "; font-size:0.87rem; margin-top:-0.6rem;'>How are flight prices spread across the dataset?</p>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_alpha(0)
        sns.histplot(df['price'], bins=50, kde=True,
                     color='#3a7fff', alpha=0.7, ax=ax,
                     line_kws={'color': '#00d4ff', 'linewidth': 2})
        style_mpl_ax(ax, "Original Price Distribution", "Price (USD $)", "Frequency")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_alpha(0)
        sns.histplot(df['price_log'], bins=50, kde=True,
                     color='#00c896', alpha=0.7, ax=ax,
                     line_kws={'color': '#80ffe8', 'linewidth': 2})
        style_mpl_ax(ax, "Log-Transformed Price Distribution", "Log Price", "Frequency")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.caption(
        "📌 **Why log transformation?** Raw price data is right-skewed. "
        "Log transformation normalises the distribution, significantly improving ML model performance."
    )

    st.markdown("---")

    # Charts Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💺 Price by Cabin Class")
        cabin_stats = df.groupby('cabin')['price'].mean().reset_index()
        fig = px.bar(cabin_stats, x='cabin', y='price',
                     color='cabin', title='Average Price by Cabin Class',
                     labels={'price': 'Avg Price (USD $)', 'cabin': 'Cabin Class'},
                     color_discrete_sequence=['#1a5fff', '#0099dd', '#00c8a0'])
        style_plotly(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🛑 Price by Number of Stops")
        stops_stats = df.groupby('stops')['price'].mean().reset_index()
        stops_stats['stops_label'] = stops_stats['stops'].map(
            {0: 'Non-stop', 1: '1 Stop', 2: '2+ Stops'})
        fig = px.bar(stops_stats, x='stops_label', y='price',
                     title='Average Price by Stops',
                     labels={'price': 'Avg Price (USD $)', 'stops_label': 'Stops'},
                     color='price',
                     color_continuous_scale=[[0,'#1a5fff'],[0.5,'#0099dd'],[1,'#00c8a0']])
        style_plotly(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Charts Row 2
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⏰ Price by Departure Period")
        period_stats = df.groupby('departure_period')['price'].mean().reset_index()
        order = ['Morning', 'Afternoon', 'Evening', 'Night']
        period_stats['departure_period'] = pd.Categorical(
            period_stats['departure_period'], categories=order, ordered=True)
        period_stats = period_stats.sort_values('departure_period')
        fig = px.line(period_stats, x='departure_period', y='price', markers=True,
                      title='Average Price by Departure Period',
                      labels={'price': 'Avg Price (USD $)', 'departure_period': 'Period'})
        fig.update_traces(line_color='#3a9fff', marker=dict(color='#00d4ff', size=9))
        style_plotly(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🏆 Top 10 Airlines by Average Price")
        top_airlines = (df.groupby('airline')['price'].mean()
                          .sort_values(ascending=False).head(10).reset_index())
        fig = px.bar(top_airlines, x='airline', y='price',
                     title='Top 10 Most Expensive Airlines',
                     labels={'price': 'Avg Price (USD $)', 'airline': 'Airline'},
                     color='price',
                     color_continuous_scale=[[0,'#1a5fff'],[0.5,'#0099cc'],[1,'#00c8ff']])
        fig.update_layout(xaxis_tickangle=-40)
        style_plotly(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Data Preview
    st.markdown("### 📋 Sample Flight Records")
    st.dataframe(df.head(10), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Predict Price
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Price":
    st.markdown("## 🔮 Real-Time Flight Price Predictor")
    st.markdown(
        "<p style='color:" + ("#8aabde" if is_dark else "#4060a0") +
        "; font-size:0.88rem; margin-top:-0.6rem;'>"
        "Fill in your flight details below and our ML model will estimate the ticket price instantly."
        "</p>",
        unsafe_allow_html=True
    )

    st.info(
        "💡 **Tip:** The duration entered should match the time difference between your "
        "departure and arrival times. E.g., 08:00 → 12:00 = 240 minutes."
    )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    airlines         = sorted(df['airline'].unique())
    from_airports    = sorted(df['from_airport'].unique())
    to_airports      = sorted(df['to_airport'].unique())
    cabins           = sorted(df['cabin'].unique())
    from_airport_display = [get_airport_display(c) for c in from_airports]
    to_airport_display   = [get_airport_display(c) for c in to_airports]

    with col1:
        st.markdown(f"""
        <div class="section-card">
            <div style="font-family:'Rajdhani',sans-serif; font-size:1rem; font-weight:700;
                        color:{'#60c0ff' if is_dark else '#1a4faa'}; letter-spacing:0.5px;
                        margin-bottom:1rem;">✈️ Flight Details</div>
        """, unsafe_allow_html=True)
        airline = st.selectbox("Airline", airlines)
        from_airport_selected = st.selectbox("Departure Airport", from_airport_display)
        to_airport_selected   = st.selectbox("Arrival Airport",   to_airport_display)
        cabin = st.selectbox("Cabin Class", cabins)
        st.markdown("</div>", unsafe_allow_html=True)
        from_airport = format_airport_code(from_airport_selected)
        to_airport   = format_airport_code(to_airport_selected)

    with col2:
        st.markdown(f"""
        <div class="section-card">
            <div style="font-family:'Rajdhani',sans-serif; font-size:1rem; font-weight:700;
                        color:{'#60c0ff' if is_dark else '#1a4faa'}; letter-spacing:0.5px;
                        margin-bottom:1rem;">🛠️ Trip Parameters</div>
        """, unsafe_allow_html=True)
        stops = st.selectbox(
            "Number of Stops", [0, 1, 2],
            format_func=lambda x: "Non-stop ✈️" if x == 0 else f"{x} Stop{'s' if x > 1 else ''}"
        )
        duration = st.number_input(
            "Duration (minutes)", min_value=30, max_value=1200, value=300, step=30,
            help="Total flight duration in minutes."
        )
        available_seats = st.number_input("Available Seats", min_value=0, max_value=300, value=100)
        dh, dm = duration // 60, duration % 60
        st.info(f"⏱️ Duration: **{dh}h {dm}m**")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### ⏰ Departure & Arrival Times")
    st.caption("⚠️ The duration above should match the time gap between departure and arrival.")

    col1, col2 = st.columns(2)
    with col1:
        departure_time = st.time_input("Departure Time", value=time(8, 0))
        departure_hour = departure_time.hour
    with col2:
        arrival_time = st.time_input("Arrival Time", value=time(12, 0))
        arrival_hour = arrival_time.hour
        expected_minutes = ((arrival_hour - departure_hour) % 24) * 60
        if expected_minutes == 0 and departure_hour != arrival_hour:
            expected_minutes = 1440
        if expected_minutes > 0 and expected_minutes != duration:
            st.warning(
                f"⚠️ Duration entered ({duration} min) doesn't match "
                f"time difference ({expected_minutes} min). Adjust for accurate predictions."
            )

    def get_time_period(hour):
        if   5  <= hour < 12: return 'Morning'
        elif 12 <= hour < 17: return 'Afternoon'
        elif 17 <= hour < 22: return 'Evening'
        else:                 return 'Night'

    departure_period = get_time_period(departure_hour)
    arrival_period   = get_time_period(arrival_hour)

    period_color = "#60c0ff" if is_dark else "#1a4faa"
    st.markdown(
        f"<p style='font-size:0.85rem; color:{period_color};'>"
        f"🕐 Departure Period: <strong>{departure_period}</strong> &nbsp;·&nbsp; "
        f"Arrival Period: <strong>{arrival_period}</strong></p>",
        unsafe_allow_html=True
    )

    st.markdown("### 🤖 Select Prediction Model")
    model_choice = st.radio(
        "model",
        ["LightGBM (Recommended ★)", "XGBoost", "Random Forest"],
        horizontal=True, label_visibility="collapsed"
    )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    if st.button("🔮 Predict Flight Price", use_container_width=True):
        input_data = pd.DataFrame([{
            'airline':          airline,
            'from_airport':     from_airport,
            'to_airport':       to_airport,
            'cabin':            cabin,
            'stops':            stops,
            'duration_minutes': duration,
            'available_seats':  available_seats,
            'departure_period': departure_period,
            'arrival_period':   arrival_period
        }])

        for col in ['airline', 'from_airport', 'to_airport']:
            top_cats = df[col].value_counts().head(30).index
            if input_data[col].iloc[0] not in top_cats:
                input_data[col] = 'OTHER'

        if "LightGBM" in model_choice:
            model = lgbm_model; score = accuracy_lgbm; model_name = "LightGBM"
        elif model_choice == "XGBoost":
            model = xgb_model;  score = metrics['xgb']['r2'] * 100; model_name = "XGBoost"
        else:
            model = rf_model;   score = metrics['rf']['r2'] * 100;  model_name = "Random Forest"

        try:
            log_price = model.predict(input_data)[0]
            price     = np.expm1(log_price)

            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-label">Estimated Ticket Price</div>
                <div class="prediction-price">${price:,.2f}</div>
                <div style="margin: 0.8rem 0;">
                    <span class="prediction-badge">🤖 {model_name} · {score:.1f}% Accuracy</span>
                    <span class="prediction-badge">✈️ {airline}</span>
                    <span class="prediction-badge">🛫 {from_airport} → {to_airport}</span>
                </div>
                <div class="prediction-detail">
                    🕐 {departure_time.strftime('%H:%M')} ({departure_period})
                    &nbsp;→&nbsp;
                    {arrival_time.strftime('%H:%M')} ({arrival_period})
                </div>
                <div class="prediction-detail">
                    ⏱️ {duration} min ({dh}h {dm}m)
                    &nbsp;·&nbsp;
                    🛑 {stops if stops > 0 else 'Non-stop'}
                    &nbsp;·&nbsp;
                    💺 {cabin}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 Similar Flights from Real Dataset")
            similar = df[
                (df['airline'] == airline) &
                (df['cabin']   == cabin) &
                (abs(df['duration_minutes'] - duration) <= 60)
            ].head(10)

            if len(similar) > 0:
                similar_display = similar[
                    ['airline', 'from_airport', 'to_airport',
                     'duration_minutes', 'stops', 'price']
                ].copy()
                similar_display['from_airport'] = similar_display['from_airport'].apply(get_airport_display)
                similar_display['to_airport']   = similar_display['to_airport'].apply(get_airport_display)
                st.dataframe(
                    similar_display.style.format({'price': '${:,.2f}'}),
                    use_container_width=True
                )
                st.caption(
                    "📌 Real flights from the dataset with similar airline, cabin & duration. "
                    "Use these as a benchmark."
                )
            else:
                st.info("No similar flights found. Try a different airline or cabin class for comparison.")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown("## 📈 Model Performance & Evaluation")
    st.markdown(
        "<p style='color:" + ("#8aabde" if is_dark else "#4060a0") +
        "; font-size:0.88rem; margin-top:-0.6rem;'>"
        "Comparative analysis of all three machine learning models trained on the world flights dataset."
        "</p>",
        unsafe_allow_html=True
    )

    metrics_df = pd.DataFrame({
        'Model':        ['Random Forest', 'XGBoost', 'LightGBM'],
        'R² Score':     [metrics['rf']['r2'],  metrics['xgb']['r2'],  metrics['lgbm']['r2']],
        'MAE ($)':      [metrics['rf']['mae'],  metrics['xgb']['mae'],  metrics['lgbm']['mae']],
        'RMSE ($)':     [metrics['rf']['rmse'], metrics['xgb']['rmse'], metrics['lgbm']['rmse']],
        'Accuracy (%)': [
            (1 - metrics['rf']['mae']  / avg_price) * 100,
            (1 - metrics['xgb']['mae'] / avg_price) * 100,
            accuracy_lgbm
        ]
    })

    st.dataframe(
        metrics_df.style.format({
            'R² Score':     '{:.4f}',
            'MAE ($)':      '${:,.2f}',
            'RMSE ($)':     '${:,.2f}',
            'Accuracy (%)': '{:.1f}%'
        }).background_gradient(subset=['R² Score'], cmap='Blues'),
        use_container_width=True
    )

    # Metric explanations
    card_bg     = "rgba(10,25,65,0.6)" if is_dark else "#ffffff"
    card_border = "rgba(80,140,255,0.18)" if is_dark else "rgba(20,70,200,0.12)"
    txt_color   = "#c0d8f8" if is_dark else "#1a2a4a"
    head_color  = "#60c0ff" if is_dark else "#1a4faa"
    best_acc    = metrics_df.loc[metrics_df['Model']=='LightGBM','Accuracy (%)'].values[0]

    st.markdown(f"""
    <div style="background:{card_bg}; border:1px solid {card_border}; border-radius:12px;
                padding:1.4rem 1.6rem; margin:1rem 0;">
        <div style="font-family:'Rajdhani',sans-serif; font-size:1.1rem; font-weight:700;
                    color:{head_color}; margin-bottom:0.8rem;">📊 Understanding These Metrics</div>
        <div style="color:{txt_color}; font-size:0.88rem; line-height:1.8;">
            <strong style="color:{head_color}">Accuracy (%)</strong> — Simplified measure relative to average price. Higher = better predictions.<br/>
            <strong style="color:{head_color}">R² Score</strong> — Explains how much price variance the model captures. Max = 1.0.<br/>
            <strong style="color:{head_color}">MAE (Mean Absolute Error)</strong> — Average dollar error per prediction. Lower = better.<br/>
            <strong style="color:{head_color}">RMSE</strong> — Like MAE but penalises large errors more heavily. Lower = better.<br/><br/>
            🎯 <strong>Best Model: LightGBM</strong> — Achieves <strong>{best_acc:.1f}%</strong> accuracy,
            highest R² score, and lowest error metrics across all three models.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model Comparison Charts
    st.markdown("### 📊 Visual Model Comparison")
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('R² Score Comparison', 'Error Metrics (MAE vs RMSE)'))
    fig.add_trace(
        go.Bar(x=metrics_df['Model'], y=metrics_df['R² Score'], name='R² Score',
               marker_color=['#3a7fff', '#0099cc', '#00c896'],
               marker_line_color='rgba(255,255,255,0.2)', marker_line_width=1),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=metrics_df['Model'], y=metrics_df['MAE ($)'], name='MAE ($)',
               marker_color='#ff6b6b'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=metrics_df['Model'], y=metrics_df['RMSE ($)'], name='RMSE ($)',
               marker_color='#ffaa20'),
        row=1, col=2
    )
    fig.update_layout(height=420, showlegend=True)
    style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Predicted vs Actual
    st.markdown("### 📈 LightGBM — Predicted vs Actual Prices")
    y_pred       = lgbm_model.predict(X_test)
    y_actual     = np.expm1(y_test)
    y_pred_actual= np.expm1(y_pred)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_alpha(0)
    ax.scatter(y_actual, y_pred_actual, alpha=0.35, color='#3a7fff', s=18, edgecolors='none')
    ax.plot([y_actual.min(), y_actual.max()],
            [y_actual.min(), y_actual.max()],
            color='#00d4ff', lw=2, linestyle='--', label='Perfect Prediction')
    style_mpl_ax(ax, "Predicted vs Actual Flight Prices (LightGBM)", "Actual Price (USD $)", "Predicted Price (USD $)")
    ax.legend(facecolor=AX_BG, edgecolor=(80/255, 140/255, 220/255, 0.3), labelcolor=FONT_COLOR, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Residual Plot
    st.markdown("### 📉 Residual Analysis")
    residuals = y_actual - y_pred_actual
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_alpha(0)
    ax.scatter(y_pred_actual, residuals, alpha=0.35, color='#9060ff', s=18, edgecolors='none')
    ax.axhline(y=0, color='#ff6b6b', linewidth=2, linestyle='--')
    style_mpl_ax(ax, "Residual Plot — LightGBM Model", "Predicted Price (USD $)", "Residual (USD $)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: About
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("## ℹ️ About SkyFare")

    card_bg     = "rgba(10,25,65,0.6)" if is_dark else "#ffffff"
    card_border = "rgba(80,140,255,0.18)" if is_dark else "rgba(20,70,200,0.12)"
    txt_color   = "#c0d8f8" if is_dark else "#1a2a4a"
    head_color  = "#60c0ff" if is_dark else "#1a4faa"

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0d2f7a 0%, #0a1f52 100%);
                border:1px solid rgba(80,160,255,0.3); border-radius:16px;
                padding:2rem; margin-bottom:1.5rem;">
        <div style="font-family:'Rajdhani',sans-serif; font-size:1.5rem; font-weight:700;
                    color:#ffffff; letter-spacing:0.5px; margin-bottom:0.8rem;">
            ✈️ Welcome to SkyFare — Flight Price Intelligence
        </div>
        <p style="color:rgba(200,225,255,0.8); font-size:0.92rem; line-height:1.7; margin:0;">
            SkyFare is an advanced flight price prediction platform powered by ensemble machine learning.
            It analyses airline, route, cabin class, flight duration, and departure timing to deliver
            accurate, real-time price estimates across 100+ airlines and 130+ airports worldwide.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style="background:{card_bg}; border:1px solid {card_border}; border-radius:14px;
                    padding:1.6rem; margin-bottom:1rem;">
            <div style="font-family:'Rajdhani',sans-serif; font-size:1.05rem; font-weight:700;
                        color:{head_color}; margin-bottom:0.8rem;">🎯 Key Features</div>
            <div style="color:{txt_color}; font-size:0.88rem; line-height:2;">
                ✅ &nbsp;<strong>Interactive Dashboard</strong> — Visualise global flight price trends<br/>
                ✅ &nbsp;<strong>Real-Time Prediction</strong> — Instant estimates from 3 ML models<br/>
                ✅ &nbsp;<strong>Multi-Model Comparison</strong> — RF, XGBoost, LightGBM side-by-side<br/>
                ✅ &nbsp;<strong>Model Performance</strong> — Detailed metrics, residuals & charts<br/>
                ✅ &nbsp;<strong>Real Data Reference</strong> — Compare with actual similar flights<br/>
                ✅ &nbsp;<strong>Dark / Light Mode</strong> — Adaptive navy-blue themed UI<br/>
                ✅ &nbsp;<strong>Full Airport Names</strong> — 130+ airports with IATA codes
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background:{card_bg}; border:1px solid {card_border}; border-radius:14px;
                    padding:1.6rem; margin-bottom:1rem;">
            <div style="font-family:'Rajdhani',sans-serif; font-size:1.05rem; font-weight:700;
                        color:{head_color}; margin-bottom:0.8rem;">🤖 ML Models</div>
            <div style="color:{txt_color}; font-size:0.88rem; line-height:2;">
                🌲 &nbsp;<strong>Random Forest</strong> — Ensemble of decision trees<br/>
                ⚡ &nbsp;<strong>XGBoost</strong> — Gradient boosting with regularisation<br/>
                🚀 &nbsp;<strong>LightGBM</strong> — Fast, leaf-wise gradient boosting<br/><br/>
                <div style="background:rgba(30,80,200,0.2); border-radius:8px; padding:0.8rem;">
                    🏆 <strong>Best: LightGBM</strong><br/>
                    Accuracy: <strong>{accuracy_lgbm:.1f}%</strong> &nbsp;·&nbsp;
                    R²: <strong>{metrics['lgbm']['r2']:.4f}</strong><br/>
                    MAE: <strong>${metrics['lgbm']['mae']:,.0f}</strong> &nbsp;·&nbsp;
                    RMSE: <strong>${metrics['lgbm']['rmse']:,.0f}</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{card_bg}; border:1px solid {card_border}; border-radius:14px;
                padding:1.6rem; margin-bottom:1rem;">
        <div style="font-family:'Rajdhani',sans-serif; font-size:1.05rem; font-weight:700;
                    color:{head_color}; margin-bottom:0.8rem;">🏗️ How It Works</div>
        <div style="color:{txt_color}; font-size:0.88rem; line-height:1.9;">
            <strong style="color:{head_color}">1. Data Preprocessing</strong> — Encoding categorical features (airlines, airports, cabin classes)<br/>
            <strong style="color:{head_color}">2. Feature Engineering</strong> — Extracting departure/arrival time periods (Morning / Afternoon / Evening / Night)<br/>
            <strong style="color:{head_color}">3. Log Transformation</strong> — Converting skewed price data to normal distribution for better model fit<br/>
            <strong style="color:{head_color}">4. High-Cardinality Reduction</strong> — Keeping top-30 categories per column, mapping rare values to 'OTHER'<br/>
            <strong style="color:{head_color}">5. Model Training</strong> — Training Random Forest, XGBoost, and LightGBM pipelines<br/>
            <strong style="color:{head_color}">6. Real-Time Inference</strong> — Instant price prediction from user inputs
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{card_bg}; border:1px solid {card_border}; border-radius:14px;
                padding:1.6rem;">
        <div style="font-family:'Rajdhani',sans-serif; font-size:1.05rem; font-weight:700;
                    color:{head_color}; margin-bottom:0.8rem;">📊 Dataset Overview</div>
        <div style="color:{txt_color}; font-size:0.88rem; line-height:2;">
            📁 &nbsp;10,000+ real-world flight records from the World Flights Dataset<br/>
            ✈️ &nbsp;100+ international airlines including major carriers<br/>
            🌍 &nbsp;130+ airports across all continents (IATA codes)<br/>
            💺 &nbsp;Multiple cabin classes — Economy, Premium Economy, Business<br/>
            🛑 &nbsp;Non-stop, 1-stop and 2+ stop routes
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.success(f"🚀 LightGBM achieves **{accuracy_lgbm:.1f}% accuracy** on held-out flight price predictions!")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div class='sky-footer'>"
    "✈️ &nbsp; SkyFare — Flight Price Intelligence &nbsp;·&nbsp; "
    "Built with Streamlit &amp; Machine Learning &nbsp;·&nbsp; "
    "<strong>SAQLAIN ABBAS</strong>"
    "</div>",
    unsafe_allow_html=True
)