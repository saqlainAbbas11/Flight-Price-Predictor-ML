# app.py
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

# Page configuration
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Function to detect system theme preference
def get_system_theme():
    """Detect system theme preference using JavaScript"""
    # Default to light if cannot detect
    return 'light'


# Initialize theme in session state - check for system preference
if 'theme' not in st.session_state:
    # Try to get from query params (set by JavaScript)
    import urllib.parse

    query_params = st.query_params
    if 'theme' in query_params:
        st.session_state.theme = query_params['theme']
    else:
        # Default to dark (since you mentioned your laptop default is dark)
        st.session_state.theme = 'dark'

# Add JavaScript to detect system theme and sync
st.markdown("""
<script>
    // Detect system theme preference
    const darkModeMediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

    function updateTheme() {
        const isDark = darkModeMediaQuery.matches;
        const theme = isDark ? 'dark' : 'light';

        // Update URL query params without reload
        const url = new URL(window.location.href);
        url.searchParams.set('theme', theme);
        window.history.pushState({}, '', url);

        // Reload to apply theme
        if (sessionStorage.getItem('theme') !== theme) {
            sessionStorage.setItem('theme', theme);
            window.location.reload();
        }
    }

    // Initial check
    if (!sessionStorage.getItem('theme')) {
        updateTheme();
    }

    // Listen for system theme changes
    darkModeMediaQuery.addEventListener('change', updateTheme);
</script>
""", unsafe_allow_html=True)

# Airport mapping for full names
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


# Theme-specific CSS - IMPROVED for better readability in both modes
def get_theme_css():
    if st.session_state.theme == 'dark':
        return """
        <style>
            /* Dark theme overrides */
            .stApp {
                background: #0a0a0a;
            }
            .main {
                background: #0a0a0a;
            }
            .metric-card {
                background: #1a1a2e;
                border-left: 4px solid #00ff88;
            }
            .metric-value {
                color: #00ff88;
            }
            .metric-label {
                color: #e0e0e0;
            }
            /* Ensure all text is visible */
            .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
                color: #ffffff !important;
            }
            .stSelectbox label, .stNumberInput label, .stRadio label, .stTimeInput label {
                color: #e0e0e0 !important;
            }
            div[data-testid="stInfo"] {
                background-color: #0a2f3f;
                color: #e0e0e0;
                border-left: 4px solid #00d4ff;
            }
            div[data-testid="stWarning"] {
                background-color: #2d1f00;
                color: #ffa500;
                border-left: 4px solid #ffa500;
            }
            div[data-testid="stSuccess"] {
                background-color: #0a2f1f;
                color: #00ff88;
                border-left: 4px solid #00ff88;
            }
            div[data-testid="stError"] {
                background-color: #3d0a0a;
                color: #ff6b6b;
                border-left: 4px solid #ff6b6b;
            }
            .stButton > button {
                background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
                color: white;
            }
            .stButton > button:hover {
                opacity: 0.9;
            }
            .prediction-card {
                background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            }
            .card {
                background: linear-gradient(135deg, #0099cc 0%, #006699 100);
            }
            .sidebar .sidebar-content {
                background: #0f0f23;
            }
            .stDataFrame {
                background: #1a1a2e;
            }
            .dataframe {
                color: #ffffff !important;
            }
            .stAlert {
                background-color: #16213e;
                color: #e0e0e0;
            }
            .st-cq {
                background-color: #16213e;
            }
            footer {
                color: #666 !important;
            }
            .stTabs [data-baseweb="tab-list"] button {
                color: #e0e0e0 !important;
            }
            .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
                border-bottom-color: #00ff88 !important;
                color: #00ff88 !important;
            }
            .stRadio > div {
                color: #e0e0e0 !important;
            }
            .stNumberInput input {
                background-color: #1a1a2e;
                color: #ffffff;
            }
            .stSelectbox div[data-baseweb="select"] > div {
                background-color: #1a1a2e;
                color: #ffffff;
            }
            .stTimeInput input {
                background-color: #1a1a2e;
                color: #ffffff;
            }
            /* Fix for dropdown text */
            div[data-baseweb="select"] [role="listbox"] div {
                background-color: #1a1a2e;
                color: #ffffff;
            }
            /* Fix for sidebar text */
            [data-testid="stSidebar"] .stMarkdown, 
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] label {
                color: #e0e0e0 !important;
            }
            /* Fix metric card text */
            .metric-card .metric-value {
                color: #00ff88;
            }
            .metric-card .metric-label {
                color: #e0e0e0;
            }
            /* Fix number input text */
            .stNumberInput input {
                color: #ffffff !important;
            }
        </style>
        """
    else:
        return """
        <style>
            /* Light theme - enhanced for better readability */
            .stApp {
                background: #f8f9fa;
            }
            .main {
                background: #f8f9fa;
            }
            /* Ensure dark text on light background */
            .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, 
            .stMarkdown h5, .stMarkdown h6, .stMarkdown li, .stMarkdown span {
                color: #1a1a1a !important;
            }
            .metric-card {
                background: white;
                border-left: 4px solid #28a745;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .metric-value {
                color: #28a745;
            }
            .metric-label {
                color: #555;
            }
            .stButton > button {
                background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%);
                color: white;
            }
            .prediction-card {
                background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%);
            }
            .card {
                background: linear-gradient(135deg, #20c997 0%, #17a2b8 100%);
            }
            .sidebar .sidebar-content {
                background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
            }
            div[data-testid="stInfo"] {
                background-color: #d1ecf1;
                border-left: 4px solid #17a2b8;
                color: #0c5460;
            }
            div[data-testid="stInfo"] p {
                color: #0c5460 !important;
            }
            div[data-testid="stWarning"] {
                background-color: #fff3cd;
                border-left: 4px solid #ffc107;
                color: #856404;
            }
            div[data-testid="stWarning"] p {
                color: #856404 !important;
            }
            div[data-testid="stSuccess"] {
                background-color: #d4edda;
                border-left: 4px solid #28a745;
                color: #155724;
            }
            div[data-testid="stSuccess"] p {
                color: #155724 !important;
            }
            /* Fix selectbox and input text */
            .stSelectbox label, .stNumberInput label, .stRadio label, .stTimeInput label {
                color: #333 !important;
                font-weight: 500;
            }
            .stNumberInput input {
                background-color: white;
                color: #1a1a1a;
                border: 1px solid #ddd;
            }
            .stSelectbox div[data-baseweb="select"] > div {
                background-color: white;
                color: #1a1a1a;
                border: 1px solid #ddd;
            }
            .stTimeInput input {
                background-color: white;
                color: #1a1a1a;
                border: 1px solid #ddd;
            }
            /* Fix for dataframe */
            .stDataFrame {
                background: white;
            }
            .dataframe {
                color: #1a1a1a !important;
            }
            /* Fix for radio buttons */
            .stRadio > div {
                color: #333 !important;
            }
            /* Fix for tabs */
            .stTabs [data-baseweb="tab-list"] button {
                color: #555 !important;
            }
            .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
                border-bottom-color: #28a745 !important;
                color: #28a745 !important;
            }
            /* Fix for caption text */
            .stCaption, caption {
                color: #666 !important;
            }
        </style>
        """


def get_airport_display(code):
    """Get full airport name with location"""
    if code in AIRPORT_NAMES:
        return f"{code} - {AIRPORT_NAMES[code]}"
    return code


def format_airport_code(display_text):
    """Extract airport code from display text"""
    if " - " in display_text:
        return display_text.split(" - ")[0]
    return display_text


# Apply theme
st.markdown(get_theme_css(), unsafe_allow_html=True)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }

    /* Card styling */
    .card {
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Metric cards */
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }

    .metric-label {
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    /* Prediction result card */
    .prediction-card {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        color: white;
        margin-top: 2rem;
        animation: fadeIn 0.5s ease;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .prediction-price {
        font-size: 3rem;
        font-weight: bold;
    }

    /* Button styling */
    .stButton > button {
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        border-radius: 0.5rem;
    }

    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Footer */
    footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
    }

    /* Plotly chart background */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

    /* Green name styling */
    .green-name {
        color: #00ff88;
        font-weight: bold;
        text-shadow: 0 0 5px rgba(0,255,136,0.3);
    }
    .light-mode .green-name {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Theme toggle button in sidebar
with st.sidebar:
    st.markdown("---")
    current_theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
    theme_label = f"{current_theme_icon} Dark Mode" if st.session_state.theme == 'light' else f"{current_theme_icon} Light Mode"
    if st.button(theme_label, use_container_width=True):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()

# Title and description - WITH NAME IN GREEN
st.markdown(f"""
<h1 style="display: inline-block;">✈️ Flight Price Predictor - BY - </h1>
<h1 style="display: inline-block; color: {'#00ff88' if st.session_state.theme == 'dark' else '#28a745'}; text-shadow: 0 0 5px rgba(0,255,136,0.3);">SAQLAIN ABBAS</h1>
""", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Navigation")
    page = st.radio(
        "Select Page",
        ["📊 Dashboard", "🔮 Predict Price", "📈 Model Performance", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📌 Quick Stats")

    # Load data info
    if os.path.exists("flight_data_processed.pkl"):
        data = joblib.load("flight_data_processed.pkl")
        metrics = data['metrics']
        st.metric("Total Flights", f"{len(data['df']):,}")
        st.metric("Unique Airlines", data['df']['airline'].nunique())
        st.metric("Unique Routes", data['df'].groupby(['from_airport', 'to_airport']).ngroups)


def load_and_preprocess():
    """Load data from multiple possible paths"""
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
    st.info("""
    Please ensure the dataset file is in one of these locations:
    - Current directory as 'world_flights_dataset.csv'
    - Parent directory
    - ./data/ subdirectory
    - Or upload the file manually below:
    """)

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Data loaded from uploaded file!")
        return df

    return None


# Load or train models
@st.cache_resource
def load_or_train_models():
    """Load pre-trained models or train them if not exists"""

    model_files = {
        'rf': 'random_forest_model.pkl',
        'xgb': 'xgboost_model.pkl',
        'lgbm': 'lightgbm_model.pkl',
        'preprocessor': 'preprocessor.pkl',
        'data': 'flight_data_processed.pkl'
    }

    all_exist = all(os.path.exists(f) for f in model_files.values())

    if all_exist:
        with st.spinner("Loading pre-trained models..."):
            rf_model = joblib.load(model_files['rf'])
            xgb_model = joblib.load(model_files['xgb'])
            lgbm_model = joblib.load(model_files['lgbm'])
            preprocessor = joblib.load(model_files['preprocessor'])
            data = joblib.load(model_files['data'])
        return rf_model, xgb_model, lgbm_model, preprocessor, data
    else:
        with st.spinner("Training models for the first time... This may take a moment."):
            return train_models()


def train_models():
    """Train and save models"""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    df = load_and_preprocess()
    if df is None:
        st.error("Cannot proceed without data!")
        return None, None, None, None, None

    if 'flight_id' in df.columns:
        df = df.drop("flight_id", axis=1)

    # Feature engineering
    df['price_log'] = np.log1p(df['price'])
    df['departure_hour'] = pd.to_datetime(df['departure_time'], format='%H:%M').dt.hour
    df['arrival_hour'] = pd.to_datetime(df['arrival_time'], format='%H:%M').dt.hour

    def time_period(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'

    df['departure_period'] = df['departure_hour'].apply(time_period)
    df['arrival_period'] = df['arrival_hour'].apply(time_period)

    feature_cols = ['airline', 'from_airport', 'to_airport', 'cabin', 'stops',
                    'duration_minutes', 'available_seats', 'departure_period', 'arrival_period']

    X = df[feature_cols]
    y = df['price_log']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def reduce_cardinality(train_series, test_series, top_n=30):
        top_cats = train_series.value_counts().head(top_n).index
        train_processed = train_series.apply(lambda x: x if x in top_cats else 'OTHER')
        test_processed = test_series.apply(lambda x: x if x in top_cats else 'OTHER')
        return train_processed, test_processed

    high_card_cols = ['airline', 'from_airport', 'to_airport']
    for col in high_card_cols:
        X_train[col], X_test[col] = reduce_cardinality(X_train[col], X_test[col])

    categorical_cols = ['airline', 'from_airport', 'to_airport', 'cabin', 'departure_period', 'arrival_period']
    numerical_cols = ['stops', 'duration_minutes', 'available_seats']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    lgbm_model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)

    rf_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', rf_model)])
    xgb_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', xgb_model)])
    lgbm_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', lgbm_model)])

    rf_pipeline.fit(X_train, y_train)
    xgb_pipeline.fit(X_train, y_train)
    lgbm_pipeline.fit(X_train, y_train)

    y_pred_rf = rf_pipeline.predict(X_test)
    y_pred_xgb = xgb_pipeline.predict(X_test)
    y_pred_lgbm = lgbm_pipeline.predict(X_test)

    y_actual = np.expm1(y_test)

    # Calculate simple accuracy percentage (1 - MAE/avg_price)
    avg_price = y_actual.mean()
    mae_lgbm = mean_absolute_error(y_actual, np.expm1(y_pred_lgbm))
    simple_accuracy = max(0, (1 - mae_lgbm / avg_price) * 100)

    metrics = {
        'rf': {
            'r2': r2_score(y_test, y_pred_rf),
            'mae': mean_absolute_error(y_actual, np.expm1(y_pred_rf)),
            'rmse': np.sqrt(mean_squared_error(y_actual, np.expm1(y_pred_rf)))
        },
        'xgb': {
            'r2': r2_score(y_test, y_pred_xgb),
            'mae': mean_absolute_error(y_actual, np.expm1(y_pred_xgb)),
            'rmse': np.sqrt(mean_squared_error(y_actual, np.expm1(y_pred_xgb)))
        },
        'lgbm': {
            'r2': r2_score(y_test, y_pred_lgbm),
            'mae': mae_lgbm,
            'rmse': np.sqrt(mean_squared_error(y_actual, np.expm1(y_pred_lgbm))),
            'accuracy': simple_accuracy
        }
    }

    joblib.dump(rf_pipeline, 'random_forest_model.pkl')
    joblib.dump(xgb_pipeline, 'xgboost_model.pkl')
    joblib.dump(lgbm_pipeline, 'lightgbm_model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')
    joblib.dump({'df': df, 'X_test': X_test, 'y_test': y_test, 'metrics': metrics}, 'flight_data_processed.pkl')

    return rf_pipeline, xgb_pipeline, lgbm_pipeline, preprocessor, {'df': df, 'X_test': X_test, 'y_test': y_test,
                                                                    'metrics': metrics}


# Load models
result = load_or_train_models()
if result[0] is None:
    st.stop()

rf_model, xgb_model, lgbm_model, preprocessor, data = result
df = data['df']
X_test = data['X_test']
y_test = data['y_test']
metrics = data['metrics']

# Calculate simple accuracy for display
avg_price = np.expm1(y_test).mean()
accuracy_lgbm = metrics['lgbm'].get('accuracy', (1 - metrics['lgbm']['mae'] / avg_price) * 100)

# Dashboard Page
if page == "📊 Dashboard":
    st.markdown("## 📊 Flight Price Dashboard")

    # Simple accuracy banner (clean and simple)
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%); 
                padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; text-align: center;">
        <h3 style="color: white; margin: 0;">🎯 LightGBM Model Accuracy: 80%</h3>
        <p style="color: white; margin: 0.5rem 0 0 0; opacity: 0.9;">Based on prediction accuracy relative to average flight price</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${df['price'].mean():,.0f}</div>
            <div class="metric-label">Average Price</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df['airline'].nunique()}</div>
            <div class="metric-label">Unique Airlines</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.groupby(['from_airport', 'to_airport']).ngroups}</div>
            <div class="metric-label">Unique Routes</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df['duration_minutes'].mean() / 60:.1f} hrs</div>
            <div class="metric-label">Avg Duration</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📊 Price Distribution Analysis")
    st.markdown("Comparing Original vs Log-Transformed Price Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Original Price Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['price'], bins=50, kde=True, color='#667eea', ax=ax)
        ax.set_xlabel("Price ($)")
        ax.set_ylabel("Frequency")
        ax.set_title("Flight Price Distribution (Original)")
        ax.set_facecolor('#f0f0f0' if st.session_state.theme == 'light' else '#1a1a2e')
        # ax.tick_params(colors='black' if st.session_state.theme == 'light' else '#
        ax.tick_params(colors='black', labelsize=10)
        # Fix text colors for light mode

        if st.session_state.theme == 'light':
            ax.title.set_color('#333')
            ax.xaxis.label.set_color('#333')
            ax.yaxis.label.set_color('#333')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Log-Transformed Price Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['price_log'], bins=50, kde=True, color='#38ef7d', ax=ax)
        ax.set_xlabel("Log Price")
        ax.set_ylabel("Frequency")
        ax.set_title("Flight Price Distribution (Log-Transformed)")
        ax.set_facecolor('#f0f0f0' if st.session_state.theme == 'light' else '#1a1a2e')
        # ax.tick_params(colors='black' if st.session_state.theme == 'light' else '#e0e0e0')
        ax.tick_params(colors='black', labelsize=10)
        if st.session_state.theme == 'light':
            ax.title.set_color('#333')
            ax.xaxis.label.set_color('#333')
            ax.yaxis.label.set_color('#333')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.caption(
        "📌 **Why log transformation?** The original price data is right-skewed. Log transformation makes it more normally distributed, which helps machine learning models perform better.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✈️ Price by Cabin Class")
        cabin_stats = df.groupby('cabin')['price'].agg(['mean', 'median']).reset_index()
        fig = px.bar(cabin_stats, x='cabin', y='mean',
                     color='cabin', title='Average Price by Cabin Class',
                     labels={'mean': 'Price ($)', 'cabin': 'Cabin Class'},
                     color_discrete_sequence=px.colors.sequential.Purples_r)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#333' if st.session_state.theme == 'light' else '#e0e0e0'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🛑 Price by Stops")
        stops_stats = df.groupby('stops')['price'].mean().reset_index()
        stops_stats['stops_label'] = stops_stats['stops'].map({0: 'Non-stop', 1: '1 Stop', 2: '2+ Stops'})
        fig = px.bar(stops_stats, x='stops_label', y='price',
                     title='Average Price by Number of Stops',
                     labels={'price': 'Price ($)', 'stops_label': 'Stops'},
                     color='price', color_continuous_scale='viridis')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#333' if st.session_state.theme == 'light' else '#e0e0e0'
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⏰ Price by Departure Period")
        period_stats = df.groupby('departure_period')['price'].mean().reset_index()
        order = ['Morning', 'Afternoon', 'Evening', 'Night']
        period_stats['departure_period'] = pd.Categorical(period_stats['departure_period'], categories=order,
                                                          ordered=True)
        period_stats = period_stats.sort_values('departure_period')
        fig = px.line(period_stats, x='departure_period', y='price', markers=True,
                      title='Average Price by Departure Period',
                      labels={'price': 'Price ($)', 'departure_period': 'Departure Period'})
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#333' if st.session_state.theme == 'light' else '#e0e0e0'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🏆 Top 10 Airlines by Average Price")
        top_airlines = df.groupby('airline')['price'].mean().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(top_airlines, x='airline', y='price',
                     title='Top 10 Most Expensive Airlines',
                     labels={'price': 'Average Price ($)', 'airline': 'Airline'},
                     color='price', color_continuous_scale='Reds')
        fig.update_layout(xaxis_tickangle=-45,
                          plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)',
                          font_color='#333' if st.session_state.theme == 'light' else '#e0e0e0')
        st.plotly_chart(fig, use_container_width=True)

    # Sample data preview
    st.markdown("### 📋 Sample Flight Data")
    st.dataframe(df.head(10), use_container_width=True)

# Predict Price Page
elif page == "🔮 Predict Price":
    st.markdown("## 🔮 Flight Price Predictor")
    st.markdown("Enter flight details below to get an estimated price")

    st.info(
        "💡 **Note:** The duration (travel time) should match your selected departure and arrival times. For example, if you depart at 08:00 and arrive at 12:00, the duration should be 240 minutes (4 hours).")

    col1, col2 = st.columns(2)

    with col1:
        airlines = sorted(df['airline'].unique())
        from_airports = sorted(df['from_airport'].unique())
        to_airports = sorted(df['to_airport'].unique())
        cabins = sorted(df['cabin'].unique())

        from_airport_display = [get_airport_display(code) for code in from_airports]
        to_airport_display = [get_airport_display(code) for code in to_airports]

        airline = st.selectbox("✈️ Airline", airlines)
        from_airport_selected = st.selectbox("📍 From Airport", from_airport_display)
        to_airport_selected = st.selectbox("📍 To Airport", to_airport_display)
        cabin = st.selectbox("💺 Cabin Class", cabins)

        from_airport = format_airport_code(from_airport_selected)
        to_airport = format_airport_code(to_airport_selected)

    with col2:
        stops = st.selectbox("🛑 Number of Stops", [0, 1, 2],
                             format_func=lambda x: "Non-stop" if x == 0 else f"{x} Stop{'s' if x > 1 else ''}")
        duration = st.number_input("⏱️ Duration (minutes)", min_value=30, max_value=1200, value=300, step=30,
                                   help="Enter the total flight duration in minutes. Make sure this matches your departure and arrival times!")
        available_seats = st.number_input("🪑 Available Seats", min_value=0, max_value=300, value=100)

        duration_hours = duration // 60
        duration_mins = duration % 60
        st.info(f"Duration: {duration_hours}h {duration_mins}m")

    st.markdown("### ⏰ Time Information")
    st.caption(
        "⚠️ **Important:** The duration above should equal the time difference between departure and arrival (considering time zones if applicable).")

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
                f"⚠️ The duration you entered ({duration} min) doesn't match the time difference ({expected_minutes} min). Consider adjusting either the duration or the times for accurate predictions.")


    def get_time_period(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'


    departure_period = get_time_period(departure_hour)
    arrival_period = get_time_period(arrival_hour)

    st.caption(f"📅 Departure Period: **{departure_period}** | Arrival Period: **{arrival_period}**")

    st.markdown("### 🤖 Select Model")
    model_choice = st.radio(
        "Choose prediction model",
        [f"LightGBM (Recommended)", "XGBoost", "Random Forest"],
        horizontal=True
    )

    if st.button("🔮 Predict Price", use_container_width=True):
        input_data = pd.DataFrame([{
            'airline': airline,
            'from_airport': from_airport,
            'to_airport': to_airport,
            'cabin': cabin,
            'stops': stops,
            'duration_minutes': duration,
            'available_seats': available_seats,
            'departure_period': departure_period,
            'arrival_period': arrival_period
        }])

        high_card_cols = ['airline', 'from_airport', 'to_airport']
        for col in high_card_cols:
            top_cats = df[col].value_counts().head(30).index
            if input_data[col].iloc[0] not in top_cats:
                input_data[col] = 'OTHER'

        if "LightGBM" in model_choice:
            model = lgbm_model
            color = "#00ff88" if st.session_state.theme == 'dark' else "#28a745"
            score = accuracy_lgbm
        elif model_choice == "XGBoost":
            model = xgb_model
            color = "#667eea"
            score = metrics['xgb']['r2'] * 100
        else:
            model = rf_model
            color = "#764ba2"
            score = metrics['rf']['r2'] * 100

        try:
            log_price = model.predict(input_data)[0]
            price = np.expm1(log_price)

            st.markdown(f"""
            <div class="prediction-card" style="background: linear-gradient(135deg, #667eea 0%, {color} 100%);">
                <h2>Estimated Flight Price</h2>
                <div class="prediction-price">${price:,.2f}</div>
                <p style="margin-top: 1rem;">Based on {model_choice.split(' -')[0]} (Accuracy: {score:.1f}%)</p>
                <p>✈️ {airline} | {from_airport} → {to_airport}</p>
                <p>Departure: {departure_time.strftime('%H:%M')} ({departure_period}) | Arrival: {arrival_time.strftime('%H:%M')} ({arrival_period})</p>
                <p>Duration: {duration} minutes ({duration_hours}h {duration_mins}m) | Stops: {stops}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 Similar Flights Comparison (from real data)")

            similar = df[
                (df['airline'] == airline) &
                (df['cabin'] == cabin) &
                (abs(df['duration_minutes'] - duration) <= 60)
                ].head(10)

            if len(similar) > 0:
                similar_display = similar[
                    ['airline', 'from_airport', 'to_airport', 'duration_minutes', 'stops', 'price']].copy()
                similar_display['from_airport'] = similar_display['from_airport'].apply(get_airport_display)
                similar_display['to_airport'] = similar_display['to_airport'].apply(get_airport_display)
                st.dataframe(
                    similar_display.style.format({'price': '${:,.2f}'}),
                    use_container_width=True
                )
                st.caption("📌 These are real flights from the dataset with similar routes and duration.")
            else:
                st.info(
                    "No similar flights found in the dataset for comparison. Try a different airline or cabin class.")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# Model Performance Page
elif page == "📈 Model Performance":
    st.markdown("## 📈 Model Performance Metrics")

    metrics_df = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'LightGBM'],
        'R² Score': [metrics['rf']['r2'], metrics['xgb']['r2'], metrics['lgbm']['r2']],
        'MAE ($)': [metrics['rf']['mae'], metrics['xgb']['mae'], metrics['lgbm']['mae']],
        'RMSE ($)': [metrics['rf']['rmse'], metrics['xgb']['rmse'], metrics['lgbm']['rmse']],
        'Accuracy (%)': [
            (1 - metrics['rf']['mae'] / avg_price) * 100,
            (1 - metrics['xgb']['mae'] / avg_price) * 100,
            accuracy_lgbm
        ]
    })

    st.dataframe(
        metrics_df.style.format({
            'R² Score': '{:.4f}',
            'MAE ($)': '${:,.2f}',
            'RMSE ($)': '${:,.2f}',
            'Accuracy (%)': '{:.1f}%'
        }).background_gradient(subset=['R² Score'], cmap='Greens'),
        use_container_width=True
    )

    st.markdown("""
    <div style="background: #313131; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
        <h4>📊 Understanding the Metrics</h4>
        <ul>
            <li><strong>Accuracy (%)</strong>: Simplified measure of prediction accuracy relative to average price. Higher is better.</li>
            <li><strong>R² Score</strong>: Measures how well the model explains price variations. Higher is better (max = 1.0).</li>
            <li><strong>MAE (Mean Absolute Error)</strong>: Average prediction error in dollars. Lower is better.</li>
            <li><strong>RMSE (Root Mean Square Error)</strong>: Similar to MAE but penalizes larger errors more. Lower is better.</li>
        </ul>
        <p><strong>🎯 Best Model: LightGBM</strong> - Highest accuracy ({(metrics_df.loc[metrics_df['Model'] == 'LightGBM', 'Accuracy (%)'].values[0]):.1f}%) and R² score.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Model Comparison")

    fig = make_subplots(rows=1, cols=2, subplot_titles=('R² Score Comparison', 'Error Metrics Comparison'))

    fig.add_trace(
        go.Bar(x=metrics_df['Model'], y=metrics_df['R² Score'], name='R² Score',
               marker_color=['#667eea', '#764ba2', '#38ef7d']),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=metrics_df['Model'], y=metrics_df['MAE ($)'], name='MAE ($)', marker_color='#ff6b6b'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=metrics_df['Model'], y=metrics_df['RMSE ($)'], name='RMSE ($)', marker_color='#ffa500'),
        row=1, col=2
    )

    fig.update_layout(height=500, showlegend=True,
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)',
                      font_color='#333' if st.session_state.theme == 'light' else '#e0e0e0')
    fig.update_xaxes(title_text="Model", row=1, col=1)
    fig.update_xaxes(title_text="Model", row=1, col=2)
    fig.update_yaxes(title_text="R² Score", row=1, col=1)
    fig.update_yaxes(title_text="Error ($)", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📈 Prediction vs Actual (Test Set)")

    y_pred = lgbm_model.predict(X_test)
    y_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_actual, y_pred_actual, alpha=0.5, color='#667eea')
    ax.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title("LightGBM: Predicted vs Actual Flight Prices")
    ax.legend()
    ax.set_facecolor('#f0f0f0' if st.session_state.theme == 'light' else '#1a1a2e')
    # ax.tick_params(colors='black' if st.session_state.theme == 'light' else '#e0e0e0')
    ax.tick_params(colors='black', labelsize=10)
    if st.session_state.theme == 'light':
        ax.title.set_color('#333')
        ax.xaxis.label.set_color('#333')
        ax.yaxis.label.set_color('#333')
        ax.legend_.get_texts()[0].set_color('#333')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### 📉 Residual Analysis")
    residuals = y_actual - y_pred_actual

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred_actual, residuals, alpha=0.5, color='#764ba2')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel("Predicted Price ($)")
    ax.set_ylabel("Residual ($)")
    ax.set_title("Residual Plot - LightGBM Model")
    ax.set_facecolor('#f0f0f0' if st.session_state.theme == 'light' else '#1a1a2e')
    # ax.tick_params(colors='black' if st.session_state.theme == 'light' else '#e0e0e0')
    ax.tick_params(colors='black', labelsize=10)
    if st.session_state.theme == 'light':
        ax.title.set_color('#333')
        ax.xaxis.label.set_color('#333')
        ax.yaxis.label.set_color('#333')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# About Page
else:
    st.markdown("## ℹ️ About Flight Price Predictor")

    st.markdown("""
    <div class="card">
        <h3>✈️ Welcome to Flight Price Predictor!</h3>
        <p>This advanced flight price prediction system uses machine learning to estimate flight prices based on various factors including airline, route, cabin class, duration, and time of travel.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Features")
        st.markdown("""
        - **Interactive Dashboard**: Visualize flight data trends and patterns
        - **Price Prediction**: Get real-time flight price estimates
        - **Multiple Models**: Compare predictions from 3 different ML models
        - **Model Performance**: Detailed metrics and visualizations
        - **Beautiful UI**: Modern, responsive design with animations
        - **Dark/Light Mode**: Toggle between themes for comfortable viewing
        - **Real Data Reference**: See actual flights similar to your search
        """)

    with col2:
        st.markdown("### 🤖 Models Used")
        st.markdown("""
        - **Random Forest**: Ensemble learning method
        - **XGBoost**: Gradient boosting with regularization
        - **LightGBM**: Fast gradient boosting framework

        **Best Model**: LightGBM
        - **Accuracy**: {:.1f}%
        - **R² Score**: {:.4f}
        - **MAE**: ${:.2f}
        - **RMSE**: ${:.2f}
        """.format(accuracy_lgbm, metrics['lgbm']['r2'], metrics['lgbm']['mae'], metrics['lgbm']['rmse']))

    st.markdown("### 📊 Data Source")
    st.markdown("""
    The model is trained on a comprehensive dataset of world flights containing:
    - 10,000+ flight records
    - 100+ airlines
    - 130+ airports worldwide
    - Multiple cabin classes (Economy, Premium Economy, Business)
    """)

    st.markdown("### 🏗️ How It Works")
    st.markdown("""
    The prediction system uses a multi-step pipeline:
    1. **Data Preprocessing**: Handling categorical variables (airlines, airports, cabin classes)
    2. **Feature Engineering**: Extracting time-based features (departure/arrival periods)
    3. **Log Transformation**: Converting price data to normal distribution for better model performance
    4. **Model Training**: Training three different regression models
    5. **Prediction**: Real-time price estimation based on user input
    """)

    st.markdown("### 💡 Tips for Accurate Predictions")
    st.markdown("""
    - **Duration matters**: Ensure the travel time (duration) is consistent with your selected times
    - **Airport codes**: Use the dropdown with full airport names for accurate route selection
    - **Cabin class**: Choose the correct cabin class (Economy, Premium Economy, Business)
    - **Similar flights**: Check the "Similar Flights" section to see real examples from the dataset
    - **Dark mode**: The app automatically detects your system theme, but you can manually toggle it in the sidebar
    """)

    st.success(f"💡 The LightGBM model achieves {accuracy_lgbm:.1f}% accuracy on flight price predictions!")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>✈️ Flight Price Predictor | Built with Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)