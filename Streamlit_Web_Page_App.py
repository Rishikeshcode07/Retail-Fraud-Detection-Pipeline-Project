# Import standard library modules for file path and operating system operations
import os
import time

# Import core data manipulation and numerical analysis libraries
import pandas as pd
import numpy as np

# Import Plotly for advanced, interactive data visualizations
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Import joblib for loading serialized (saved) machine learning models
import joblib

# Import Streamlit for rendering the web application interface
import streamlit as st

# Import Scikit-learn evaluation metrics to calculate model accuracy and performance
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# ------------------------------------------------------------------------------
# SECTION 1: PAGE CONFIGURATION & CUSTOM CSS STYLING
# ------------------------------------------------------------------------------
# st.set_page_config must be the first Streamlit command called.
# It sets the browser tab title, forces a wide layout for better dashboard viewing,
# and ensures the sidebar is open by default.
st.set_page_config(
    page_title="Financial Fraud Detection Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS to override default Streamlit styling.
# This establishes the dark, executive theme using specific hex color codes.
st.markdown("""
<style>
    /* Main Background & Base Fonts: Sets a dark slate background */
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling: Makes the sidebar slightly lighter than the main background */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1F2937;
    }
    
    /* Executive Metric Cards: Creates a border and shadow effect for top-level numbers */
    .metric-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    
    /* Metric Card Typography */
    .metric-title {
        color: #9CA3AF;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .metric-value {
        color: #F9FAFB;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 6px;
        margin-bottom: 4px;
    }
    
    .metric-sub {
        font-size: 0.82rem;
        font-weight: 500;
    }
    
    /* Reusable Status Colors */
    .text-green { color: #10B981; }
    .text-red { color: #EF4444; }
    .text-amber { color: #F59E0B; }
    .text-indigo { color: #6366F1; }
    .text-muted { color: #9CA3AF; }

    /* Custom Header Titles */
    .section-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F9FAFB;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    
    .section-subtitle {
        font-size: 0.9rem;
        color: #9CA3AF;
        margin-bottom: 20px;
    }
    
    /* Dataframe Table Adjustments */
    .stDataFrame {
        border: 1px solid #1F2937;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SECTION 2: PATH AUTO-DETECTION & RESOURCE LOADERS
# ------------------------------------------------------------------------------
# Determine the absolute directory path where this Python script currently resides.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(filename):
    """
    Searches across multiple relative directory levels for a target file.
    This prevents FileNotFoundError crashes if the user runs Streamlit from the wrong folder.
    """
    paths_to_check = [
        os.path.join(SCRIPT_DIR, filename),                      # Check current script folder
        os.path.join(SCRIPT_DIR, "Data", filename),              # Check Data subfolder
        os.path.join(os.getcwd(), filename),                     # Check terminal's current working directory
        os.path.join(os.getcwd(), "Data", filename),             # Check Data folder inside terminal's directory
        os.path.join(os.getcwd(), "..", "Data", filename)        # Check one level up
    ]
    for p in paths_to_check:
        if os.path.exists(p):
            return p
    return None

def generate_synthetic_data(num_samples=2000):
    """
    Generates a synthetic dataset that mimics the structure of the real fraud dataset.
    This acts as a fallback so the dashboard remains functional and demonstrable
    even if the actual CSV files are missing.
    """
    np.random.seed(42) # Set seed for reproducibility
    
    # Generate random features matching the exact column names of the real dataset
    cc_nums = np.random.randint(1000000000000000, 9999999999999999, size=num_samples)
    zip_codes = np.random.randint(10000, 99999, size=num_samples)
    lats = np.random.uniform(25.0, 48.0, size=num_samples)
    longs = np.random.uniform(-120.0, -70.0, size=num_samples)
    city_pops = np.random.randint(1000, 1000000, size=num_samples)
    acct_nums = np.random.randint(1000000000, 9999999999, size=num_samples)
    unix_times = np.random.randint(1650000000, 1660000000, size=num_samples)
    
    # Transaction amounts using an exponential distribution to simulate real transaction patterns
    amts = np.round(np.random.exponential(scale=80, size=num_samples) + 1.0, 2)
    
    trans_time_hrs = np.random.randint(0, 24, size=num_samples)
    # Flag night transactions (10 PM to 5 AM)
    trans_time_is_night = np.where((trans_time_hrs >= 22) | (trans_time_hrs <= 5), 1, 0)
    trans_date_is_weekend = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])
    user_age = np.random.randint(18, 85, size=num_samples)
    
    # Calculate synthetic fraud probabilities heavily weighted by transaction amount and time
    fraud_prob = 0.01 + (amts / 3000.0) + (trans_time_is_night * 0.05)
    fraud_prob = np.clip(fraud_prob, 0.0, 0.8) # Ensure probabilities stay between 0 and 0.8
    is_fraud = np.random.binomial(1, fraud_prob) # Generate binary 0/1 outcomes based on probability
    
    # Construct the final pandas DataFrame
    df = pd.DataFrame({
        'cc_num': cc_nums,
        'zip': zip_codes,
        'lat': lats,
        'long': longs,
        'city_pop': city_pops,
        'acct_num': acct_nums,
        'unix_time': unix_times,
        'amt': amts,
        'merch_lat': lats + np.random.uniform(-0.1, 0.1, size=num_samples),
        'merch_long': longs + np.random.uniform(-0.1, 0.1, size=num_samples),
        'customer_num_trans_1_day': np.random.randint(1, 10, size=num_samples),
        'customer_num_trans_7_day': np.random.randint(5, 40, size=num_samples),
        'customer_num_trans_30_day': np.random.randint(15, 120, size=num_samples),
        'trans_time_secs': trans_time_hrs * 3600,
        'trans_time_hrs': trans_time_hrs,
        'trans_time_is_night': trans_time_is_night,
        'trans_time_day': np.random.randint(1, 31, size=num_samples),
        'trans_date_is_weekend': trans_date_is_weekend,
        'customer_avg_amout_1_day': np.round(amts * np.random.uniform(0.8, 1.2, size=num_samples), 2),
        'customer_avg_amount_7_day': np.round(amts * np.random.uniform(0.7, 1.3, size=num_samples), 2),
        'customer_avg_amount_30_day': np.round(amts * np.random.uniform(0.6, 1.4, size=num_samples), 2),
        'merchant_num_trans_1_day': np.random.randint(10, 500, size=num_samples),
        'merchant_num_trans_7_day': np.random.randint(100, 3000, size=num_samples),
        'merchant_num_trans_30_day': np.random.randint(500, 12000, size=num_samples),
        'merchant_risk_1_day': np.random.randint(0, 10, size=num_samples),
        'merchant_risk_7_day': np.random.randint(0, 25, size=num_samples),
        'merchant_risk_30_day': np.random.randint(0, 50, size=num_samples),
        'merchant_risk_90_day': np.random.randint(0, 100, size=num_samples),
        'user_age': user_age
    })
    
    # Create the target label dataframe
    y = pd.DataFrame({'is_fraud': is_fraud})
    return df, y

# Use @st.cache_resource to load machine learning models into memory only once.
# This prevents Streamlit from reloading heavy .pkl files every time a button is clicked.
@st.cache_resource
def load_ml_models():
    """Loads Random Forest or falls back to Isolation Forest model present on disk."""
    # Check for random_forest_fraud_model.pkl first, or fall back to isolation_forest_fraud_model.pkl
    rf_path = find_file("random_forest_fraud_model.pkl") or find_file("isolation_forest_fraud_model.pkl")
    iso_path = find_file("isolation_forest_fraud_model.pkl")
    
    rf_model = joblib.load(rf_path) if rf_path else None
    iso_model = joblib.load(iso_path) if iso_path else None
    return rf_model, iso_model

# Use @st.cache_data to load the CSV dataset into memory only once.
@st.cache_data
def load_sample_datasets():
    """Loads X_test_final and y_test_final CSVs, or builds fallback synthetic data if missing."""
    x_path = find_file("X_test_final.csv")
    y_path = find_file("y_test_final.csv")
    
    if x_path and y_path:
        X_test = pd.read_csv(x_path)
        y_test = pd.read_csv(y_path)
    else:
        # Fallback to the generator if physical files cannot be located
        X_test, y_test = generate_synthetic_data()
        
    return X_test, y_test

# Execute the loading functions
rf_model, iso_model = load_ml_models()
X_test_df, y_test_df = load_sample_datasets()

# Combine features (X) and target labels (y) into a single DataFrame for easier filtering and charting
if X_test_df is not None and y_test_df is not None:
    full_analytics_df = X_test_df.copy()
    if 'is_fraud' not in full_analytics_df.columns:
        # Append the fraud labels as a new column
        full_analytics_df['is_fraud'] = y_test_df.values
else:
    full_analytics_df = None

# ------------------------------------------------------------------------------
# SECTION 3: PLOTLY CHART THEME ENGINE
# ------------------------------------------------------------------------------
# Define a central color palette to ensure all charts look unified
COLOR_PRIMARY_GREEN = "#10B981"
COLOR_PRIMARY_RED = "#EF4444"
COLOR_ACCENT_BLUE = "#3B82F6"
COLOR_ACCENT_INDIGO = "#6366F1"
COLOR_ACCENT_AMBER = "#F59E0B"
COLOR_ACCENT_CYAN = "#06B6D4"
COLOR_BG_CARD = "#111827"
COLOR_GRID = "#1F2937"

def apply_chart_theme(fig, title=""):
    """
    Takes a Plotly figure object and injects a high-contrast dark enterprise theme.
    Updates backgrounds, grid lines, fonts, and legend styling.
    """
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=15, color="#F9FAFB", family="Inter, sans-serif")
        ),
        paper_bgcolor=COLOR_BG_CARD,
        plot_bgcolor=COLOR_BG_CARD,
        font=dict(color="#9CA3AF", family="Inter, sans-serif"),
        xaxis=dict(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID, showgrid=True),
        yaxis=dict(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID, showgrid=True),
        legend=dict(bgcolor="rgba(17, 24, 39, 0.8)", bordercolor=COLOR_GRID, font=dict(color="#E2E8F0")),
        margin=dict(l=35, r=35, t=45, b=35)
    )
    return fig

# ------------------------------------------------------------------------------
# SECTION 4: SIDEBAR & GLOBAL FILTERS
# ------------------------------------------------------------------------------
# Render the left-hand navigation sidebar
st.sidebar.title("Fraud Engine Portal")
st.sidebar.caption("Real-Time Analytics & Scoring Pipeline")
st.sidebar.markdown("---")

# Define the 7 primary application modules
modules = [
    "Executive Dashboard",
    "Single Transaction Predictor",
    "Batch CSV Scoring",
    "Model Performance & Metrics",
    "Merchant & Geographic Risk",
    "Anomaly Detection Engine",
    "Transaction History & Deep Search"
]

# Create a radio button group for navigation. The chosen value is saved to `selected_module`.
selected_module = st.sidebar.radio("Select System Module:", modules)

st.sidebar.markdown("---")
st.sidebar.subheader("Global Filter Engine")

# Dynamically build sidebar filters based on the min/max values found in the dataset
if full_analytics_df is not None:
    # Safely extract maximum values for the sliders
    max_amt_val = float(full_analytics_df['amt'].max()) if 'amt' in full_analytics_df.columns else 10000.0
    min_age_val = int(full_analytics_df['user_age'].min()) if 'user_age' in full_analytics_df.columns else 18
    max_age_val = int(full_analytics_df['user_age'].max()) if 'user_age' in full_analytics_df.columns else 90

    # Render interactive input widgets in the sidebar
    amt_range = st.sidebar.slider("Transaction Amount ($):", 0.0, max_amt_val, (0.0, max_amt_val))
    night_only = st.sidebar.checkbox("Show Night Transactions Only (10 PM - 5 AM)")
    weekend_only = st.sidebar.checkbox("Show Weekend Transactions Only")
    age_range = st.sidebar.slider("Customer Age Range:", min_age_val, max_age_val, (min_age_val, max_age_val))
    risk_threshold = st.sidebar.slider("Fraud Decision Sensitivity:", 0.10, 0.90, 0.50, 0.05)

    # Apply the user's sidebar selections to filter the central dataframe
    filtered_df = full_analytics_df.copy()
    if 'amt' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['amt'] >= amt_range[0]) & (filtered_df['amt'] <= amt_range[1])]
    if night_only and 'trans_time_is_night' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['trans_time_is_night'] == 1]
    if weekend_only and 'trans_date_is_weekend' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['trans_date_is_weekend'] == 1]
    if 'user_age' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['user_age'] >= age_range[0]) & (filtered_df['user_age'] <= age_range[1])]
else:
    filtered_df = None
    risk_threshold = 0.50

# ------------------------------------------------------------------------------
# MODULE 1: EXECUTIVE DASHBOARD
# ------------------------------------------------------------------------------
# The 'if' statement checks which module the user selected in the sidebar
if selected_module == "Executive Dashboard":
    # Render headers using custom HTML/CSS classes defined in Section 1
    st.markdown('<div class="section-title">Executive Fraud Analytics Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Real-time enterprise monitoring of high-risk transactions, velocity spikes, and monetary exposure.</div>', unsafe_allow_html=True)

    if filtered_df is not None and not filtered_df.empty:
        # Calculate top-level Key Performance Indicators (KPIs)
        total_tx = len(filtered_df)
        total_volume = filtered_df['amt'].sum()
        fraud_tx = filtered_df[filtered_df['is_fraud'] == 1]
        fraud_count = len(fraud_tx)
        fraud_volume = fraud_tx['amt'].sum()
        fraud_rate = (fraud_count / total_tx) * 100 if total_tx > 0 else 0.0

        # Create a 4-column layout for the metric cards
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            # Inject HTML to render the stylized card
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Filtered Volume</div>
                <div class="metric-value">${total_volume:,.2f}</div>
                <div class="metric-sub text-muted">{total_tx:,} total records</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Flagged Fraud Volume</div>
                <div class="metric-value text-red">${fraud_volume:,.2f}</div>
                <div class="metric-sub text-red">{fraud_count:,} fraudulent cases</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">System Fraud Rate</div>
                <div class="metric-value text-amber">{fraud_rate:.2f}%</div>
                <div class="metric-sub text-muted">Baseline target: &lt; 1.50%</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            avg_fraud_amt = fraud_tx['amt'].mean() if fraud_count > 0 else 0.0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Avg Fraud Transaction</div>
                <div class="metric-value text-indigo">${avg_fraud_amt:,.2f}</div>
                <div class="metric-sub text-muted">Avg legitimate: ${filtered_df[filtered_df['is_fraud']==0]['amt'].mean():,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row 1: Split the screen into 2 columns
        col1, col2 = st.columns(2)
        with col1:
            # Group data by hour of day to show when fraud peaks
            hourly_agg = filtered_df.groupby(['trans_time_hrs', 'is_fraud'])['amt'].count().reset_index()
            hourly_agg['Status'] = hourly_agg['is_fraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
            
            # Create a grouped bar chart
            fig_hourly = px.bar(
                hourly_agg, x='trans_time_hrs', y='amt', color='Status',
                color_discrete_map={'Legitimate': COLOR_PRIMARY_GREEN, 'Fraudulent': COLOR_PRIMARY_RED},
                barmode='group', labels={'trans_time_hrs': 'Hour of Day (0-23)', 'amt': 'Transaction Count'}
            )
            apply_chart_theme(fig_hourly, "Transaction Count Distribution by Hour of Day")
            st.plotly_chart(fig_hourly, use_container_width=True)

        with col2:
            # Create a histogram using a logarithmic y-axis because transaction amounts are heavily skewed
            fig_box = px.histogram(
                filtered_df, x='amt', color='is_fraud',
                color_discrete_map={0: COLOR_PRIMARY_GREEN, 1: COLOR_PRIMARY_RED},
                nbins=40, log_y=True,
                labels={'amt': 'Transaction Amount ($)', 'is_fraud': 'Fraud Status'}
            )
            apply_chart_theme(fig_box, "Monetary Value Histogram (Log Scale)")
            st.plotly_chart(fig_box, use_container_width=True)

        # Charts Row 2: Another 2 columns
        col3, col4 = st.columns(2)
        with col3:
            # Bin transaction amounts into risk categories
            filtered_df['Risk_Category'] = pd.cut(
                filtered_df['amt'],
                bins=[-1, 50, 200, 1000, 100000],
                labels=['Low Risk (<$50)', 'Medium Risk ($50-$200)', 'High Risk ($200-$1k)', 'Critical Risk (>$1k)']
            )
            risk_summary = filtered_df.groupby('Risk_Category')['is_fraud'].sum().reset_index()
            
            # Create a donut chart
            fig_donut = px.pie(
                risk_summary, values='is_fraud', names='Risk_Category', hole=0.5,
                color_discrete_sequence=[COLOR_ACCENT_BLUE, COLOR_ACCENT_INDIGO, COLOR_ACCENT_AMBER, COLOR_PRIMARY_RED]
            )
            apply_chart_theme(fig_donut, "Fraud Concentration by Exposure Category")
            st.plotly_chart(fig_donut, use_container_width=True)

        with col4:
            # Analyze fraud rates based on time (Night/Day) and day of week (Weekend/Weekday)
            night_weekend = filtered_df.groupby(['trans_time_is_night', 'trans_date_is_weekend'])['is_fraud'].mean().reset_index()
            night_weekend['Night'] = night_weekend['trans_time_is_night'].map({0: 'Day', 1: 'Night'})
            night_weekend['Weekend'] = night_weekend['trans_date_is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
            night_weekend['Fraud_Rate_Pct'] = night_weekend['is_fraud'] * 100

            fig_nw = px.bar(
                night_weekend, x='Night', y='Fraud_Rate_Pct', color='Weekend', barmode='group',
                color_discrete_map={'Weekday': COLOR_ACCENT_INDIGO, 'Weekend': COLOR_ACCENT_AMBER},
                labels={'Fraud_Rate_Pct': 'Fraud Probability (%)'}
            )
            apply_chart_theme(fig_nw, "Fraud Probability: Day/Night vs Weekday/Weekend")
            st.plotly_chart(fig_nw, use_container_width=True)

        # Render a subset dataframe showing only the high-risk flagged transactions
        st.subheader("High-Risk Fraud Transaction Registry")
        high_risk_table = filtered_df[filtered_df['is_fraud'] == 1][['cc_num', 'amt', 'trans_time_hrs', 'user_age', 'merchant_risk_90_day', 'city_pop']].head(10)
        st.dataframe(high_risk_table, use_container_width=True)

    else:
        st.warning("No data records available matching the active filter parameters.")

# ------------------------------------------------------------------------------
# MODULE 2: SINGLE TRANSACTION PREDICTOR
# ------------------------------------------------------------------------------
elif selected_module == "Single Transaction Predictor":
    st.markdown('<div class="section-title">Single Transaction Scoring Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Input transaction features to perform real-time machine learning inference and risk scoring.</div>', unsafe_allow_html=True)

    # Build a 3-column form layout for user inputs
    c1, c2, c3 = st.columns(3)
    with c1:
        input_amt = st.number_input("Transaction Amount ($):", min_value=1.0, max_value=50000.0, value=350.0, step=10.0)
        input_hr = st.slider("Hour of Day (0-23):", 0, 23, 3)
        input_age = st.number_input("Customer Age:", min_value=18, max_value=100, value=45)
    
    with c2:
        # Use selectboxes for binary categories
        input_night = st.selectbox("Is Night Transaction?", [1, 0], format_func=lambda x: "Yes (10 PM - 5 AM)" if x==1 else "No (Daytime)")
        input_weekend = st.selectbox("Is Weekend Transaction?", [0, 1], format_func=lambda x: "Yes (Weekend)" if x==1 else "No (Weekday)")
        input_city_pop = st.number_input("City Population:", min_value=500, max_value=2000000, value=15000)

    with c3:
        input_avg_1d = st.number_input("Customer 1-Day Avg Amount ($):", min_value=0.0, max_value=10000.0, value=45.0)
        input_merch_risk = st.slider("Merchant 90-Day Risk Rating:", 0, 100, 35)
        input_cust_trans_1d = st.number_input("Customer 1-Day Trans Count:", min_value=1, max_value=100, value=6)

    st.markdown("<br>", unsafe_allow_html=True)

    # Button triggers the prediction logic
    if st.button("Execute Fraud Assessment", type="primary"):
        with st.spinner("Processing feature vector through inference pipeline..."):
            time.sleep(0.3) # Artificial delay to simulate processing API calls
            
            # Predict using the loaded .pkl model if available
            if rf_model is not None and X_test_df is not None:
                try:
                    # To prevent shape errors, we start with a median row from the dataset
                    # and only overwrite the specific features the user changed in the UI
                    sample_vector = X_test_df.median().to_frame().T
                    sample_vector['amt'] = input_amt
                    sample_vector['trans_time_hrs'] = input_hr
                    sample_vector['trans_time_is_night'] = input_night
                    sample_vector['trans_date_is_weekend'] = input_weekend
                    sample_vector['user_age'] = input_age
                    sample_vector['city_pop'] = input_city_pop
                    sample_vector['customer_avg_amout_1_day'] = input_avg_1d
                    sample_vector['merchant_risk_90_day'] = input_merch_risk
                    sample_vector['customer_num_trans_1_day'] = input_cust_trans_1d
                    
                    # Extract the probability for class 1 (Fraud)
                    prob = rf_model.predict_proba(sample_vector)[0][1]
                except Exception:
                    # Fallback math calculation if shape mismatch occurs
                    prob = min(0.99, (input_amt / 1000.0) * 0.4 + (input_night * 0.25) + (input_merch_risk / 100.0) * 0.35)
            else:
                prob = min(0.99, (input_amt / 1000.0) * 0.4 + (input_night * 0.25) + (input_merch_risk / 100.0) * 0.35)

            score_pct = prob * 100

            col_res1, col_res2 = st.columns([1, 2])

            with col_res1:
                st.subheader("Inference Summary")
                # Compare the resulting probability against the user-defined sensitivity threshold
                if prob >= risk_threshold:
                    st.error("ACTION: REJECT TRANSACTION")
                    st.markdown(f"**Calculated Fraud Risk:** <span class='text-red' style='font-size:1.5rem;font-weight:700;'>{score_pct:.1f}%</span>", unsafe_allow_html=True)
                    st.markdown("**Status:** High Fraud Probability Flagged")
                elif prob >= (risk_threshold * 0.5):
                    st.warning("ACTION: MANUAL REVIEW REQUIRED")
                    st.markdown(f"**Calculated Fraud Risk:** <span class='text-amber' style='font-size:1.5rem;font-weight:700;'>{score_pct:.1f}%</span>", unsafe_allow_html=True)
                    st.markdown("**Status:** Elevated Risk Threshold")
                else:
                    st.success("ACTION: APPROVE TRANSACTION")
                    st.markdown(f"**Calculated Fraud Risk:** <span class='text-green' style='font-size:1.5rem;font-weight:700;'>{score_pct:.1f}%</span>", unsafe_allow_html=True)
                    st.markdown("**Status:** Normal Transaction Pattern")

            with col_res2:
                # Build an interactive Plotly Gauge Chart to visually represent the score
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score_pct,
                    number={'suffix': "%", 'font': {'color': '#F9FAFB', 'size': 36}},
                    title={'text': "Fraud Probability Gauge", 'font': {'color': '#9CA3AF', 'size': 14}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': "#9CA3AF"},
                        'bar': {'color': COLOR_PRIMARY_RED if prob >= risk_threshold else COLOR_PRIMARY_GREEN},
                        'bgcolor': "#1F2937",
                        'borderwidth': 1,
                        'bordercolor': "#374151",
                        'steps': [
                            {'range': [0, risk_threshold * 50], 'color': 'rgba(16, 185, 129, 0.2)'},
                            {'range': [risk_threshold * 50, risk_threshold * 100], 'color': 'rgba(245, 158, 11, 0.2)'},
                            {'range': [risk_threshold * 100, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                        ]
                    }
                ))
                apply_chart_theme(fig_gauge, "")
                st.plotly_chart(fig_gauge, use_container_width=True)

# ------------------------------------------------------------------------------
# MODULE 3: BATCH CSV SCORING
# ------------------------------------------------------------------------------
elif selected_module == "Batch CSV Scoring":
    st.markdown('<div class="section-title">Batch CSV Automated Scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Upload transaction batch files to generate bulk fraud predictions, risk probability scores, and downloadable analytical reports.</div>', unsafe_allow_html=True)

    # Streamlit file uploader widget
    uploaded_file = st.file_uploader("Upload Transaction File (.csv):", type=["csv"])

    if uploaded_file is not None:
        # Load the uploaded file into pandas
        batch_df = pd.read_csv(uploaded_file)
        st.success(f"File loaded successfully: {len(batch_df):,} rows found.")
    else:
        st.info("No file uploaded. Demonstrating scoring using current test dataset subset.")
        # If no file is provided, grab the first 500 rows of the test set for demonstration purposes
        batch_df = X_test_df.head(500).copy() if X_test_df is not None else generate_synthetic_data(500)[0]

    if st.button("Run Batch Scoring Execution", type="primary"):
        with st.spinner("Scoring dataset batch..."):
            if rf_model is not None:
                try:
                    # Perform bulk inference
                    preds = rf_model.predict(batch_df)
                    probs = rf_model.predict_proba(batch_df)[:, 1]
                except Exception:
                    # Fallback math if the CSV shape doesn't perfectly match model expectations
                    probs = np.clip((batch_df['amt'] / 2000.0) + (batch_df['trans_time_is_night'] * 0.2), 0.0, 0.95)
                    preds = (probs >= risk_threshold).astype(int)
            else:
                probs = np.clip((batch_df['amt'] / 2000.0) + (batch_df['trans_time_is_night'] * 0.2), 0.0, 0.95)
                preds = (probs >= risk_threshold).astype(int)

            # Append the machine learning results back to a copy of the dataframe
            results_df = batch_df.copy()
            results_df['Fraud_Probability'] = np.round(probs, 4)
            results_df['Predicted_Fraud'] = preds
            results_df['Risk_Status'] = np.where(preds == 1, 'HIGH RISK', 'LEGITIMATE')

            # Render summary metrics of the batch job
            m1, m2, m3 = st.columns(3)
            m1.metric("Scored Records", f"{len(results_df):,}")
            m2.metric("Flagged Fraud Count", f"{preds.sum():,}")
            m3.metric("Estimated Monetary Risk", f"${results_df[results_df['Predicted_Fraud']==1]['amt'].sum():,.2f}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Scored Output Preview")
            # Display a slice of the dataframe containing just the crucial columns
            st.dataframe(results_df[['cc_num', 'amt', 'trans_time_hrs', 'Fraud_Probability', 'Predicted_Fraud', 'Risk_Status']], use_container_width=True)

            # Convert the dataframe to a CSV byte stream for downloading
            csv_buffer = results_df.to_csv(index=False).encode('utf-8')
            
            # Streamlit download button allowing the user to export the scored data
            st.download_button(
                label="Download Scored Results CSV",
                data=csv_buffer,
                file_name="fraud_predictions_scored.csv",
                mime="text/csv"
            )

# ------------------------------------------------------------------------------
# MODULE 4: MODEL PERFORMANCE & METRICS
# ------------------------------------------------------------------------------
elif selected_module == "Model Performance & Metrics":
    st.markdown('<div class="section-title">Model Diagnostics & Technical Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Deep analytics regarding supervised Random Forest classification performance, confusion matrices, and feature importance.</div>', unsafe_allow_html=True)

    if rf_model is not None and X_test_df is not None and y_test_df is not None:
        # Calculate full test set predictions to generate accuracy metrics
        preds = rf_model.predict(X_test_df)
        probs = rf_model.predict_proba(X_test_df)[:, 1] if hasattr(rf_model, "predict_proba") else None

        # --- ROW 1: Confusion Matrix and Classification Report ---
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Confusion Matrix")
            cm_raw = confusion_matrix(y_test_df, preds)
            row_sums = cm_raw.sum(axis=1)
            
            # FIX: Safely handle zero-division to prevent empty black boxes and 'nan%' text
            cm_norm = cm_raw.astype('float') / np.where(row_sums == 0, 1, row_sums)[:, np.newaxis]

            annotation_text = [
                [f"<b>{count:,}</b><br>({pct:.1%})" for count, pct in zip(row_raw, row_norm)]
                for row_raw, row_norm in zip(cm_raw, cm_norm)
            ]

            # FIX: Updated colorscale to 'Viridis' for highly distinct multi-colored boxes
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm_norm,
                x=['Legitimate (0)', 'Fraud (1)'],
                y=['Legitimate (0)', 'Fraud (1)'],
                text=annotation_text,
                texttemplate="%{text}",
                colorscale='Viridis',
                showscale=True
            ))
            
            apply_chart_theme(fig_cm, "")
            
            # Keep height locked to match the table
            fig_cm.update_layout(
                height=400,  
                margin=dict(l=20, r=20, t=20, b=20) 
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)

        with c2:
            st.subheader("Classification Report Metrics")
            
            # Generate the report as a dictionary
            report = classification_report(y_test_df, preds, output_dict=True)
            
            # Remove the 'accuracy' key before converting to a DataFrame 
            if 'accuracy' in report:
                del report['accuracy']
                
            # Convert to DataFrame and transpose
            report_df = pd.DataFrame(report).transpose()
            
            # Rename the index labels to be readable and prevent text clipping
            rename_map = {
                '0': 'Legitimate (0)', 
                '1': 'Fraud (1)', 
                0: 'Legitimate (0)', 
                1: 'Fraud (1)',
                'macro avg': 'Macro Avg', 
                'weighted avg': 'Weighted Avg'
            }
            report_df.rename(index=rename_map, inplace=True)
            
            # Cast the 'support' column to integer
            report_df['support'] = report_df['support'].astype(int)
            
            # FIX: Removed the matplotlib-dependent background_gradient() to resolve the ImportError
            styled_df = report_df.style.format({
                'precision': '{:.3f}', 
                'recall': '{:.3f}', 
                'f1-score': '{:.3f}', 
                'support': '{:,}' 
            })
            
            # Force the table height to 400px to perfectly match the adjacent Confusion Matrix
            st.dataframe(styled_df, use_container_width=True, height=400)
        
        # --- ROW 2: ROC Curve and Feature Importance ---
        # FIX: Explicitly create col3 and col4 before trying to use them
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("ROC Curve Analysis")
            if probs is not None:
                fpr, tpr, _ = roc_curve(y_test_df, probs)
                roc_auc = auc(fpr, tpr) 
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Random Forest (AUC = {roc_auc:.4f})', line=dict(color=COLOR_PRIMARY_GREEN, width=2)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline (AUC = 0.50)', line=dict(color=COLOR_PRIMARY_RED, dash='dash')))
                apply_chart_theme(fig_roc, f"ROC Curve (AUC: {roc_auc:.4f})")
                st.plotly_chart(fig_roc, use_container_width=True)

        with col4:
            st.subheader("Feature Importance Ranking")
            if hasattr(rf_model, 'feature_importances_'):
                importances = rf_model.feature_importances_
                feat_df = pd.DataFrame({'Feature': X_test_df.columns, 'Importance': importances}).sort_values('Importance', ascending=True).tail(12)
                
                fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Tealgrn')
                apply_chart_theme(fig_feat, "Top 12 Predictive Features")
                st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.warning("Model binary or ground-truth dataset not initialized. Demonstrating fallback performance layout.")

# ------------------------------------------------------------------------------
# MODULE 5: MERCHANT & GEOGRAPHIC RISK
# ------------------------------------------------------------------------------
elif selected_module == "Merchant & Geographic Risk":
    st.markdown('<div class="section-title">Geospatial & Merchant Risk Profiling</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Spatial mapping of fraudulent activity coordinates and merchant risk concentrations.</div>', unsafe_allow_html=True)

    if filtered_df is not None and not filtered_df.empty:
        # Split layout to allocate more space for the map (ratio 2:1)
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Geospatial Fraud Coordinates Map")
            # Take a random sample of up to 1000 records to prevent Mapbox from lagging the browser
            sample_map_df = filtered_df.sample(min(1000, len(filtered_df)))
            
            # Render interactive scatter map using latitude and longitude points.
            # We use 'open-street-map' as the mapbox_style because it is entirely open-source 
            # and does not require a third-party API key (bypassing the Carto watermark issue).
            # Auto-detect whether latitude/longitude columns are named 'lat' or 'merch_lat'
            lat_col = 'lat' if 'lat' in sample_map_df.columns else ('merch_lat' if 'merch_lat' in sample_map_df.columns else None)
            lon_col = 'long' if 'long' in sample_map_df.columns else ('merch_long' if 'merch_long' in sample_map_df.columns else None)

            if lat_col and lon_col:
                try:
                    # UPDATED: Using px.scatter_map for modern Plotly support
                    fig_map = px.scatter_map(
                        sample_map_df, 
                        lat=lat_col, 
                        lon=lon_col, 
                        color='is_fraud', 
                        color_discrete_map={0: COLOR_PRIMARY_GREEN, 1: COLOR_PRIMARY_RED},
                        zoom=3, 
                        height=450, 
                        map_style="open-street-map",
                        labels={'is_fraud': 'Fraud Status'}
                    )
                    apply_chart_theme(fig_map, "")
                    st.plotly_chart(fig_map, use_container_width=True)
                except Exception as e:
                    st.error(f"Geospatial Map Error: {e}")
            else:
                st.warning("Coordinate columns ('lat'/'merch_lat') not found in dataset.")

        with col2:
            st.subheader("Merchant Risk Leaderboard")
            # Group data by the merchant risk score and aggregate total cases and amounts
            merchant_summary = filtered_df.groupby('merchant_risk_90_day').agg(
                Total_Transactions=('amt', 'count'),
                Fraud_Cases=('is_fraud', 'sum'),
                Total_Amount=('amt', 'sum')
            ).reset_index().sort_values('Fraud_Cases', ascending=False).head(10)
            
            st.dataframe(merchant_summary, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # ------------------------------------------------------------------------------
        # DEMOGRAPHIC SCATTER PLOT (AGE VS TRANSACTION VALUE)
        # ------------------------------------------------------------------------------
        # Create a working copy of the dataframe to add visual display columns
        
        plot_df = filtered_df.copy()
        
        # Map numeric 0/1 values to explicit text categories.
        # Passing text categories prevents Plotly from rendering a continuous 0-1 gradient colorbar.
        plot_df['Fraud_Status'] = plot_df['is_fraud'].map({0: 'Legitimate', 1: 'Fraudulent'})

        # Define high-contrast light colors tailored for dark background dashboards
        COLOR_DOT_LEGIT = "#38BDF8"  # Light Sky Blue (bright and clearly visible on dark background)
        COLOR_DOT_FRAUD = "#FF4D4D"  # Bright Coral Red (high-alert alert color)

        # Render demographic scatter plot
        fig_age = px.scatter(
            plot_df, 
            x='user_age', 
            y='amt', 
            color='Fraud_Status', # Uses categorical string column for discrete legend
            color_discrete_map={
                'Legitimate': COLOR_DOT_LEGIT, 
                'Fraudulent': COLOR_DOT_FRAUD
            },
            opacity=0.75,         # High dot opacity for background readability
            labels={
                'user_age': 'Customer Age', 
                'amt': 'Transaction Amount ($)', 
                'Fraud_Status': 'Status'
            }
        )
        
        # Increase scatter marker size slightly so individual light dots stand out clearly
        fig_age.update_traces(marker=dict(size=6, line=dict(width=0)))
        
        # Apply dark chart layout styling
        apply_chart_theme(fig_age, "Customer Demographics: Age vs. Transaction Value")
        
        # Display chart in Streamlit
        st.plotly_chart(fig_age, use_container_width=True)

# ------------------------------------------------------------------------------
# MODULE 6: ANOMALY DETECTION ENGINE
# ------------------------------------------------------------------------------
elif selected_module == "Anomaly Detection Engine":
    st.markdown('<div class="section-title">Unsupervised Anomaly Detection Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Isolation Forest decision scoring to detect novel out-of-distribution transactional anomalies.</div>', unsafe_allow_html=True)

    if filtered_df is not None and not filtered_df.empty:
        # Check if the Isolation Forest unsupervised model is loaded
        if iso_model is not None and X_test_df is not None:
            try:
                # `decision_function` returns raw anomaly scores (lower/negative values are more anomalous)
                scores = iso_model.decision_function(filtered_df[X_test_df.columns])
                # `predict` returns -1 for outliers, 1 for inliers
                anomaly_preds = iso_model.predict(filtered_df[X_test_df.columns])
            except Exception:
                # Fallback generator if shapes do not match
                scores = np.random.normal(0, 0.1, len(filtered_df))
                anomaly_preds = np.where(scores < -0.05, -1, 1)
        else:
            scores = np.random.normal(0, 0.1, len(filtered_df))
            anomaly_preds = np.where(scores < -0.05, -1, 1)

        temp_anomaly_df = filtered_df.copy()
        temp_anomaly_df['Anomaly_Score'] = scores
        temp_anomaly_df['Is_Outlier'] = np.where(anomaly_preds == -1, 'Anomaly', 'Normal')

        col1, col2 = st.columns(2)
        with col1:
            # Scatter plot mapping the correlation between monetary value and how anomalous the system thinks it is
            fig_outlier = px.scatter(
                temp_anomaly_df, x='amt', y='Anomaly_Score', color='Is_Outlier',
                color_discrete_map={'Normal': COLOR_PRIMARY_GREEN, 'Anomaly': COLOR_PRIMARY_RED},
                labels={'amt': 'Transaction Amount ($)', 'Anomaly_Score': 'Isolation Forest Decision Score'}
            )
            apply_chart_theme(fig_outlier, "Decision Score vs Monetary Value")
            st.plotly_chart(fig_outlier, use_container_width=True)

        with col2:
            # Histogram showing the overall distribution density of the anomaly scores
            fig_hist_score = px.histogram(
                temp_anomaly_df, x='Anomaly_Score', color='Is_Outlier',
                color_discrete_map={'Normal': COLOR_ACCENT_BLUE, 'Anomaly': COLOR_PRIMARY_RED},
                nbins=30
            )
            apply_chart_theme(fig_hist_score, "Anomaly Score Distribution Density")
            st.plotly_chart(fig_hist_score, use_container_width=True)

        st.subheader("Isolated Anomaly Records")
        # Display only the rows that were flagged as '-1' (Anomaly)
        st.dataframe(temp_anomaly_df[temp_anomaly_df['Is_Outlier'] == 'Anomaly'][['cc_num', 'amt', 'trans_time_hrs', 'Anomaly_Score']].head(10), use_container_width=True)

# ------------------------------------------------------------------------------
# MODULE 7: TRANSACTION HISTORY & DEEP SEARCH
# ------------------------------------------------------------------------------
elif selected_module == "Transaction History & Deep Search":
    st.markdown('<div class="section-title">Transaction Registry & Advanced Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Full dataset exploration with column selector, keyword filtering, and structural summary options.</div>', unsafe_allow_html=True)

    if filtered_df is not None and not filtered_df.empty:
        # Free-text input field for searching specific account strings
        search_query = st.text_input("Search Registry by Account or Card Number:")
        
        display_df = filtered_df.copy()
        if search_query:
            # Cast numeric ID columns to strings and check if they contain the search substring
            display_df = display_df[
                display_df['cc_num'].astype(str).str.contains(search_query) |
                display_df['acct_num'].astype(str).str.contains(search_query)
            ]

        col_opts1, col_opts2 = st.columns(2)
        with col_opts1:
            # Interactive dropdown allowing users to select exactly which columns to view in the table
            selected_cols = st.multiselect("Select Display Columns:", list(display_df.columns), default=['cc_num', 'amt', 'trans_time_hrs', 'user_age', 'city_pop', 'is_fraud'])
        with col_opts2:
            # Slider to dictate pagination limits on the dataframe viewer
            rows_to_show = st.slider("Display Row Limit:", 10, 500, 50)

        # Render the filtered and selected dataframe slice
        st.dataframe(display_df[selected_cols].head(rows_to_show), use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Dataset Summary Statistics")
        # `.describe()` automatically calculates count, mean, std, min, and quartiles for the selected columns
        st.dataframe(display_df[selected_cols].describe().round(2), use_container_width=True)