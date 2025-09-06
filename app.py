import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Configuration ---
st.set_page_config(
    page_title="MF to Wellness Upsell Analysis",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions ---

@st.cache_data
def load_data(file):
    """Loads and caches the excel data to improve performance."""
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error loading the file: {e}")
        return None

def preprocess_data(df):
    """Cleans and prepares the data for analysis."""
    # Convert DATE to datetime
    df['DATE'] = pd.to_datetime(df['DATE'])
    # Set DOW as categorical with a specific order for proper sorting
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['DOW'] = pd.Categorical(df['DOW'], categories=dow_order, ordered=True)
    
    # Rename columns for easier access
    df.columns = [
        'Date', 'DOW', 'City Walk_MF', 'Index_MF', 'DKP_MF', 'Total_MF',
        'City Walk_Wellness', 'Index_Wellness', 'DKP_Wellness', 'Total_Wellness'
    ]
    return df

@st.cache_data
def calculate_lagged_correlation(df, mf_col, wellness_col, max_lag):
    """Calculates correlations between MF and Wellness for a given range of lags."""
    correlations = {}
    for lag in range(-max_lag, max_lag + 1):
        # Shift the MF data by the lag amount
        # A positive lag means MF from 'lag' days ago
        # A negative lag means MF from 'lag' days in the future
        mf_shifted = df[mf_col].shift(lag)
        
        # Calculate correlation between current Wellness and shifted MF
        # and drop NA values which result from shifting
        corr = df[wellness_col].corr(mf_shifted)
        correlations[lag] = corr
        
    lag_df = pd.DataFrame(list(correlations.items()), columns=['Lag (Days)', 'Correlation'])
    return lag_df.dropna()

def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode('utf-8')

# --- Main Application ---

# Title
st.title("📈 MF to Wellness Upsell Performance Dashboard")
st.markdown("An interactive dashboard to analyze the relationship between the Main Service (MF) and the upsell service (Wellness) across branches.")

# File Uploader in Sidebar
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file (dash.xlsx)", type=["xlsx"])

if uploaded_file is None:
    st.info("Please upload your `dash.xlsx` file using the sidebar to begin analysis.")
    st.stop()

# --- Load and Process Data ---
df_raw = load_data(uploaded_file)
if df_raw is not None:
    df = preprocess_data(df_raw.copy())

    # --- Sidebar Controls ---
    st.sidebar.header("Dashboard Filters")
    
    # Branch Selection
    branch_options = {
        'City Walk': ('City Walk_MF', 'City Walk_Wellness'),
        'Index': ('Index_MF', 'Index_Wellness'),
        'DKP': ('DKP_MF', 'DKP_Wellness'),
        'Total': ('Total_MF', 'Total_Wellness')
    }
    selected_branch_name = st.sidebar.selectbox(
        "Select a Branch to Analyze",
        options=list(branch_options.keys())
    )
    mf_col, wellness_col = branch_options[selected_branch_name]

    # Rolling Average Toggle
    use_rolling_avg = st.sidebar.checkbox("Apply 7-Day Rolling Average Smoothing", value=True)

    # --- Data Preparation for Analysis ---
    analysis_df = df[['Date', 'DOW', mf_col, wellness_col]].copy()
    
    # Apply rolling average if selected
    if use_rolling_avg:
        analysis_df[mf_col] = analysis_df[mf_col].rolling(window=7, min_periods=1).mean()
        analysis_df[wellness_col] = analysis_df[wellness_col].rolling(window=7, min_periods=1).mean()
        st.sidebar.info("7-day smoothing is applied to time series and scatter plots to reduce noise and highlight trends.")

    # Calculate Upsell Rate (avoid division by zero)
    analysis_df['Upsell_Rate'] = (analysis_df[wellness_col] / analysis_df[mf_col]).replace([np.inf, -np.inf], 0).fillna(0) * 100

    # --- KPI Section ---
    st.header(f"Key Performance Indicators for: {selected_branch_name}")
    
    # Use the original, non-smoothed data for KPIs
    total_mf = df[mf_col].sum()
    total_wellness = df[wellness_col].sum()
    overall_upsell_rate = (total_wellness / total_mf) * 100 if total_mf > 0 else 0
    same_day_corr = df[mf_col].corr(df[wellness_col])

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(label="Total MF (Main Service)", value=f"{total_mf:,.0f}")
    kpi2.metric(label="Total Wellness (Upsell)", value=f"{total_wellness:,.0f}")
    kpi3.metric(label="Overall Upsell Rate", value=f"{overall_upsell_rate:.2f}%")
    kpi4.metric(label="Same-Day Correlation (r)", value=f"{same_day_corr:.3f}")
    st.markdown("---")


    # --- Visualizations ---
    st.header("Visual Analysis")
    
    # Time Series Plot
    st.subheader("MF vs. Wellness Over Time")
    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ts.add_trace(go.Scatter(x=analysis_df['Date'], y=analysis_df[mf_col], name='MF (Main Service)', mode='lines', line=dict(color='#1f77b4')), secondary_y=False)
    fig_ts.add_trace(go.Scatter(x=analysis_df['Date'], y=analysis_df[wellness_col], name='Wellness (Upsell)', mode='lines', line=dict(color='#ff7f0e')), secondary_y=True)
    fig_ts.update_layout(
        title_text=f"Daily MF and Wellness Volumes for {selected_branch_name}" + (" (7-Day Smoothed)" if use_rolling_avg else ""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_ts.update_yaxes(title_text="MF (Main Service) Volume", secondary_y=False)
    fig_ts.update_yaxes(title_text="Wellness (Upsell) Volume", secondary_y=True)
    st.plotly_chart(fig_ts, use_container_width=True)

    # Scatter Plot
    st.subheader("Wellness vs. MF Relationship")
    fig_scatter = px.scatter(
        analysis_df,
        x=mf_col,
        y=wellness_col,
        trendline="ols",
        trendline_color_override="red",
        title=f"Correlation between Wellness and MF for {selected_branch_name}" + (" (7-Day Smoothed)" if use_rolling_avg else ""),
        labels={mf_col: "MF (Main Service) Volume", wellness_col: "Wellness (Upsell) Volume"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown("The red line represents the Ordinary Least Squares (OLS) regression trendline, indicating the general relationship between the two services.")
    st.markdown("---")
    
    # --- Advanced Analysis ---
    st.header("Advanced Correlation & Day-of-Week Analysis")
    
    col1, col2 = st.columns(2)

    with col1:
        # Lagged Correlation Analysis
        st.subheader("Lagged Correlation Analysis")
        max_lag = st.slider("Select Max Lag Window (Days)", 1, 30, 14, help="Adjust the window to check if MF today predicts Wellness in the future (positive lag) or vice-versa.")
        
        # We use the original, non-smoothed data for lag analysis
        lag_df = calculate_lagged_correlation(df, mf_col, wellness_col, max_lag)
        
        fig_lag = px.bar(
            lag_df,
            x='Lag (Days)',
            y='Correlation',
            title=f"MF vs. Wellness Correlation at Different Lags",
            labels={'Lag (Days)': 'Lag (MF leads Wellness for positive lags)', 'Correlation': 'Pearson Correlation (r)'},
            color='Correlation',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        fig_lag.add_hline(y=0, line_dash="dash", line_color="grey")
        st.plotly_chart(fig_lag, use_container_width=True)
        st.markdown("""
        **How to read this chart:**
        - **Positive Lag (e.g., +7):** Correlation between today's Wellness and MF from 7 days *ago*. A high positive correlation here suggests MF increases lead to Wellness increases later.
        - **Negative Lag (e.g., -3):** Correlation between today's Wellness and MF 3 days *in the future*. This is less common but could indicate external factors driving both.
        - **Lag 0:** The same-day correlation.
        """)
        
    with col2:
        # Day-of-Week Analysis
        st.subheader("Day-of-Week Performance")
        dow_summary = df.groupby('DOW')[[mf_col, wellness_col]].mean().reset_index()
        dow_summary['Upsell_Rate'] = (dow_summary[wellness_col] / dow_summary[mf_col]) * 100
        
        fig_dow = px.bar(
            dow_summary,
            x='DOW',
            y='Upsell_Rate',
            title=f'Average Upsell Rate by Day of the Week ({selected_branch_name})',
            labels={'DOW': 'Day of the Week', 'Upsell_Rate': 'Upsell Rate (%)'},
            color='Upsell_Rate',
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_dow, use_container_width=True)

        st.dataframe(dow_summary.round(2), use_container_width=True)
    st.markdown("---")

    # --- Data Export ---
    st.header("Export Data")
    st.markdown("Download the processed data or analysis results for offline use.")

    export_col1, export_col2 = st.columns(2)
    with export_col1:
        # Prepare processed data for download
        processed_data_csv = convert_df_to_csv(analysis_df)
        st.download_button(
            label="📥 Download Processed Data as CSV",
            data=processed_data_csv,
            file_name=f"{selected_branch_name}_processed_data.csv",
            mime="text/csv",
        )
    with export_col2:
        # Prepare lag data for download
        lag_data_csv = convert_df_to_csv(lag_df)
        st.download_button(
            label="📥 Download Lag Correlation Data as CSV",
            data=lag_data_csv,
            file_name=f"{selected_branch_name}_lag_correlation.csv",
            mime="text/csv",
        )
