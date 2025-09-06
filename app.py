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
        'All Branches Comparison': ('All', 'All'), # New comparison option
        'City Walk': ('City Walk_MF', 'City Walk_Wellness'),
        'Index': ('Index_MF', 'Index_Wellness'),
        'DKP': ('DKP_MF', 'DKP_Wellness'),
        'Total': ('Total_MF', 'Total_Wellness')
    }
    selected_view = st.sidebar.selectbox(
        "Select a View",
        options=list(branch_options.keys())
    )

    # Rolling Average Toggle
    use_rolling_avg = st.sidebar.checkbox("Apply 7-Day Rolling Average Smoothing", value=True)

    # --- Main Dashboard Logic ---
    if selected_view == 'All Branches Comparison':
        st.header("All Branches Comparison")
        st.markdown("This view provides a side-by-side comparison of City Walk, Index, and DKP branches.")
        if use_rolling_avg:
            st.sidebar.info("7-day smoothing is applied to the charts to reduce noise and highlight trends.")
        
        # --- Comparison Time Series Plot ---
        st.subheader("MF and Wellness Over Time (All Branches)")
        fig_ts_comp = go.Figure()
        branches = ['City Walk', 'Index', 'DKP']
        colors = px.colors.qualitative.Plotly
        
        for i, branch in enumerate(branches):
            mf_col_comp = f"{branch}_MF"
            wellness_col_comp = f"{branch}_Wellness"
            
            mf_data = df[mf_col_comp].rolling(window=7).mean() if use_rolling_avg else df[mf_col_comp]
            wellness_data = df[wellness_col_comp].rolling(window=7).mean() if use_rolling_avg else df[wellness_col_comp]
            
            fig_ts_comp.add_trace(go.Scatter(x=df['Date'], y=mf_data, name=f'{branch} MF', mode='lines', line=dict(color=colors[i]), legendgroup=branch))
            fig_ts_comp.add_trace(go.Scatter(x=df['Date'], y=wellness_data, name=f'{branch} Wellness', mode='lines', line=dict(color=colors[i], dash='dash'), legendgroup=branch))

        fig_ts_comp.update_layout(
            title_text="Daily MF and Wellness Volumes" + (" (7-Day Smoothed)" if use_rolling_avg else ""),
            legend_title_text='Branch & Service',
            yaxis_title="Volume"
        )
        st.plotly_chart(fig_ts_comp, use_container_width=True)

        # --- Comparison Scatter Plot ---
        st.subheader("Wellness vs. MF Relationship (All Branches)")
        # Reshape data from wide to long format for plotting
        df_long = pd.melt(df, id_vars=['Date'], value_vars=['City Walk_MF', 'Index_MF', 'DKP_MF', 'City Walk_Wellness', 'Index_Wellness', 'DKP_Wellness'], var_name='Metric', value_name='Value')
        df_long[['Branch', 'Service']] = df_long['Metric'].str.split('_', expand=True)
        df_pivoted = df_long.pivot_table(index=['Date', 'Branch'], columns='Service', values='Value').reset_index()

        if use_rolling_avg:
            df_pivoted['MF'] = df_pivoted.groupby('Branch')['MF'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
            df_pivoted['Wellness'] = df_pivoted.groupby('Branch')['Wellness'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

        fig_scatter_comp = px.scatter(
            df_pivoted.dropna(), x="MF", y="Wellness", color="Branch",
            title="Wellness vs. MF by Branch" + (" (7-Day Smoothed)" if use_rolling_avg else ""),
            labels={"MF": "MF (Main Service) Volume", "Wellness": "Wellness (Upsell) Volume"}
        )
        st.plotly_chart(fig_scatter_comp, use_container_width=True)
        st.markdown("---")
        
        # --- Comparison Day-of-Week Analysis ---
        st.subheader("Average Upsell Rate by Day of the Week")
        all_dow_summary = []
        for branch in ['City Walk', 'Index', 'DKP']:
            mf_col_comp = f"{branch}_MF"
            wellness_col_comp = f"{branch}_Wellness"
            
            dow_summary = df.groupby('DOW', observed=True)[[mf_col_comp, wellness_col_comp]].mean().reset_index()
            dow_summary['Upsell_Rate'] = (dow_summary[wellness_col_comp] / dow_summary[mf_col_comp]) * 100
            dow_summary['Branch'] = branch
            all_dow_summary.append(dow_summary)
            
        combined_dow_df = pd.concat(all_dow_summary)
        
        fig_dow_comp = px.bar(
            combined_dow_df, x='DOW', y='Upsell_Rate', color='Branch', barmode='group',
            title='Average Upsell Rate by Day and Branch',
            labels={'DOW': 'Day of the Week', 'Upsell_Rate': 'Upsell Rate (%)'}
        )
        st.plotly_chart(fig_dow_comp, use_container_width=True)
        st.markdown("---")

        # --- Data Export ---
        st.header("Export Data")
        comparison_csv = convert_df_to_csv(df_pivoted)
        st.download_button(label="📥 Download Comparison Data as CSV", data=comparison_csv, file_name="branch_comparison_data.csv", mime="text/csv")

    else: # --- Single Branch/Total View ---
        mf_col, wellness_col = branch_options[selected_view]
        analysis_df = df[['Date', 'DOW', mf_col, wellness_col]].copy()
        
        if use_rolling_avg:
            analysis_df[mf_col] = analysis_df[mf_col].rolling(window=7, min_periods=1).mean()
            analysis_df[wellness_col] = analysis_df[wellness_col].rolling(window=7, min_periods=1).mean()
            st.sidebar.info("7-day smoothing is applied to time series and scatter plots.")

        analysis_df['Upsell_Rate'] = (analysis_df[wellness_col] / analysis_df[mf_col]).replace([np.inf, -np.inf], 0).fillna(0) * 100

        st.header(f"Key Performance Indicators for: {selected_view}")
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

        st.header("Visual Analysis")
        st.subheader("MF vs. Wellness Over Time")
        fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ts.add_trace(go.Scatter(x=analysis_df['Date'], y=analysis_df[mf_col], name='MF (Main Service)', mode='lines', line=dict(color='#1f77b4')), secondary_y=False)
        fig_ts.add_trace(go.Scatter(x=analysis_df['Date'], y=analysis_df[wellness_col], name='Wellness (Upsell)', mode='lines', line=dict(color='#ff7f0e')), secondary_y=True)
        fig_ts.update_layout(title_text=f"Daily MF and Wellness Volumes for {selected_view}" + (" (7-Day Smoothed)" if use_rolling_avg else ""), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_ts.update_yaxes(title_text="MF (Main Service) Volume", secondary_y=False)
        fig_ts.update_yaxes(title_text="Wellness (Upsell) Volume", secondary_y=True)
        st.plotly_chart(fig_ts, use_container_width=True)

        st.subheader("Wellness vs. MF Relationship")
        fig_scatter = px.scatter(analysis_df, x=mf_col, y=wellness_col, title=f"Correlation for {selected_view}" + (" (7-Day Smoothed)" if use_rolling_avg else ""), labels={mf_col: "MF Volume", wellness_col: "Wellness Volume"})
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("---")
        
        st.header("Advanced Correlation & Day-of-Week Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Lagged Correlation Analysis")
            max_lag = st.slider("Select Max Lag Window (Days)", 1, 30, 14)
            lag_df = calculate_lagged_correlation(df, mf_col, wellness_col, max_lag)
            fig_lag = px.bar(lag_df, x='Lag (Days)', y='Correlation', title=f"MF vs. Wellness Correlation at Different Lags", color='Correlation', color_continuous_scale=px.colors.sequential.Viridis)
            fig_lag.add_hline(y=0, line_dash="dash", line_color="grey")
            st.plotly_chart(fig_lag, use_container_width=True)
            st.markdown("- **Positive Lag:** Correlation between today's Wellness and MF from X days *ago*. High correlation suggests MF increases lead to Wellness increases later.\n- **Negative Lag:** Correlation between today's Wellness and MF X days *in the future*.")
            
        with col2:
            st.subheader("Day-of-Week Performance")
            dow_summary = df.groupby('DOW', observed=True)[[mf_col, wellness_col]].mean().reset_index()
            dow_summary['Upsell_Rate'] = (dow_summary[wellness_col] / dow_summary[mf_col]) * 100
            fig_dow = px.bar(dow_summary, x='DOW', y='Upsell_Rate', title=f'Average Upsell Rate by Day ({selected_view})', color='Upsell_Rate', color_continuous_scale=px.colors.sequential.Plasma)
            st.plotly_chart(fig_dow, use_container_width=True)
            st.dataframe(dow_summary.round(2), use_container_width=True)
        st.markdown("---")

        st.header("Export Data")
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            processed_data_csv = convert_df_to_csv(analysis_df)
            st.download_button(label="📥 Download Processed Data as CSV", data=processed_data_csv, file_name=f"{selected_view}_processed_data.csv", mime="text/csv")
        with export_col2:
            lag_data_csv = convert_df_to_csv(lag_df)
            st.download_button(label="📥 Download Lag Correlation Data as CSV", data=lag_data_csv, file_name=f"{selected_view}_lag_correlation.csv", mime="text/csv")

