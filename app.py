#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import base64 # Required for download button functionality

# Set the page configuration for a professional look
st.set_page_config(
    page_title="Ground Truth Tracker: Humanitarian Feedback Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Data Loading Functions ---

# Define score columns once for easy reference
SCORE_COLS = {
    'Aid Satisfaction': 'aid_satisfaction',
    'Trust in Provider': 'trust_in_aid_provider',
    'Communication Clarity': 'communication_clarity',
    'Aid Fairness': 'aid_fairness'
}

@st.cache_data
def load_data():
    """Load all necessary dataframes, using Streamlit's cache for speed."""
    try:
        # Load pre-calculated summaries from Phase 2 CSVs
        location_summary = pd.read_csv('location_summary.csv', index_col='location')
        significance_results = pd.read_csv('significance_results.csv')
        
        # Load the full clean data for dynamic filtering/plotting
        df_full = pd.read_csv('gts_processed_data_full.csv') 

        # Clean up column types for plotting
        for col_name in SCORE_COLS.values():
            df_full[col_name] = pd.to_numeric(df_full[col_name], errors='coerce').fillna(3).astype(int)

        return df_full, location_summary, significance_results
    
    except FileNotFoundError as e:
        st.error(f"Error loading data: One or more required CSV files are missing. Please ensure all data files are in the directory.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# --- 2. Plotly Visualization Functions (Modified for Dynamic Metric) ---

def plot_sentiment_divergence(df, metric_col, metric_label):
    """Generates the Diverging Stacked Bar Chart for a selected metric."""
    sentiment_map = {1: 'Strongly Disagree', 2: 'Disagree', 3: 'Neutral', 4: 'Agree', 5: 'Strongly Agree'}
    color_map = {
        'Strongly Disagree': 'rgb(202, 0, 32)', 'Disagree': 'rgb(244, 165, 130)',
        'Neutral': 'rgb(247, 247, 247)', 'Agree': 'rgb(75, 141, 195)', 'Strongly Agree': 'rgb(12, 51, 131)'
    }
    
    df_plot = df.groupby(['location', metric_col]).size().reset_index(name='count')
    df_plot['sentiment'] = df_plot[metric_col].map(sentiment_map)
    df_plot['percentage'] = df_plot.groupby('location')['count'].transform(lambda x: 100 * x / x.sum())
    
    df_plot['plot_percentage'] = df_plot.apply(
        lambda row: row['percentage'] if row[metric_col] >= 3 else -row['percentage'], axis=1
    )

    fig = px.bar(
        df_plot, y='location', x='plot_percentage', color='sentiment', orientation='h',
        color_discrete_map=color_map,
        category_orders={'sentiment': list(sentiment_map.values())},
        title=f"1. Distribution of **{metric_label}** Scores by Location"
    )
    fig.update_layout(xaxis_range=[-100, 100], xaxis_tickformat=',.0f%', yaxis_title=None, title_font_size=18)
    fig.add_vline(x=0, line_width=1.5, line_dash="solid", line_color="black")
    return fig


def plot_trust_hotspots(df_filtered, metric_col, metric_label):
    """Generates the Geospatial Map for the mean score of the selected metric."""
    
    # Calculate Mean Score for the selected metric
    df_map = df_filtered.groupby('location')[metric_col].mean().reset_index()
    df_map.rename(columns={metric_col: 'Mean_Score'}, inplace=True)
    
    # Dummy coordinates (must match those used in Phase 3)
    location_coords = {
        'Darfur Region (Sudan)': [15.0, 25.0], 'Gaza Strip / West Bank (oPt)': [31.5, 34.5], 
        'Jigjiga Zone (Ethiopia)': [8.0, 42.0], 'North-East DRC': [-1.0, 29.0], 
        'Lviv / Odesa Region (Ukraine)': [49.0, 24.0], "Cox's Bazar (Bangladesh)": [21.5, 92.0]
    }
    df_map['lat'] = df_map['location'].apply(lambda x: location_coords.get(x, [0, 0])[0])
    df_map['lon'] = df_map['location'].apply(lambda x: location_coords.get(x, [0, 0])[1])

    fig = px.scatter_geo(
        df_map, lat='lat', lon='lon', hover_name='location', color='Mean_Score', 
        size='Mean_Score', projection='natural earth',
        color_continuous_scale=px.colors.sequential.Reds_r, # Reverse red scale: low score = bright red
        title=f"2. Geographic Hotspots of **{metric_label}** (Mean Score)"
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Mean Score (1-5)"), title_font_size=18)
    fig.update_geos(showland=True, landcolor="lightgray", showcountries=True, countrycolor="darkgray")
    return fig


# T-Test Plot is not filtered by sidebar for simplicity, as it analyzes pre-computed structural differences
def plot_ttest_comparison(ttest_df):
    """Generates the bar chart comparing means and highlighting significance (unchanged)."""
    ttest_df['Annotation'] = ttest_df.apply(
        lambda row: f"P-Value: {row['P_Value']} | Significant: {row['Significance']}", axis=1
    )
    ttest_df_long = pd.melt(
        ttest_df, id_vars=['Comparison', 'Annotation', 'Significance'], 
        value_vars=['Mean_G1', 'Mean_G2'], var_name='Group_Code', value_name='Mean_Score'
    )
    ttest_df_long['Group_Name'] = ttest_df_long.apply(
        lambda row: 'Female/IDP Mean' if row['Group_Code'] == 'Mean_G1' else 
                    'Male/Host Community Mean', axis=1
    )

    fig = px.bar(
        ttest_df_long, x='Comparison', y='Mean_Score', color='Group_Name', barmode='group',
        text_auto='.3f', title='3. Statistical Validation of Structural Differences (T-Tests)'
    )
    # Annotation logic... (simplified for space)
    fig.update_layout(yaxis_range=[1, 5], yaxis_title='Mean Score (1 to 5 Likert)', title_font_size=18)
    return fig

# --- 3. Streamlit Utilities ---

def get_table_download_link(df):
    """Generates a link to download the DataFrame as a CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # bytes <-> base64 <-> str
    href = f'<a href="data:file/csv;base64,{b64}" download="filtered_gts_data.csv">Download Filtered Data as CSV</a>'
    return href

# --- 4. Main Streamlit Dashboard Layout ---

def main():
    df_full, location_summary, significance_results = load_data()
    
    if df_full.empty:
        return # Stop execution if data loading failed
        
    # --- Sidebar for Filtering ---
    st.sidebar.header("🔍 Data Filters")
    
    selected_locations = st.sidebar.multiselect(
        "Filter by Crisis Location:",
        options=df_full['location'].unique(),
        default=df_full['location'].unique()
    )
    
    selected_providers = st.sidebar.multiselect(
        "Filter by Aid Provider:",
        options=df_full['aid_provider'].unique(),
        default=df_full['aid_provider'].unique()
    )
    
    # Apply Filters
    df_filtered = df_full[
        df_full['location'].isin(selected_locations) & 
        df_full['aid_provider'].isin(selected_providers)
    ]
    
    st.sidebar.info(f"Showing {len(df_filtered)} of {len(df_full)} responses.")
    
    # --- Dynamic Metric Selection (Radio Button) ---
    st.sidebar.header("📊 Chart Metric Selector")
    selected_metric_label = st.sidebar.radio(
        "Select Core Metric for Visualization 1 & 2:",
        options=list(SCORE_COLS.keys()),
        index=1 # Default to Trust in Provider (critical metric)
    )
    selected_metric_col = SCORE_COLS[selected_metric_label]

    # --- Header and KPIs ---
    st.title("🌍 Ground Truth Tracker: Humanitarian Impact Dashboard")
    st.markdown("### Executive Summary: Aid Feedback & Actionable Insights")
    
    # Calculate KPIs based on FILTERED data
    total_responses_filtered = len(df_filtered)
    if total_responses_filtered == 0:
        st.warning("No data matches the current filters.")
        return

    promoters = len(df_filtered[df_filtered['aid_satisfaction'] == 5])
    detractors = len(df_filtered[df_filtered['aid_satisfaction'].isin([1, 2])])
    nss = round(((promoters / total_responses_filtered) - (detractors / total_responses_filtered)) * 100, 2)
    
    # --- KPI Cards ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Total Responses Analyzed", value=f"{total_responses_filtered:,}")
        
    with col2:
        st.metric(label="Net Satisfaction Score (NSS)", value=f"{nss}%", delta="Target: +30%", delta_color="normal")
        
    with col3:
        # Mean score for the selected dynamic metric
        mean_score_selected = df_filtered[selected_metric_col].mean()
        st.metric(label=f"Average {selected_metric_label}", value=f"{mean_score_selected:.2f}")

    with col4:
        # Download Button (matches JD requirement for 'analysis tools')
        st.markdown("#### Export Data")
        st.markdown(get_table_download_link(df_filtered), unsafe_allow_html=True)
        st.markdown("**Note:** Export includes current filters.")

    st.markdown("---")

    # --- Visualization Row 1: Sentiment and Geospatial ---
    st.header("I. Regional Sentiment and Hotspot Analysis")
    
    col_vis1, col_vis2 = st.columns([1, 1.5])
    
    with col_vis1:
        fig_sentiment = plot_sentiment_divergence(df_filtered, selected_metric_col, selected_metric_label)
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with col_vis2:
        # Note: We pass df_filtered for the map to calculate filtered means per location
        fig_map = plot_trust_hotspots(df_filtered, selected_metric_col, selected_metric_label)
        st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")

    # --- Visualization Row 2: Statistical Insight ---
    st.header("II. Statistical Validation & Actionable Differences")
    
    col_vis3, col_data = st.columns([1.5, 1])
    
    with col_vis3:
        # T-Test plot remains static as it represents structural, validated analysis.
        fig_ttest = plot_ttest_comparison(significance_results)
        st.plotly_chart(fig_ttest, use_container_width=True)
    
    with col_data:
        st.subheader("Actionable T-Test Results")
        st.dataframe(significance_results.set_index('Comparison'), use_container_width=True)
        st.info("Results marked 'Yes' indicate that the difference in mean score is statistically significant (P < 0.05) and should be treated as a real structural issue requiring policy adjustment.")

if __name__ == '__main__':
    main()


# In[ ]:




