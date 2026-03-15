import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import time

st.set_page_config(
    page_title="FinTrust | AI Analytics",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_premium_ui():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
            
            .stApp {
                background-color: #0b0c10;
                background-image: 
                    radial-gradient(at 0% 0%, rgba(31, 40, 51, 0.5) 0, transparent 50%), 
                    radial-gradient(at 50% 0%, rgba(102, 252, 241, 0.15) 0, transparent 50%);
                color: #c5c6c7;
                font-family: 'Inter', sans-serif;
            }
            h1, h2, h3, h4, h5, h6 { 
                font-family: 'Space Grotesk', sans-serif; 
                color: #ffffff; 
            }
            
            .glass-card {
                background: rgba(31, 40, 51, 0.4);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border-radius: 16px;
                border: 1px solid rgba(102, 252, 241, 0.1);
                padding: 24px;
                margin-bottom: 24px;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
                transition: transform 0.3s ease, border-color 0.3s ease;
            }
            .glass-card:hover {
                border-color: rgba(102, 252, 241, 0.3);
                transform: translateY(-2px);
            }
            
            .hero-banner {
                background: #0b0c10;
                border: 1px solid rgba(102, 252, 241, 0.2);
                padding: 40px;
                border-radius: 24px;
                margin-bottom: 35px;
                position: relative;
                overflow: hidden;
                box-shadow: 0 20px 50px rgba(0,0,0,0.5);
            }
            .hero-banner::before, .hero-banner::after {
                content: '';
                position: absolute;
                top: -100%; left: -100%; width: 300%; height: 300%;
                background: radial-gradient(circle, rgba(102,252,241,0.1) 0%, transparent 40%);
                animation: rotate-mesh 20s linear infinite;
                z-index: 0;
            }
            .hero-banner::after {
                background: radial-gradient(circle, rgba(69,162,158,0.1) 0%, transparent 40%);
                animation: rotate-mesh 15s linear reverse infinite;
                opacity: 0.5;
            }
            @keyframes rotate-mesh {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }

            .shimmer-text {
                background: linear-gradient(90deg, #ffffff, #66fcf1, #ffffff);
                background-size: 200% auto;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: shine 5s linear infinite;
                font-weight: 700;
            }
            @keyframes shine {
                to { background-position: 200% center; }
            }
            
            .kpi-title { font-size: 0.9rem; color: #45a29e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
            .kpi-value { font-size: 2.5rem; font-weight: 700; color: #ffffff; font-family: 'Space Grotesk', sans-serif; }
            .kpi-value.neon { background: -webkit-linear-gradient(45deg, #66fcf1, #45a29e); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            
            .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
            .stTabs [data-baseweb="tab"] {
                background-color: rgba(31, 40, 51, 0.6);
                border-radius: 8px 8px 0 0;
                border: 1px solid rgba(102, 252, 241, 0.1);
                border-bottom: none;
                color: #c5c6c7 !important;
                padding: 12px 24px;
                font-family: 'Space Grotesk', sans-serif;
                transition: all 0.3s ease;
            }
            .stTabs [aria-selected="true"] {
                background: linear-gradient(180deg, rgba(102, 252, 241, 0.1) 0%, rgba(31, 40, 51, 0) 100%);
                border-top: 2px solid #66fcf1;
                color: #ffffff !important;
            }
            .footer-text { color: #45a29e; font-size: 0.85rem; letter-spacing: 1px; }
            
            /* Custom styling for Streamlit selectboxes to fit the dark theme */
            div[data-baseweb="select"] > div { background-color: rgba(31, 40, 51, 0.8); border-color: rgba(102, 252, 241, 0.3); color: white; }
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    try:
        raw_df = pd.read_csv("atm_cash_management_dataset.csv")
        raw_df['Date'] = pd.to_datetime(raw_df['Date'])
        raw_df = raw_df.sort_values(by=['ATM_ID', 'Date'])
        raw_df['Day_of_Week'] = raw_df['Date'].dt.day_name()
        raw_df['Is_Weekend'] = raw_df['Day_of_Week'].isin(['Saturday', 'Sunday']).astype(int)
        raw_df['Rolling_Mean_Withdrawals'] = raw_df.groupby('ATM_ID')['Total_Withdrawals'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        raw_df['Daily_Change_Pct'] = raw_df.groupby('ATM_ID')['Total_Withdrawals'].pct_change().fillna(0) * 100
        raw_df['Daily_Change_Pct'] = raw_df['Daily_Change_Pct'].replace([np.inf, -np.inf], 0)
        return raw_df, True
    except Exception:
        return pd.DataFrame(), False

def render_kpi_card(title, value, is_neon=False):
    neon_class = "neon" if is_neon else ""
    st.markdown(f'<div class="glass-card"><div class="kpi-title">{title}</div><div class="kpi-value {neon_class}">{value}</div></div>', unsafe_allow_html=True)

def create_base_plotly_template():
    custom_template = go.layout.Template()
    custom_template.layout.plot_bgcolor = "rgba(0,0,0,0)"
    custom_template.layout.paper_bgcolor = "rgba(0,0,0,0)"
    custom_template.layout.font.color = "#c5c6c7"
    custom_template.layout.xaxis.gridcolor = "rgba(102,252,241,0.05)"
    custom_template.layout.yaxis.gridcolor = "rgba(102,252,241,0.05)"
    custom_template.layout.xaxis.zerolinecolor = "rgba(102,252,241,0.1)"
    custom_template.layout.yaxis.zerolinecolor = "rgba(102,252,241,0.1)"
    return custom_template

inject_premium_ui()
df, success = load_and_preprocess_data()
base_template = create_base_plotly_template()

with st.sidebar:
    st.markdown('<div style="text-align: center; margin-bottom: 30px;"><h1 class="shimmer-text" style="font-size: 2.2rem; margin:0;">💠 FinTrust</h1><p style="color: #45a29e; font-size: 0.9rem; margin:0; letter-spacing: 2px;">NEURAL CORE</p></div>', unsafe_allow_html=True)
    nav_mode = st.radio("OPERATING LAYER", ["🌐 Global Overview", "📈 Market Dynamics", "⚡ Risk & Diagnostics", "🔮 Predictive Forecasting"], label_visibility="collapsed")
    
    st.markdown("<hr style='border-color: rgba(102,252,241,0.2);'><p style='color: #45a29e; font-size:0.85rem; font-weight:600; margin-bottom:10px;'>DATA STREAMS</p>", unsafe_allow_html=True)
    available_cols = ['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level', 'Nearby_Competitor_ATMs', 'Cash_Demand_Next_Day', 'Rolling_Mean_Withdrawals']
    selected_metrics = st.multiselect("Select vectors for AI analysis:", options=available_cols, default=['Total_Withdrawals', 'Total_Deposits', 'Nearby_Competitor_ATMs'], label_visibility="collapsed")
    
    st.markdown("<hr style='border-color: rgba(102,252,241,0.2);'><p style='color: #45a29e; font-size:0.85rem; font-weight:600; margin-bottom:10px;'>HYPERPARAMETERS</p>", unsafe_allow_html=True)
    k_clusters = st.slider("Segmentation Nodes (K)", 2, 8, 3)
    anomaly_threshold = st.slider("Anomaly Contamination", 0.01, 0.15, 0.05)
    
    st.markdown("<hr style='border-color: rgba(102,252,241,0.2);'><p style='color: #45a29e; font-size:0.85rem; font-weight:600; margin-bottom:10px;'>FLEET FILTER</p>", unsafe_allow_html=True)
    if success:
        unique_locations = df['Location_Type'].unique().tolist()
        selected_locations = st.multiselect("Filter by Location Type", options=unique_locations, default=unique_locations, label_visibility="collapsed")

if not success:
    st.error("System Failure: 'atm_cash_management_dataset.csv' disconnected.")
    st.stop()
if not selected_locations:
    st.warning("Awaiting Location Selection...")
    st.stop()

filtered_df = df[df['Location_Type'].isin(selected_locations)].copy()

st.markdown(
    """
    <div class="hero-banner">
        <div style="position: relative; z-index: 1;">
            <h1 class="shimmer-text" style="margin: 0; font-size: 2.8rem; letter-spacing: -1px;">ATM Fleet Intelligence Suite</h1>
            <p style="color: #45a29e; font-size: 1.1rem; margin: 5px 0 0 0; font-weight: 500;">
                Real-Time Telemetry • Predictive Clustering • Anomaly Isolation
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if nav_mode == "🌐 Global Overview":
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1: render_kpi_card("Active Nodes", str(filtered_df['ATM_ID'].nunique()), True)
    with col_kpi2: render_kpi_card("Mean Throughput", f"${filtered_df['Total_Withdrawals'].mean():,.0f}", False)
    with col_kpi3: render_kpi_card("Max Spike", f"${filtered_df['Total_Withdrawals'].max():,.0f}", False)
    with col_kpi4: render_kpi_card("Active Vectors", str(len(selected_metrics)), True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Fleet Throughput Timeline")
    
    # NEW GRAPH CONTROLS
    ctrl_col1, ctrl_col2 = st.columns([1, 1])
    with ctrl_col1:
        timeline_metric = st.selectbox("Select Metric to Plot:", ['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level'], index=0)
    with ctrl_col2:
        timeline_grouping = st.radio("Display Mode:", ['Aggregate Fleet View', 'Compare by Location'], horizontal=True)
    
    if timeline_grouping == 'Aggregate Fleet View':
        timeline_df = filtered_df.groupby('Date')[timeline_metric].sum().reset_index()
        fig_timeline = px.line(timeline_df, x='Date', y=timeline_metric, line_shape='spline', color_discrete_sequence=['#66fcf1'])
        fig_timeline.update_traces(fill='tozeroy', fillcolor='rgba(102, 252, 241, 0.1)')
    else:
        timeline_df = filtered_df.groupby(['Date', 'Location_Type'])[timeline_metric].sum().reset_index()
        fig_timeline = px.line(timeline_df, x='Date', y=timeline_metric, color='Location_Type', line_shape='spline', color_discrete_sequence=['#66fcf1', '#45a29e', '#1f2833', '#c5c6c7'])
        
    fig_timeline.update_layout(template=base_template, margin=dict(t=10, b=10, l=10, r=10), height=350, hovermode="x unified")
    st.plotly_chart(fig_timeline, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 🗄️ Secure Data Ledger")
    st.dataframe(filtered_df.head(250), use_container_width=True, height=400)

elif nav_mode == "📈 Market Dynamics":
    if len(selected_metrics) < 2:
        st.warning("⚠️ Insufficient Data Vectors. Select at least 2 in the Neural Core.")
    else:
        col_dyn1, col_dyn2 = st.columns([1.2, 1])
        with col_dyn1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🧬 Vector Interaction Matrix")
            corr_matrix = filtered_df[selected_metrics].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='Tealgrn', aspect="auto")
            fig_corr.update_layout(template=base_template, margin=dict(t=30, b=10, l=10, r=10), height=450)
            st.plotly_chart(fig_corr, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_dyn2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Topography by Location")
            
            # NEW GRAPH CONTROL
            boxplot_metric = st.selectbox("Select Topography Metric:", available_cols, index=0)
            
            fig_dist = px.box(filtered_df, y=boxplot_metric, color="Location_Type", color_discrete_sequence=['#66fcf1', '#45a29e', '#1f2833', '#c5c6c7'])
            fig_dist.update_layout(template=base_template, margin=dict(t=10, b=10, l=10, r=10), showlegend=False, height=410)
            st.plotly_chart(fig_dist, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

elif nav_mode == "⚡ Risk & Diagnostics":
    if len(selected_metrics) < 2:
        st.error("⚠️ AI Diagnostic Engine requires at least 2 Data Streams.")
    else:
        with st.spinner("Compiling Neural Models..."):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(filtered_df[selected_metrics])
            kmeans_model = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
            filtered_df['Assigned_Cluster'] = kmeans_model.fit_predict(X_scaled)
            iso_forest = IsolationForest(contamination=anomaly_threshold, random_state=42)
            filtered_df['Anomaly_Signal'] = iso_forest.fit_predict(X_scaled)
            filtered_df['Diagnostic_Status'] = filtered_df['Anomaly_Signal'].map({1: "Nominal", -1: "Critical Spike"})
            
            if len(selected_metrics) > 3:
                pca = PCA(n_components=3)
                pca_components = pca.fit_transform(X_scaled)
                filtered_df['PCA_1'] = pca_components[:, 0]
                filtered_df['PCA_2'] = pca_components[:, 1]
                filtered_df['PCA_3'] = pca_components[:, 2]
                
            time.sleep(0.4)

        tab_cluster, tab_anomaly = st.tabs(["🧩 High-Dimensional Clustering", "🚨 Anomaly Isolation"])
        with tab_cluster:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            if len(selected_metrics) > 3:
                fig_cluster = px.scatter_3d(filtered_df, x='PCA_1', y='PCA_2', z='PCA_3', color=filtered_df['Assigned_Cluster'].astype(str), color_discrete_sequence=px.colors.qualitative.Set3, title="PCA Reduced Dimensional Space")
            elif len(selected_metrics) == 3:
                fig_cluster = px.scatter_3d(filtered_df, x=selected_metrics[0], y=selected_metrics[1], z=selected_metrics[2], color=filtered_df['Assigned_Cluster'].astype(str), color_discrete_sequence=px.colors.qualitative.Set3)
            else:
                fig_cluster = px.scatter(filtered_df, x=selected_metrics[0], y=selected_metrics[1], color=filtered_df['Assigned_Cluster'].astype(str), color_discrete_sequence=px.colors.qualitative.Set3, symbol="Location_Type")
            
            fig_cluster.update_traces(marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
            fig_cluster.update_layout(template=base_template, margin=dict(t=30, b=10, l=10, r=10), height=550, legend_title_text="Behavioral Cluster")
            st.plotly_chart(fig_cluster, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tab_anomaly:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig_isolation = px.scatter(filtered_df, x="Date", y="Total_Withdrawals", color="Diagnostic_Status", color_discrete_map={"Critical Spike": "#ff4444", "Nominal": "rgba(102, 252, 241, 0.2)"})
            fig_isolation.update_traces(marker=dict(size=9, line=dict(width=0.5, color='white')), selector=dict(mode='markers'))
            fig_isolation.update_layout(template=base_template, margin=dict(t=10, b=10, l=10, r=10), height=450, hovermode="x unified")
            st.plotly_chart(fig_isolation, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            critical_events = filtered_df[filtered_df['Diagnostic_Status'] == "Critical Spike"].sort_values('Total_Withdrawals', ascending=False)
            st.markdown(f"### ⚠️ Priority Action Queue ({len(critical_events)} Operations Required)")
            display_cols = ['ATM_ID', 'Date', 'Location_Type'] + selected_metrics
            st.dataframe(critical_events[display_cols], use_container_width=True)

elif nav_mode == "🔮 Predictive Forecasting":
    # NEW GRAPH CONTROL
    st.markdown("### 🎯 Forecasting Target Selection")
    forecast_target = st.selectbox("Select Target Node for Forecasting Analysis:", ['Entire Fleet Output'] + list(filtered_df['ATM_ID'].unique()))
    
    if forecast_target == 'Entire Fleet Output':
        forecast_df = filtered_df
        title_suffix = "(Entire Fleet)"
    else:
        forecast_df = filtered_df[filtered_df['ATM_ID'] == forecast_target]
        title_suffix = f"(Node: {forecast_target})"

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f"### 📅 Withdrawal Velocity by Day of Week {title_suffix}")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_stats = forecast_df.groupby('Day_of_Week')['Total_Withdrawals'].mean().reindex(day_order).reset_index()
    fig_days = px.bar(daily_stats, x='Day_of_Week', y='Total_Withdrawals', color='Total_Withdrawals', color_continuous_scale='Tealgrn')
    fig_days.update_layout(template=base_template, margin=dict(t=10, b=10, l=10, r=10), height=400, xaxis_title="Operational Day", yaxis_title="Average Velocity ($)")
    st.plotly_chart(fig_days, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_for1, col_for2 = st.columns(2)
    with col_for1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"### 🔄 7-Day Rolling Trajectory {title_suffix}")
        fig_rolling = px.line(forecast_df.groupby('Date')['Rolling_Mean_Withdrawals'].mean().reset_index(), x='Date', y='Rolling_Mean_Withdrawals', line_shape='spline', color_discrete_sequence=['#45a29e'])
        fig_rolling.update_layout(template=base_template, margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig_rolling, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_for2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"### 🔀 Day-over-Day Volatility (%) {title_suffix}")
        fig_vol = px.histogram(forecast_df, x='Daily_Change_Pct', nbins=50, color_discrete_sequence=['#66fcf1'])
        fig_vol.update_layout(template=base_template, margin=dict(t=10, b=10, l=10, r=10), height=300, xaxis_title="Volatility Percentage", yaxis_title="Frequency")
        st.plotly_chart(fig_vol, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid rgba(102, 252, 241, 0.1);">
        <p class="footer-text">SECURE NODE CONNECTION • FINTRUST ENTERPRISE GRADE MACHINE LEARNING</p>
    </div>
    """, 
    unsafe_allow_html=True
)

