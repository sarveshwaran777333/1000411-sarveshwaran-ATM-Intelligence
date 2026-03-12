import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="FinTrust ATM Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_global_theme():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #0f0f1a; color: #ffffff; }
        .main { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }
        
        div[data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 700; }
        div[data-testid="stMetricLabel"] { color: #rgba(255,255,255,0.7) !important; }
        
        .section-header {
            background: rgba(102, 126, 234, 0.15); 
            padding: 20px; 
            border-radius: 15px; 
            margin-bottom: 20px; 
            border-left: 5px solid #667eea;
        }
        
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #1a1a2e;
            border-radius: 8px 8px 0 0;
            color: #ffffff !important;
            padding: 10px 20px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #667eea !important;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv("atm_cash_management_dataset.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df, True
    except:
        return pd.DataFrame(), False

apply_global_theme()
df, success = load_and_clean_data()

# --- SIDEBAR: Gradient Controls ---
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
    <h2 style="color: white; margin: 0; font-size: 1.2em;">🎛️ AI Control Hub</h2>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    selected_metrics = st.multiselect(
        "Active Data Streams",
        options=['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level', 'Nearby_Competitor_ATMs'],
        default=['Total_Withdrawals', 'Total_Deposits']
    )
    st.divider()
    k_clusters = st.slider("Segmentation Density (K)", 2, 6, 3)
    anomaly_threshold = st.slider("Risk Sensitivity", 0.01, 0.15, 0.05)
    st.divider()
    nav_mode = st.radio("Intelligence Layer", ["Strategic Overview", "Market Behavior", "Risk Intelligence"])

# --- HEADER SECTION ---
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 35px; border-radius: 20px; margin-bottom: 25px; box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);">
    <h1 style="color: white; margin: 0; text-align: center; font-size: 2.5em;">🏦 FinTrust ATM Intelligence</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; text-align: center; font-size: 1.1em;">
        Enterprise High-Dimensional Forecasting & Anomaly Detection Suite
    </p>
</div>
""", unsafe_allow_html=True)

if not success:
    st.error("Error: Could not locate 'atm_cash_management_dataset.csv'.")
    st.stop()

# --- CONTENT LAYERS ---
if nav_mode == "Strategic Overview":
    st.markdown('<div class="section-header"><h2>📊 Fleet Metrics Summary</h2></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Global Fleet", f"{df['ATM_ID'].nunique()} Nodes")
    c2.metric("Mean Throughput", f"${df['Total_Withdrawals'].mean():,.0f}")
    c3.metric("System Liquidity", f"${df['Previous_Day_Cash_Level'].mean():,.0f}")
    c4.metric("Active Components", len(selected_metrics))
    
    st.divider()
    st.subheader("Asset Registry Explorer")
    st.dataframe(df.head(100), use_container_width=True)

elif nav_mode == "Market Behavior":
    st.markdown('<div class="section-header"><h2>📈 Behavioral Correlation Logic</h2></div>', unsafe_allow_html=True)
    if len(selected_metrics) < 2:
        st.warning("Please select at least 2 metrics in the Control Hub.")
    else:
        col_l, col_r = st.columns(2)
        with col_l:
            corr = df[selected_metrics].corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Picnic', title="Component Interaction Heatmap")
            fig_corr.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_corr, use_container_width=True)
        with col_r:
            fig_dist = px.violin(df, y="Total_Withdrawals", x="Location_Type", color="Location_Type", box=True, title="Demand Density by Location")
            fig_dist.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_dist, use_container_width=True)

elif nav_mode == "Risk Intelligence":
    st.markdown('<div class="section-header"><h2>⚡ AI Diagnostic Engine</h2></div>', unsafe_allow_html=True)
    if len(selected_metrics) < 2:
        st.error("Minimum 2 components required for ML modeling.")
    else:
        # ML Engine
        X = StandardScaler().fit_transform(df[selected_metrics])
        df['Cluster'] = KMeans(n_clusters=k_clusters, n_init=10, random_state=42).fit_predict(X)
        df['Signal'] = IsolationForest(contamination=anomaly_threshold, random_state=42).fit_predict(X)
        df['Status'] = df['Signal'].map({1: "Normal", -1: "Critical Spike"})

        t1, t2 = st.tabs(["Demand Segmentation", "Anomaly Tracking"])
        
        with t1:
            fig_cl = px.scatter(df, x=selected_metrics[0], y=selected_metrics[1], color=df['Cluster'].astype(str),
                               symbol="Location_Type", title=f"K-Means Clustering (K={k_clusters})")
            fig_cl.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_cl, use_container_width=True)
            
        with t2:
            fig_anom = px.scatter(df, x="Date", y="Total_Withdrawals", color="Status", 
                                 color_discrete_map={"Critical Spike": "#ff4444", "Normal": "#00ff88"})
            fig_anom.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_anom, use_container_width=True)
            
            st.subheader("High-Priority System Flags")
            st.table(df[df['Status'] == "Critical Spike"][['ATM_ID', 'Date', 'Location_Type', 'Total_Withdrawals']].head(10))
