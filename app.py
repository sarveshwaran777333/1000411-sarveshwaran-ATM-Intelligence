import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import time

st.set_page_config(
    page_title="FinTrust | AI Analytics",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_premium_ui():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
        
        .stApp {
            background-color: #0b0c10;
            background-image: radial-gradient(circle at 50% 0%, #1f2833 0%, #0b0c10 70%);
            color: #c5c6c7;
            font-family: 'Inter', sans-serif;
        }
        
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: #ffffff; }
        
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
            background: linear-gradient(135deg, #0b0c10 0%, #1f2833 100%);
            border-left: 6px solid #45a29e;
            border-right: 6px solid #66fcf1;
            padding: 30px 40px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(102, 252, 241, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .hero-banner::after {
            content: '';
            position: absolute;
            top: -50%; left: -50%; width: 200%; height: 200%;
            background: radial-gradient(circle, rgba(102,252,241,0.05) 0%, transparent 50%);
            animation: rotate 10s linear infinite;
        }
        @keyframes rotate { 100% { transform: rotate(360deg); } }
        
        .kpi-title { font-size: 0.9rem; color: #45a29e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;}
        .kpi-value { font-size: 2.5rem; font-weight: 700; color: #ffffff; font-family: 'Space Grotesk', sans-serif; }
        .kpi-value.neon { 
            background: -webkit-linear-gradient(45deg, #66fcf1, #45a29e); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
        }

        .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(31, 40, 51, 0.6);
            border-radius: 8px 8px 0 0;
            border: 1px solid rgba(102, 252, 241, 0.1);
            border-bottom: none;
            color: #c5c6c7 !important;
            padding: 12px 24px;
            font-family: 'Space Grotesk', sans-serif;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(180deg, rgba(102, 252, 241, 0.1) 0%, rgba(31, 40, 51, 0) 100%);
            border-top: 2px solid #66fcf1;
            color: #ffffff !important;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    try:
        df = pd.read_csv("atm_cash_management_dataset.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df, True
    except:
        return pd.DataFrame(), False

inject_premium_ui()
df, success = load_data()

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #66fcf1; font-size: 2rem; margin:0;">💠 FinTrust</h1>
        <p style="color: #45a29e; font-size: 0.9rem; margin:0; letter-spacing: 2px;">NEURAL CORE</p>
    </div>
    """, unsafe_allow_html=True)
    
    nav_mode = st.radio("OPERATING LAYER", ["🌐 Global Overview", "📈 Market Dynamics", "⚡ Risk & Diagnostics"], label_visibility="collapsed")
    
    st.markdown("<hr style='border-color: rgba(102,252,241,0.2);'>", unsafe_allow_html=True)
    st.markdown("<p style='color: #45a29e; font-size:0.85rem; font-weight:600; margin-bottom:10px;'>DATA STREAMS</p>", unsafe_allow_html=True)
    
    available_cols = ['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level', 'Nearby_Competitor_ATMs', 'Cash_Demand_Next_Day']
    selected_metrics = st.multiselect(
        "Select vectors for AI analysis:",
        options=available_cols,
        default=['Total_Withdrawals', 'Total_Deposits', 'Nearby_Competitor_ATMs'],
        label_visibility="collapsed"
    )
    
    st.markdown("<hr style='border-color: rgba(102,252,241,0.2);'>", unsafe_allow_html=True)
    st.markdown("<p style='color: #45a29e; font-size:0.85rem; font-weight:600; margin-bottom:10px;'>HYPERPARAMETERS</p>", unsafe_allow_html=True)
    k_clusters = st.slider("Segmentation Nodes (K)", 2, 6, 3)
    anomaly_threshold = st.slider("Anomaly Contamination", 0.01, 0.10, 0.05)

if not success:
    st.error("System Failure: 'atm_cash_management_dataset.csv' disconnected.")
    st.stop()

st.markdown("""
<div class="hero-banner">
    <div style="position: relative; z-index: 1;">
        <h1 style="margin: 0; font-size: 2.8rem; letter-spacing: -1px;">ATM Fleet Intelligence Suite</h1>
        <p style="color: #45a29e; font-size: 1.1rem; margin: 5px 0 0 0; font-weight: 500;">
            Real-Time Telemetry • Predictive Clustering • Anomaly Isolation
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

custom_template = go.layout.Template()
custom_template.layout.plot_bgcolor = "rgba(0,0,0,0)"
custom_template.layout.paper_bgcolor = "rgba(0,0,0,0)"
custom_template.layout.font.color = "#c5c6c7"
custom_template.layout.xaxis.gridcolor = "rgba(102,252,241,0.1)"
custom_template.layout.yaxis.gridcolor = "rgba(102,252,241,0.1)"

if nav_mode == "🌐 Global Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="glass-card"><div class="kpi-title">Active Nodes</div><div class="kpi-value neon">{df['ATM_ID'].nunique()}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="glass-card"><div class="kpi-title">Mean Throughput</div><div class="kpi-value">${df['Total_Withdrawals'].mean():,.0f}</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="glass-card"><div class="kpi-title">Max Spike</div><div class="kpi-value">${df['Total_Withdrawals'].max():,.0f}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="glass-card"><div class="kpi-title">Active Vectors</div><div class="kpi-value neon">{len(selected_metrics)}</div></div>""", unsafe_allow_html=True)

    st.markdown("### 🗄️ Secure Data Ledger")
    st.dataframe(
        df.head(100),
        use_container_width=True, height=400
    )

elif nav_mode == "📈 Market Dynamics":
    if len(selected_metrics) < 2:
        st.warning("⚠️ Insufficient Data Vectors. Select at least 2 in the Neural Core.")
    else:
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🧬 Vector Interaction Matrix")
            corr = df[selected_metrics].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale='Tealgrn', aspect="auto")
            fig_corr.update_layout(template=custom_template, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_corr, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Distribution Topography")
            fig_dist = px.box(df, y=selected_metrics[0], color="Location_Type", 
                              color_discrete_sequence=['#66fcf1', '#45a29e', '#1f2833'])
            fig_dist.update_layout(template=custom_template, margin=dict(t=30, b=0, l=0, r=0), showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

elif nav_mode == "⚡ Risk & Diagnostics":
    if len(selected_metrics) < 2:
        st.error("⚠️ AI Diagnostic Engine requires at least 2 Data Streams.")
    else:
        with st.spinner("Compiling Neural Models..."):
            X = StandardScaler().fit_transform(df[selected_metrics])
            df['Cluster'] = KMeans(n_clusters=k_clusters, n_init=10, random_state=42).fit_predict(X)
            df['Signal'] = IsolationForest(contamination=anomaly_threshold, random_state=42).fit_predict(X)
            df['Risk_Level'] = df['Signal'].map({1: "Nominal", -1: "Critical Spike"})
            time.sleep(0.3)
            st.toast("AI Models Recalibrated Successfully", icon="✅")

        t1, t2 = st.tabs(["🧩 High-Dimensional Clustering", "🚨 Anomaly Isolation"])
        
        with t1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            if len(selected_metrics) >= 3:
                
                fig_cl = px.scatter_3d(df, x=selected_metrics[0], y=selected_metrics[1], z=selected_metrics[2],
                                      color=df['Cluster'].astype(str), color_discrete_sequence=px.colors.qualitative.Set3)
                fig_cl.update_traces(marker=dict(size=5, line=dict(width=1, color='DarkSlateGrey')))
            else:
                
                fig_cl = px.scatter(df, x=selected_metrics[0], y=selected_metrics[1], color=df['Cluster'].astype(str),
                                   color_discrete_sequence=px.colors.qualitative.Set3, symbol="Location_Type")
                fig_cl.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
            
            fig_cl.update_layout(template=custom_template, margin=dict(t=10, b=10, l=10, r=10), height=500)
            st.plotly_chart(fig_cl, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with t2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            fig_anom = px.scatter(df, x="Date", y="Total_Withdrawals", color="Risk_Level", 
                                 color_discrete_map={"Critical Spike": "#ff4444", "Nominal": "rgba(102, 252, 241, 0.3)"})
            fig_anom.update_traces(marker=dict(size=8))
            fig_anom.update_layout(template=custom_template, margin=dict(t=10, b=10, l=10, r=10), height=400)
            st.plotly_chart(fig_anom, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            anomalies = df[df['Risk_Level'] == "Critical Spike"].sort_values('Total_Withdrawals', ascending=False)
            st.markdown(f"### ⚠️ Priority Action Queue ({len(anomalies)} Flags)")
            st.dataframe(anomalies[['ATM_ID', 'Date', 'Location_Type'] + selected_metrics], use_container_width=True)

st.markdown("""
<div style="text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid rgba(102, 252, 241, 0.1);">
    <p style="color: #45a29e; font-size: 0.85rem; letter-spacing: 1px;">
        SECURE NODE CONNECTION • FINTRUST ENTERPRISE GRADE MACHINE LEARNING
    </p>
</div>
""", unsafe_allow_html=True)
