# 1000411-sarveshwaran-ATM-Intelligence

# FinTrust: ATM Fleet Intelligence Suite
Transforming raw ATM telemetry into actionable, AI-driven cash management strategies.

# Project Overview
FinTrust is an interactive enterprise dashboard designed for FinTrust Bank Ltd. to optimize ATM cash replenishment. By logging and analyzing historical transaction data, the app uses machine learning to cluster ATM behaviors, isolate critical demand spikes, and utilizes Generative AI to recommend actionable cash allocation strategies for the fleet.

[Live App](https://1000411-sarveshwaran-atm-intelligence-fpd3yqgbpew9gptmzffdys.streamlit.app/)

# User Focus
Target Audience: ATM Fleet Managers, Bank Operations Analysts, and Financial Planners.

Problem Solved: Bridges the gap between raw transaction data and operational efficiency by predicting cash stock-outs, identifying anomalies, and minimizing idle cash traps.

Design Philosophy: Uses an "Enterprise Command Center" approach with dark-mode glassmorphism and neon-cyan accents to deliver dense data clearly and professionally.

# Key Features
As per the project brief, this application integrates the following:

Real-Time Telemetry: Live global overview of fleet throughput, active nodes, and max demand spikes.

Market Dynamics: Interactive spatial demand topography and feature correlation matrices.

Predictive Clustering: Uses K-Means Machine Learning to group ATMs by behavioral topography, visualized in a 3D PCA space.

Anomaly Isolation: Employs an Isolation Forest algorithm to detect and flag critical spikes and holiday surges into a Priority Action Queue.

AI Cash Allocation: Integrates Google Gemini API to autonomously analyze telemetry and output natural language recommendations on which ATMs to load or optimize.

Interactive Forecasting: 7-day rolling trajectories and day-over-day volatility tracking per node.

# Integration & Logic
Core Constructs: Built with Pandas and NumPy for complex data wrangling, time-series aggregation, and rolling mean calculations.

Modular Design: Integrates Scikit-Learn for unsupervised ML (K-Means, Isolation Forest, PCA) and Streamlit caching for lightning-fast performance.

UI/UX: Developed in Streamlit using a custom ultra-premium CSS injection (multi-layered mesh gradients, shimmering text, interactive hover tooltips).

# Deployment Instructions
To view the project, you can visit the Web App Link

# To run locally:
Clone this repository.

Ensure you have the dependencies installed (pip install -r requirements.txt).

Set up your .streamlit/secrets.toml file with your GEMINI_API_KEY.

Run the command: streamlit run app.py.

# Application Flow
The FinTrust app follows a sequential flow designed to move the user from global data exploration to targeted AI action:

# Data Ingestion Stage (The Foundation)
Dataset Parsing: The app reads historical ATM data (atm_cash_management_dataset.csv) and preprocesses it by calculating rolling means, daily volatility percentages, and weekend flags.

# EDA & Telemetry Stage (Global Overview)
Fleet Dashboard: Displays a timeline of fleet throughput and key performance indicators (Active Nodes, Mean Throughput).
Market Dynamics: Analyzes correlations and visualizes demand distributions via box plots based on location types.

# Machine Learning Stage (Risk & Diagnostics)
Clustering: Standardizes data and applies K-Means to group ATMs into behavioral profiles (e.g., High-Demand vs. Steady).
Anomaly Detection: Isolation Forest flags unusual spikes, pushing them to a Priority Action Queue dataframe for managers.

# Feedback & AI Strategy Stage (Predictive Forecasting)
Generative AI: Passes ATM summaries to Google's Gemini model to generate human-readable cash allocation strategies (Load More vs. Optimize/Load Less).
Interactive Trajectories: Users can isolate specific ATMs to view day-over-day volatility histograms.

# Deployment & Export
Cloud Hosting: The final solution is deployed via Streamlit Cloud.
Data Export: Allows managers to download the newly filtered and clustered dataset for external reporting.

# Repository Structure
app.py: Main Python file containing the Streamlit interface, ML models, and Gemini AI logic.

requirements.txt: List of necessary Python packages (streamlit, pandas, scikit-learn, plotly, google-generativeai).

atm_cash_management_dataset.csv: The dataset containing historical ATM telemetry.

.streamlit/secrets.toml: Secure file holding the Gemini API key (not tracked in version control).

# Story board
[story board](https://www.canva.com/design/DAHC9wQJQVQ/JCWtCRmRlGH76DRGkPm2GQ/edit?utm_content=DAHC9wQJQVQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

[screenshots](https://drive.google.com/drive/folders/1-yJ8vJtxeJ8nAJtRBsbQr13MyEbGaAb6?usp=sharing)

# Tested by
Sister: tested the UI layout, visual hierarchy, and navigation.
Saif (friend): tested the AI allocation engine and the CSV dataset export feature.

# Credits & Acknowledgements
This project was developed as part of the Formative Assessment-2 (FA-2) for the Data Mining course under the Artificial Intelligence program.

School Name: Jain Vidyalaya

Student Name: K.Sarveshwaran

Class: XI

Registration ID: 1000411

Mentor: Aruljothi.M
