"""
Streamlit app: Interactive UI for IsolationForest anomaly detection
Default data path: set DEFAULT_PATH to a local filename (e.g. "iso_cred.csv") or leave as a full path.
Features:
- Load default CSV from DEFAULT_PATH or upload your own CSV
- Choose scaler (Standard / MinMax / None)
- Select features for training and visualization
- Set IsolationForest hyperparameters (n_estimators, contamination, max_samples, random_state)
- Fit model, show anomaly scores and labels, display interactive plots (scatter, histogram), and download results

Run: streamlit run streamlit_isolation_forest_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import plotly.express as px
import io

# NOTE for Windows users:
# If you want to use a default CSV file, put the CSV in the same folder as this script
# and set DEFAULT_PATH = "iso_cred.csv" (or the filename you used).
# Alternatively, use the sidebar Upload control to load any CSV without editing this file.
DEFAULT_PATH = "C:\\Users\\LENOVO\\OneDrive\\Desktop\\credit_card\\credit_card.csv"

st.set_page_config(page_title="IsolationForest Explorer", layout="wide")
st.title("IsolationForest — Interactive Anomaly Detection Explorer")

# Sidebar: data loading
st.sidebar.header("1) Data")
use_default = st.sidebar.checkbox("Use default dataset", value=True)
uploaded_file = st.sidebar.file_uploader("Or upload a CSV file", type=["csv"])

@st.cache_data
def load_default(path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.sidebar.error(f"Failed to load default file: {e}")
        return None
    return df

if use_default and uploaded_file is None:
    df = load_default(DEFAULT_PATH)
else:
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.sidebar.error(f"Failed to read uploaded CSV: {e}")
            df = None
    else:
        df = None

if df is None:
    st.warning("No dataset loaded yet. Upload a CSV or enable the default dataset in the sidebar.")
    st.stop()

st.markdown(f"**Dataset loaded** — {df.shape[0]} rows × {df.shape[1]} columns")

# Show a sample of data
if st.checkbox("Show raw data (first 100 rows)"):
    st.dataframe(df.head(100))

# Sidebar: preprocessing and feature selection
st.sidebar.header("2) Preprocessing & Features")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found in the dataset — IsolationForest requires numeric features.")
    st.stop()

selected_features = st.sidebar.multiselect("Select numeric features to use", numeric_cols, default=numeric_cols[:min(5,len(numeric_cols))])
if len(selected_features) < 1:
    st.error("Please select at least one numeric feature.")
    st.stop()

scaler_choice = st.sidebar.selectbox("Scaler", ["StandardScaler", "MinMaxScaler", "None"], index=0)

@st.cache_data
def scale_data(X, scaler_choice):
    if scaler_choice == "StandardScaler":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    elif scaler_choice == "MinMaxScaler":
        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = X.values
    return Xs

X = df[selected_features]
X_scaled = scale_data(X, scaler_choice)

# Sidebar: model parameters
st.sidebar.header("3) IsolationForest parameters")
n_estimators = st.sidebar.number_input("n_estimators", min_value=10, max_value=2000, value=100, step=10)
contamination = st.sidebar.slider("contamination (expected fraction of outliers)", min_value=0.0, max_value=0.5, value=0.02, step=0.005)
max_samples = st.sidebar.text_input("max_samples (int or 'auto')", value='auto')
random_state = st.sidebar.number_input("random_state (seed)", min_value=0, max_value=9999, value=42, step=1)

# Convert max_samples
try:
    if str(max_samples).strip().lower() == 'auto':
        max_s = 'auto'
    else:
        max_s = int(max_samples)
except Exception:
    max_s = 'auto'

run_button = st.sidebar.button("Fit IsolationForest")

# Main: model training and results
if run_button:
    with st.spinner("Training IsolationForest..."):
        iso = IsolationForest(n_estimators=int(n_estimators), contamination=float(contamination), max_samples=max_s, random_state=int(random_state))
        iso.fit(X_scaled)
        # anomaly score: negative scores are more abnormal; use decision_function or score_samples
        scores = iso.score_samples(X_scaled)  # higher -> more normal
        # For user-friendly anomaly score: invert so higher = more anomalous
        anomaly_score = -scores
        preds = iso.predict(X_scaled)  # -1 for outliers, 1 for inliers
        is_anomaly = (preds == -1).astype(int)

        results = df.copy()
        results['_anomaly_score'] = anomaly_score
        results['_is_anomaly'] = is_anomaly

    # Results summary
    col1, col2, col3 = st.columns([1,1,1])
    col1.metric("Rows", f"{results.shape[0]}")
    col2.metric("Anomalies", f"{results['_is_anomaly'].sum()} ({results['_is_anomaly'].mean()*100:.2f}%)")
    col3.metric("Features used", ", ".join(selected_features[:3]) + ("..." if len(selected_features)>3 else ""))

    st.subheader("Anomaly table (top rows)")
    st.dataframe(results.sort_values('_anomaly_score', ascending=False).head(200))

    # Download button
    csv_buf = io.StringIO()
    results.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode('utf-8')
    st.download_button("Download results CSV", data=csv_bytes, file_name="isolation_forest_results.csv", mime='text/csv')

    # Visualization area
    st.subheader("Visualizations")
    viz_col1, viz_col2 = st.columns(2)

    # Scatter plot: allow choosing two features or PCA components
    with viz_col1:
        st.markdown("**2D Scatter (select X and Y)**")
        vis_x = st.selectbox("X axis", options=selected_features, index=0)
        vis_y = st.selectbox("Y axis", options=selected_features, index=min(1, len(selected_features)-1))

        fig = px.scatter(results, x=vis_x, y=vis_y, color=results['_is_anomaly'].astype(str), hover_data=selected_features + ['_anomaly_score'])
        fig.update_layout(legend_title_text='is_anomaly (1=outlier)')
        st.plotly_chart(fig, use_container_width=True)

    with viz_col2:
        st.markdown("**Anomaly score distribution**")
        fig2 = px.histogram(results, x='_anomaly_score', nbins=50)
        st.plotly_chart(fig2, use_container_width=True)

    # PCA 2D projection
    if len(selected_features) > 1:
        st.subheader("PCA projection (2 components)")
        pca = PCA(n_components=2)
        proj = pca.fit_transform(X_scaled)
        proj_df = pd.DataFrame(proj, columns=['PC1','PC2'])
        proj_df['_is_anomaly'] = results['_is_anomaly']
        proj_df['_anomaly_score'] = results['_anomaly_score']
        fig3 = px.scatter(proj_df, x='PC1', y='PC2', color=proj_df['_is_anomaly'].astype(str), hover_data=['_anomaly_score'])
        st.plotly_chart(fig3, use_container_width=True)

    st.info("Tip: try changing contamination and max_samples to see how detection changes.")

else:
    st.info("Adjust parameters in the sidebar and click 'Fit IsolationForest' to run.")

# Footer: show how to reproduce quickly
st.markdown("""**Quick run**: `streamlit run streamlit_isolation_forest_app.py`

This app will try to load the default dataset at `/mnt/data/iso_cred` if "use default dataset" is checked. You can also upload your own CSV.""")
