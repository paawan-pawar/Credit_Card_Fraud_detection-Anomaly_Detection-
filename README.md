# Credit_Card_Fraud_detection-Anomaly_Detection-
📌 Isolation Forest Anomaly Detection – Streamlit App

This project provides an interactive Streamlit interface for performing anomaly detection using Isolation Forest.
You can upload your own CSV dataset, explore features, visualize outliers, and download results — all through a simple UI.

🚀 Features

📂 Upload any CSV dataset

⚙️ Customize model parameters

n_estimators

contamination

max_samples

random_state

📊 Automatic feature detection (numeric columns)

🔍 Interactive visualizations

2D scatter plot with anomaly coloring

Histogram of anomaly scores

PCA 2D projection

📉 Outlier detection using Isolation Forest

⬇️ Download processed results as CSV

📦 Installation

Clone the repo:

git clone <your-repo-url>
cd <your-project-folder>


Create a virtual environment (optional but recommended):

python -m venv venv
venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt

▶️ Running the App

Run the Streamlit app with:

streamlit run streamlit_isolation_forest_app.py


Or:

python -m streamlit run streamlit_isolation_forest_app.py

📁 Dataset

You can load your data in two ways:

1. Upload CSV (recommended)

Use the file uploader in the sidebar to upload your dataset.

2. Default Dataset (optional)

Place a CSV file in the same folder and update:

DEFAULT_PATH = "yourfile.csv"

🖼️ UI Overview

Sidebar controls for selecting features and tweaking model parameters

Main area displays:

Dataset preview

Outlier summary

Interactive plots

Outlier table

Download button

📤 Output

After model execution, the app generates:

_anomaly_score — numeric score (higher = more anomalous)

_is_anomaly — 1 = anomaly, 0 = normal

You can download the full results as a CSV file.

🧠 Model Used

This app uses:

IsolationForest
from sklearn.ensemble


Isolation Forest is ideal for:

Credit card fraud detection

Rare event identification

Unsupervised anomaly detection

📄 License

This project is open-source and free to use.
