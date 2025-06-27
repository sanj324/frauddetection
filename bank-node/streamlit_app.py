import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import shap
import streamlit.components.v1 as components

# Page config
st.set_page_config(page_title="🧠 Suspicious Account Detector", layout="wide")

# Load model and feature columns
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "suspicious_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

# Title
st.markdown("<h1 style='color:navy;'>🔍 Suspicious Account Detector</h1>", unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("📁 Upload CSV file with account data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Align input with training features
    df_features_only = df[feature_columns]

    # Predict
    predictions = model.predict(df_features_only)
    df["prediction"] = predictions
    df["prediction_label"] = df["prediction"].apply(lambda x: "🟥 Suspicious" if x == 1 else "🟩 Normal")

    # KPIs
    total = len(df)
    suspicious = (df["prediction"] == 1).sum()
    normal = total - suspicious
    suspicious_rate = (suspicious / total) * 100

    st.markdown("---")
    st.markdown("### 📈 <span style='color:darkblue;'>Account Summary KPIs</span>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔢 Total Accounts", total)
    col2.metric("🟥 Suspicious", suspicious)
    col3.metric("🟩 Normal", normal)
    col4.metric("⚠️ Suspicious Rate", f"{suspicious_rate:.2f}%")

    # Table
    st.success("✅ Prediction Complete")
    st.markdown("### 🧾 <span style='color:darkgreen;'>Prediction Table</span>", unsafe_allow_html=True)
    st.dataframe(df)

    # Pie Chart
    st.markdown("### 📊 <span style='color:purple;'>Prediction Summary</span>", unsafe_allow_html=True)
    summary = df["prediction_label"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(summary, labels=summary.index, autopct='%1.1f%%', startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    # SHAP Global Summary
    st.markdown("---")
    st.markdown("### 🧠 <span style='color:brown;'>Global Feature Impact (SHAP)</span>", unsafe_allow_html=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_features_only)

    fig_summary, ax_summary = plt.subplots()
    shap.summary_plot(shap_values, df_features_only, show=False)
    st.pyplot(fig_summary)

    # SHAP Record-Level Force Plots
    st.markdown("### 🔍 <span style='color:#aa3333;'>Record-Level Explanation (SHAP Force Plot)</span>", unsafe_allow_html=True)
    for i in range(min(5, len(df))):
        st.markdown(f"**Record {i + 1}**")
        force_plot_html = shap.force_plot(
            explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            shap_values[1][i] if isinstance(shap_values, list) else shap_values[i],
            df_features_only.iloc[i],
            matplotlib=False
        )
        components.html(force_plot_html.html(), height=300)

    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv"
    )

else:
    st.warning("👆 Please upload a CSV file to begin.")
