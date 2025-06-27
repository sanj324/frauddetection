import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import shap

# 🎨 Page Configuration
st.set_page_config(page_title="🧠 Suspicious Account Detector", layout="wide")

# 📂 Path Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "suspicious_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")

# ✅ Load Model and Feature Columns
model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

# 🏷️ UI Header
st.title("🔍 Suspicious Account Detector")

uploaded_file = st.file_uploader("📁 Upload CSV file with account data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("### 📊 Uploaded Data Preview")
    st.dataframe(df.head())

    # 🧠 Model Inference
    df_features_only = df[feature_columns]
    predictions = model.predict(df_features_only)
    df["prediction"] = predictions
    df["prediction_label"] = df["prediction"].apply(lambda x: "🟥 Suspicious" if x == 1 else "🟩 Normal")

    # 📈 KPIs
    total = len(df)
    suspicious = (df["prediction"] == 1).sum()
    normal = total - suspicious
    suspicious_rate = (suspicious / total) * 100

    st.markdown("---")
    st.markdown("### 📈 Summary KPIs")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔢 Total Accounts", total)
    col2.metric("🟥 Suspicious", suspicious)
    col3.metric("🟩 Normal", normal)
    col4.metric("⚠️ Suspicious Rate", f"{suspicious_rate:.2f}%")

    # ✅ Data Table
    st.success("✅ Prediction Complete")
    st.markdown("### 🧾 Full Prediction Table")
    st.dataframe(df)

    # 📊 Pie Chart
    st.subheader("📊 Prediction Distribution")
    summary = df["prediction_label"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(summary, labels=summary.index, autopct='%1.1f%%', startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    # 🧠 SHAP Explainability
    st.markdown("---")
    st.subheader("🧠 Global Feature Impact (SHAP)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_features_only)

    fig_summary, ax_summary = plt.subplots()
    shap.summary_plot(shap_values, df_features_only, show=False)
    st.pyplot(fig_summary)

    # 🔍 Individual Record Insights
    st.subheader("🔍 Why is a Record Suspicious?")
    for i in range(min(5, len(df))):
        st.markdown(f"**Record {i + 1}**")
        shap_html = shap.plots.force(
            explainer.expected_value[1], shap_values[1][i], df_features_only.iloc[i], matplotlib=False
        )
        st.components.v1.html(shap_html.html(), height=150)

    # 💾 Download Button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv"
    )
else:
    st.warning("👆 Please upload a CSV file to begin.")
