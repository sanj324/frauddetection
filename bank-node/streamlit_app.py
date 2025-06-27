import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import shap

# Page config
st.set_page_config(page_title="🧠 Suspicious Account Detector", layout="wide")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "suspicious_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")

# Load model and features
model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

# UI
st.title("🧠 Suspicious Account Detector")
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_features = df[feature_columns]

    # Predict
    predictions = model.predict(df_features)
    df["prediction"] = predictions
    df["label"] = df["prediction"].apply(lambda x: "🟥 Suspicious" if x == 1 else "🟩 Normal")

    # Summary
    total = len(df)
    suspicious = (df["prediction"] == 1).sum()
    st.metric("Total Accounts", total)
    st.metric("Suspicious", suspicious)
    st.dataframe(df)

    # SHAP setup
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_features)

    # SHAP Summary Plot
    st.markdown("### 📊 SHAP Summary Plot")
    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_values, df_features, show=False)
    st.pyplot(fig1)

    # Force Plot (matplotlib safe fallback)
    st.markdown("### 🔍 Record-Level SHAP Force Plot (Top 3)")
    for i in range(min(3, len(df))):
        st.markdown(f"**Record {i+1}**")
        try:
            fig2, ax2 = plt.subplots()
            shap.force_plot(
                base_value=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                shap_values=shap_values[1][i] if isinstance(shap_values, list) else shap_values[i],
                features=df_features.iloc[i],
                matplotlib=True,
                show=False
            )
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"⚠️ Could not render force plot for record {i+1}: {e}")
else:
    st.info("Upload a CSV to begin.")
