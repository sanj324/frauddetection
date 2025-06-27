import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib
matplotlib.use("Agg")  # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
import shap

# Page config
st.set_page_config(page_title="🧠 Suspicious Account Detector", layout="wide")

# Load model and feature columns
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "suspicious_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

st.markdown("<h1 style='color:navy;'>🔍 Suspicious Account Detector</h1>", unsafe_allow_html=True)

# Upload section
uploaded_file = st.file_uploader("📁 Upload CSV file with account data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if not all(col in df.columns for col in feature_columns):
        st.error("Uploaded file is missing required columns.")
    else:
        df_features = df[feature_columns]

        # Predict
        predictions = model.predict(df_features)
        df["prediction"] = predictions
        df["prediction_label"] = df["prediction"].apply(lambda x: "🟥 Suspicious" if x == 1 else "🟩 Normal")

        # KPIs
        total = len(df)
        suspicious = (df["prediction"] == 1).sum()
        normal = total - suspicious
        suspicious_rate = (suspicious / total) * 100

        st.markdown("---")
        st.markdown("### 📈 Account Summary KPIs", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔢 Total Accounts", total)
        col2.metric("🟥 Suspicious", suspicious)
        col3.metric("🟩 Normal", normal)
        col4.metric("⚠️ Suspicious Rate", f"{suspicious_rate:.2f}%")

        # Results table
        st.success("✅ Prediction Complete")
        st.markdown("### 🧾 Prediction Table", unsafe_allow_html=True)
        st.dataframe(df)

        # Pie Chart
        st.markdown("### 📊 Prediction Summary", unsafe_allow_html=True)
        summary = df["prediction_label"].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(summary, labels=summary.index, autopct='%1.1f%%', startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1)

        # SHAP Explanation
        st.markdown("---")
        st.markdown("### 🧠 Global Feature Impact (SHAP)", unsafe_allow_html=True)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_features)

        fig2, ax2 = plt.subplots()
        shap.summary_plot(shap_values, df_features, show=False)
        st.pyplot(fig2)

        st.markdown("### 🔍 Record-Level SHAP Force Plot", unsafe_allow_html=True)
        for i in range(min(3, len(df))):
            st.markdown(f"**Record {i + 1}**")
            try:
                fig3 = plt.figure()
                shap.force_plot(
                    base_value=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                    shap_values=shap_values[1][i] if isinstance(shap_values, list) else shap_values[i],
                    features=df_features.iloc[i],
                    matplotlib=True,
                    show=False
                )
                st.pyplot(fig3)
            except Exception as e:
                st.warning(f"⚠️ Could not render force plot for record {i+1}: {e}")

        # Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results as CSV", data=csv, file_name="prediction_results.csv", mime="text/csv")
else:
    st.info("👆 Please upload a CSV file to begin.")
