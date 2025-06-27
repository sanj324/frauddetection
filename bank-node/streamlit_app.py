import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import shap
import streamlit.components.v1 as components

# Streamlit page config
st.set_page_config(page_title="🧠 Suspicious Account Detector", layout="wide")

# Helper to render force plot in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Load model and feature columns
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "suspicious_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

# Title
st.markdown("<h1 style='color:navy;'>🔍 Suspicious Account Detector</h1>", unsafe_allow_html=True)

# File Upload
uploaded_file = st.file_uploader("📁 Upload CSV file with account data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_features = df[feature_columns]

    # Make prediction
    predictions = model.predict(df_features)
    df["prediction"] = predictions
    df["prediction_label"] = df["prediction"].apply(lambda x: "🔵 Suspicious" if x == 1 else "🔴 Normal")

    # Display metrics
    total = len(df)
    suspicious = (df["prediction"] == 1).sum()
    normal = total - suspicious
    suspicious_rate = (suspicious / total) * 100

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔢 Total Accounts", total)
    col2.metric("🔵 Suspicious", suspicious)
    col3.metric("🔴 Normal", normal)
    col4.metric("⚠️ Suspicious Rate", f"{suspicious_rate:.2f}%")

    # Show prediction table
    st.markdown("### 🧾 Prediction Table")
    st.dataframe(df)

    # Pie chart
    st.markdown("### 📊 Prediction Summary")
    pie_counts = df["prediction_label"].value_counts()
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(pie_counts, labels=pie_counts.index, autopct="%1.1f%%", startangle=90)
    ax_pie.axis("equal")
    st.pyplot(fig_pie)

    # SHAP Explainability
    st.markdown("### 🧠 Global Feature Impact (SHAP)")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_features)

        st.code(f"SHAP shape: {shap_values[1].shape}")
        st.code(f"Input shape: {df_features.shape}")

        # Only use class 1 (suspicious) for binary classifier
        class_index = 1 if isinstance(shap_values, list) else 0
        class_shap_values = shap_values[class_index]

        if class_shap_values.shape == df_features.shape:
            fig_summary = plt.figure()
            shap.summary_plot(class_shap_values, df_features, show=False)
            st.pyplot(fig_summary)
        else:
            st.warning("⚠️ SHAP value shape mismatch. Cannot plot summary.")

        # Record-Level SHAP Force Plot
        st.markdown("### 🔍 Record-Level SHAP Force Plot")
        for i in range(min(3, len(df))):
            st.markdown(f"**Record {i+1}**")
            try:
                force_plot = shap.force_plot(
                    base_value=explainer.expected_value[class_index],
                    shap_values=class_shap_values[i],
                    features=df_features.iloc[i],
                    matplotlib=False
                )
                st_shap(force_plot, height=300)
            except Exception as e:
                st.warning(f"⚠️ Could not render force plot for record {i+1}: {e}")

    except Exception as e:
        st.warning(f"⚠️ Could not generate SHAP plots: {e}")

    # Download Button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results as CSV", csv, "prediction_results.csv", "text/csv")

else:
    st.warning("👆 Please upload a CSV file to begin.")
