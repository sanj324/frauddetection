import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import shap
import streamlit.components.v1 as components

# Helper to display force plot in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Set Streamlit page config
st.set_page_config(page_title="\U0001f9e0 Suspicious Account Detector", layout="wide")

# Load model and features
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "suspicious_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

# Title
st.markdown("<h1 style='color:navy;'>\U0001f50d Suspicious Account Detector</h1>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("\U0001f4c1 Upload CSV file with account data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_features_only = df[feature_columns]

    # Prediction
    predictions = model.predict(df_features_only)
    df["prediction"] = predictions
    df["prediction_label"] = df["prediction"].apply(lambda x: "\U0001f535 Suspicious" if x == 1 else "\U0001f534 Normal")

    # Metrics
    total = len(df)
    suspicious = (df["prediction"] == 1).sum()
    normal = total - suspicious
    suspicious_rate = (suspicious / total) * 100

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("\U0001f522 Total Accounts", total)
    col2.metric("\U0001f535 Suspicious", suspicious)
    col3.metric("\U0001f534 Normal", normal)
    col4.metric("\u26a0\ufe0f Suspicious Rate", f"{suspicious_rate:.2f}%")

    # Results table
    st.markdown("### \U0001f9fe Prediction Table")
    st.dataframe(df)

    # Pie chart
    st.markdown("### \U0001f4ca Prediction Summary")
    summary = df["prediction_label"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(summary, labels=summary.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    # SHAP Explainability
    st.markdown("---")
    st.markdown("### \U0001f9e0 Global Feature Impact (SHAP)")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_features_only)

        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values_to_use = shap_values[1]  # Use class index 1
        elif len(shap_values.shape) == 3:
            shap_values_to_use = shap_values[:, :, 1]
        else:
            shap_values_to_use = shap_values

        st.markdown(f"**SHAP shape:** `{shap_values_to_use.shape}`")
        st.markdown(f"**Input shape:** `{df_features_only.shape}`")

        if shap_values_to_use.shape == df_features_only.shape:
            fig_summary = plt.figure()
            shap.summary_plot(shap_values_to_use, df_features_only, show=False)
            st.pyplot(fig_summary)
        else:
            st.warning("\u26a0\ufe0f SHAP value shape mismatch. Cannot plot summary.")

        # Record-level force plots
        st.markdown("### \U0001f50d Record-Level SHAP Force Plot")
        for i in range(min(3, len(df))):
            st.markdown(f"**Record {i + 1}**")
            try:
                force_plot = shap.force_plot(
                    base_value=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                    shap_values=shap_values_to_use[i],
                    features=df_features_only.iloc[i],
                    matplotlib=False
                )
                st_shap(force_plot, height=300)
            except Exception as e:
                st.warning(f"\u26a0\ufe0f Could not render force plot for record {i + 1}: {e}")

    except Exception as e:
        st.warning(f"\u26a0\ufe0f Could not generate SHAP plots: {e}")

    # Download CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("\u2b07\ufe0f Download Results as CSV", csv, "prediction_results.csv", "text/csv")

else:
    st.warning("\ud83d\udc46 Please upload a CSV file to begin.")
