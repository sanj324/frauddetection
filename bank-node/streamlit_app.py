import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ------------------🎯 Path Setup------------------ #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "suspicious_model.pkl")
model = joblib.load(MODEL_PATH)

# ------------------🧠 Page Setup------------------ #
st.set_page_config(page_title="🧠 Suspicious Account Detector", layout="wide")
st.markdown("""
    <h1 style='color:#1f77b4;'>🔍 Suspicious Account Detector</h1>
    <p style='color:#444;'>Upload transaction data to detect unusual or suspicious accounts.</p>
""", unsafe_allow_html=True)

# ------------------📁 File Upload------------------ #
uploaded_file = st.file_uploader("📁 Upload CSV file with account data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.markdown("""
        <h3 style='color:#2ca02c;'>📊 Uploaded Data Preview</h3>
    """, unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

    # ------------------🔮 Prediction------------------ #
    predictions = model.predict(df)
    df["prediction"] = predictions
    df["prediction_label"] = df["prediction"].apply(lambda x: "🟥 Suspicious" if x == 1 else "🟩 Normal")

    st.markdown("""
        <h3 style='color:#d62728;'>🔍 Prediction Results</h3>
    """, unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)

    # ------------------📈 KPIs------------------ #
    total_accounts = len(df)
    suspicious_count = (df['prediction'] == 1).sum()
    normal_count = total_accounts - suspicious_count
    suspicious_rate = round((suspicious_count / total_accounts) * 100, 2)

    st.markdown("""
        <h3 style='color:#9467bd;'>📈 Summary KPIs</h3>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔢 Total Accounts", total_accounts)
    col2.metric("🟥 Suspicious", suspicious_count)
    col3.metric("🟩 Normal", normal_count)
    col4.metric("⚠️ Suspicious Rate (%)", suspicious_rate)

    # ------------------📊 Pie Chart------------------ #
    st.markdown("""
        <h3 style='color:#ff7f0e;'>📊 Prediction Summary Chart</h3>
    """, unsafe_allow_html=True)
    summary = df["prediction_label"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(summary, labels=summary.index, autopct='%1.1f%%', startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    # ------------------⬇️ Download Button------------------ #
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv"
    )

else:
    st.warning("👆 Please upload a CSV file to begin.")
