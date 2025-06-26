import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "suspicious_model.pkl")

model = joblib.load(MODEL_PATH)


MODEL_PATH = "bank-node/model/suspicious_model.pkl"


# Load model
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="🧠 Suspicious Account Detector", layout="wide")
st.title("🔍 Suspicious Account Detector")

uploaded_file = st.file_uploader("📁 Upload CSV file with account data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data:", df.head())

    # Predict
    predictions = model.predict(df)
    df["prediction"] = predictions
    df["prediction_label"] = df["prediction"].apply(lambda x: "🟥 Suspicious" if x == 1 else "🟩 Normal")

    st.success("✅ Prediction Complete")
    st.dataframe(df)

    # Pie chart
    summary = df["prediction_label"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(summary, labels=summary.index, autopct='%1.1f%%', startangle=90)
    ax.axis("equal")
    st.subheader("📊 Prediction Summary")
    st.pyplot(fig)

    # Download results
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv"
    )
else:
    st.warning("👆 Please upload a CSV file to begin.")
