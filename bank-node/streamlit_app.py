import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# 📌 Set absolute path to model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "suspicious_model.pkl")

# ✅ Load model
model = joblib.load(MODEL_PATH)

# 🎨 Page config
st.set_page_config(page_title="🧠 Suspicious Account Detector", layout="wide")
st.title("🔍 Suspicious Account Detector")

# 📁 Upload CSV
uploaded_file = st.file_uploader("📁 Upload CSV file with account data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data:", df.head())

    # 🤖 Make Predictions
    predictions = model.predict(df)
    df["prediction"] = predictions
    df["prediction_label"] = df["prediction"].apply(lambda x: "🟥 Suspicious" if x == 1 else "🟩 Normal")

    # ✅ KPIs
    total = len(df)
    suspicious = df["prediction"].sum()
    normal = total - suspicious
    rate = round((suspicious / total) * 100, 2)

    st.markdown("### 📊 Prediction KPIs")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔢 Total Accounts", total)
    col2.metric("🟥 Suspicious", suspicious)
    col3.metric("🟩 Normal", normal)
    col4.metric("⚠️ Suspicious Rate", f"{rate}%")

    # 🧾 Show results
    st.dataframe(df)

    # 📊 Pie Chart
    st.subheader("📊 Prediction Summary")
    summary = df["prediction_label"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(summary, labels=summary.index, autopct='%1.1f%%', startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    # 💾 Download Results
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv"
    )
else:
    st.warning("👆 Please upload a CSV file to begin.")
