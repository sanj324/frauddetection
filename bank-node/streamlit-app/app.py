import streamlit as st
import pandas as pd
import sys
import os

# ✅ Add bank-node path to import inference_model
sys.path.append(os.path.abspath("../bank-node"))

from inference_model.infer import predict

# === UI Logic ===
st.set_page_config(page_title="🔍 Suspicious Account Detector", layout="wide")
st.title("🔍 Suspicious Account Detection via AI")

uploaded_file = st.file_uploader("📤 Upload transaction CSV", type="csv")

if uploaded_file:
    input_path = "bank-node/bank-data/inference_data.csv"

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    # Run inference
    df = predict(input_path)

    # Show results
    st.success("✅ Prediction Completed")
    st.dataframe(df)

    # Download output
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions", csv, "predicted_output.csv", "text/csv")
