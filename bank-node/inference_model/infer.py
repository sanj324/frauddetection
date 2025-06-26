import pandas as pd
import joblib
import os

# Define model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/model.pkl")

# Load the saved model
model = joblib.load(MODEL_PATH)

def predict(input_csv_path: str):
    # Load input data
    df = pd.read_csv(input_csv_path)
    print(f"📄 Loaded inference data: {df.shape[0]} rows")

    # Run prediction
    preds = model.predict(df)

    # Output results
    df["prediction"] = preds
    print("✅ Predictions:")
    print(df)

    return df
