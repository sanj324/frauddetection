import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import joblib
import os

# Constants
DATA_PATH = "../../bank-node/bank-data/train.csv"
MODEL_PATH = "E:/sad-fed/bank-node/model/suspicious_model.pkl"
EXPERIMENT_NAME = "suspicious_account_detector"
TARGET_COLUMN = "is_suspicious"

# 1. Load data
df = pd.read_csv(DATA_PATH)
print("🔍 Columns in dataset:", df.columns.tolist())

# 2. Train/Test Split
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")

# 5. Log with MLflow (optional but still included)
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, artifact_path="model")

# 6. Save model manually as .pkl for Streamlit
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")
