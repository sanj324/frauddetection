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
FEATURES_PATH = "E:/sad-fed/bank-node/model/feature_columns.pkl"
EXPERIMENT_NAME = "suspicious_account_detector"
TARGET_COLUMN = "is_suspicious"

# 1. Load data
df = pd.read_csv(DATA_PATH)
print("🔍 Columns in dataset:", df.columns.tolist())

# 2. Split data into features and target
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# 3. Save the feature column names
os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, FEATURES_PATH)
print(f"📁 Feature columns saved at {FEATURES_PATH}")

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")

# 7. Log to MLflow (optional)
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, artifact_path="model")

# 8. Save model manually for Streamlit
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")
