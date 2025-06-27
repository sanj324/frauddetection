
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from streamlit_shap import st_shap

# Sample Data Load (Replace with your real data)
df = pd.DataFrame({
    'account_age_days': [100, 200, 150, 300, 250]*3,
    'balance': [5000, 3000, 4000, 10000, 6000]*3,
    'last_5_days_avg': [200, 220, 210, 250, 230]*3,
    'txn_count': [5, 10, 7, 15, 8]*3,
    'label': [0, 1, 0, 1, 0]*3
})

df_features = df.drop(columns=['label'])
labels = df['label']

X_train, X_test, y_train, y_test = train_test_split(df_features, labels, test_size=0.2, random_state=42)

# Train LightGBM model
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

st.title("🧠 Global Feature Impact (SHAP)")

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_features)

    class_index = 1 if isinstance(shap_values, list) and len(shap_values) > 1 else 0
    class_shap_values = shap_values[class_index]
    class_shap_values = class_shap_values[:len(df_features)]

    st.code(f"SHAP shape: {class_shap_values.shape}")
    st.code(f"Input shape: {df_features.shape}")

    if class_shap_values.shape == df_features.shape:
        fig_summary = plt.figure()
        shap.summary_plot(class_shap_values, df_features, show=False)
        st.pyplot(fig_summary)
    else:
        st.warning("⚠️ SHAP value shape mismatch. Cannot plot summary.")

    st.markdown("### 🔍 Record-Level SHAP Force Plot")
    for i in range(min(3, len(df_features))):
        st.markdown(f"**Record {i+1}**")
        try:
            st_shap(
                shap.force_plot(
                    base_value=explainer.expected_value[class_index],
                    shap_values=class_shap_values[i],
                    features=df_features.iloc[i],
                    matplotlib=False
                ),
                height=300
            )
        except Exception as e:
            st.warning(f"⚠️ Could not render force plot for record {i+1}: {e}")
except Exception as e:
    st.warning(f"⚠️ Could not generate SHAP plots: {e}")
