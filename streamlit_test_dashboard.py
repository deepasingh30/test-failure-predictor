import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Test Failure Predictor", layout="wide")
st.title("Test Failure Prediction Dashboard")

uploaded_file = st.file_uploader("Upload test results CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Encode categorical fields
    df['Result'] = df['Result'].map({'Pass': 0, 'Fail': 1})
    df['Module_Encoded'] = LabelEncoder().fit_transform(df['Module_Name'])
    df['Error_Encoded'] = LabelEncoder().fit_transform(df['Error_Message'].fillna('None'))

    features = ['Module_Encoded', 'Code_Churn', 'Run_Count_Since_Fail', 'Error_Encoded']
    X = df[features]
    y = df['Result']

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    st.success("Model trained on uploaded data ✅")

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.subheader("🔍 Feature Importance (SHAP)")
    try:
        shap.summary_plot(shap_values, X, show=False)
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.error(f"Could not display SHAP plot: {e}")

    st.subheader("📈 Predict on New Data")
    predict_file = st.file_uploader("Upload new data to predict (same columns)", type="csv", key="predict")

    if predict_file:
        new_df = pd.read_csv(predict_file)
        new_df['Module_Encoded'] = LabelEncoder().fit_transform(new_df['Module_Name'])
        new_df['Error_Encoded'] = LabelEncoder().fit_transform(new_df['Error_Message'].fillna('None'))
        new_X = new_df[features]

        predictions = model.predict(new_X)
        new_df['Predicted_Result'] = np.where(predictions == 1, 'Fail', 'Pass')
        st.dataframe(new_df[['Test_Case_ID', 'Module_Name', 'Predicted_Result']])

        csv = new_df.to_csv(index=False).encode('utf
