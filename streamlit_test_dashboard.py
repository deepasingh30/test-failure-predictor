
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.express as px

st.set_page_config(page_title="Test Failure Insight Dashboard", layout="wide")
st.title("📉 Test Failure Insight & Prediction")

uploaded_file = st.file_uploader("Upload 'TestFailures_Report.xlsx'", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name='Result')
    st.subheader("📄 Raw Failure Data Preview")
    st.dataframe(df.head())

    # Display top exceptions
    st.subheader("⚠️ Top Failure Exceptions")
    top_errors = df['Exception'].value_counts().head(10).reset_index()
    top_errors.columns = ['Exception', 'Count']
    fig_ex = px.bar(top_errors, x='Count', y='Exception', orientation='h', title="Most Frequent Exceptions")
    st.plotly_chart(fig_ex, use_container_width=True)

    # Display top failure-prone feature files
    st.subheader("🧩 Feature Files Causing Most Failures")
    top_features = df['Feature File Name'].value_counts().head(10).reset_index()
    top_features.columns = ['Feature File Name', 'Failure Count']
    fig_feat = px.bar(top_features, x='Failure Count', y='Feature File Name', orientation='h', title="Top Failing Features")
    st.plotly_chart(fig_feat, use_container_width=True)

    # Predictive Modeling: Can we predict the Feature File Name from Exception
    st.subheader("🤖 Predicting Failure Feature from Exception Text")
    df = df.dropna(subset=['Exception', 'Feature File Name'])

    X = df['Exception']
    y = df['Feature File Name']

    vectorizer = CountVectorizer(max_features=1000)
    X_vec = vectorizer.fit_transform(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y_enc, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    st.success(f"Model Accuracy on Test Data: {acc:.2f}")

    # Show predictions
    st.subheader("🔍 Predict Feature File from a New Exception")
    user_input = st.text_area("Enter an exception message:")
    if user_input:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)
        predicted_feature = le.inverse_transform(prediction)[0]
        st.info(f"📂 Predicted Feature File: **{predicted_feature}**")
