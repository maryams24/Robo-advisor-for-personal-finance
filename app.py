!pip install joblib

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Page config for better look
st.set_page_config(page_title="Finance Advisor", page_icon="💸", layout="centered")

# Custom CSS for style
st.markdown("""
    <style>
        .main {background-color: #f2f6fc;}
        .stButton>button {background-color: #0099ff; color: white; font-size: 18px;}
        .stTextInput, .stSelectbox {font-size: 18px;}
        .stTitle {color: #0099ff;}
        .stSubheader {color: #0077cc;}
        .stMarkdown {font-size: 18px;}
    </style>
    """, unsafe_allow_html=True)

st.title("💸 Finance Advisor Web App")
st.write("Get personalized investment advice based on your preferences.")

# Load data directly from disk
try:
    original_df = pd.read_csv('Original_data.csv')
    finance_df = pd.read_csv('Finance_data.csv')
except Exception as e:
    st.error(f"Error loading data files: {e}")
    st.stop()

# Standardize column names
original_df.columns = original_df.columns.str.strip().str.lower().str.replace(' ', '_')
finance_df.columns = finance_df.columns.str.strip().str.lower().str.replace(' ', '_')

# 1. EDA Section
st.subheader("1. Exploratory Data Analysis (EDA)")
with st.expander("Show EDA"):
    st.write("**Original Data Overview**")
    st.write(original_df.head())
    st.write(original_df.describe())
    st.write("**Finance Data Overview**")
    st.write(finance_df.head())
    st.write(finance_df.describe())
    st.write("**Missing Values in Finance Data**")
    st.write(finance_df.isnull().sum())

# 2. Data Preprocessing Section
st.subheader("2. Data Preprocessing")
with st.expander("Show Preprocessing"):
    # Fill missing values for demonstration (customize as needed)
    finance_df = finance_df.fillna(method='ffill')
    st.write("Filled missing values using forward fill.")
    # Encode categorical columns for demonstration
    if 'gender' in finance_df.columns:
        finance_df['gender'] = finance_df['gender'].map({'Female': 0, 'Male': 1}).fillna(0)
    st.write("Encoded gender column.")

# 3. Feature Engineering Section
st.subheader("3. Feature Engineering")
with st.expander("Show Feature Engineering"):
    # Create age group feature
    if 'age' in finance_df.columns:
        finance_df['age_group'] = pd.cut(finance_df['age'], bins=[18, 30, 50, 100], labels=['Young', 'Mid', 'Senior'])
        st.write("Created age_group feature.")
        st.write(finance_df[['age', 'age_group']].head())
    # Encode age_group
    if 'age_group' in finance_df.columns:
        finance_df['age_group'] = finance_df['age_group'].map({'Young': 0, 'Mid': 1, 'Senior': 2}).fillna(0)

# 4. Model Training Section
st.subheader("4. Model Training")
with st.expander("Show Model Training"):
    # Prepare features and target
    feature_cols = ['gender', 'age']
    if 'age_group' in finance_df.columns:
        feature_cols.append('age_group')
    model_df = finance_df.dropna(subset=['avenue'])
    X = model_df[feature_cols]
    y = model_df['avenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    st.write("Model trained on selected features.")

# 5. Evaluation Section
st.subheader("5. Model Evaluation")
with st.expander("Show Evaluation"):
    y_pred = model.predict(X_test)
    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred))
    st.write("**Confusion Matrix:**")
    st.write(confusion_matrix(y_test, y_pred))

# 6. Export Model Section
st.subheader("6. Export Model")
with st.expander("Show Export"):
    joblib.dump(model, 'finance_advisor_model.pkl')
    st.write("Model exported as 'finance_advisor_model.pkl'.")

# 7. Deployment with Streamlit (User Form)
st.subheader("7. Get Your Investment Advice")
def get_options(col, default=None):
    orig = original_df[col].dropna().unique().tolist() if col in original_df.columns else []
    fin = finance_df[col].dropna().unique().tolist() if col in finance_df.columns else []
    # Convert all values to strings for safe sorting and display
    opts = sorted(list(set([str(x) for x in orig + fin])))
    if default and str(default) not in opts:
        opts = [str(default)] + opts
    return opts if opts else [str(default)] if default else []


with st.form("advisor_form"):
    gender = st.selectbox("Select your gender:", get_options('gender', 'Female'))
    age = st.number_input("Select your age:", min_value=18, max_value=100, value=21)
    submitted = st.form_submit_button("Get Investment Advice")

if submitted:
    # Prepare features for prediction
    gender_encoded = 0 if gender == 'Female' else 1
    age_group = 'Young' if age <= 30 else 'Mid' if age <= 50 else 'Senior'
    age_group_encoded = {'Young': 0, 'Mid': 1, 'Senior': 2}[age_group]
    user_features = np.array([[gender_encoded, age, age_group_encoded]])
    # Load model and predict
    model = joblib.load('finance_advisor_model.pkl')
    prediction = model.predict(user_features)
    st.success(f"Recommended Investment Avenue: **{prediction[0]}**")
    st.info("Recommendation is based on your gender and age. For more personalized advice, expand the feature set.")

# 8. Push Code into GitHub Section
st.subheader("8. Push Code to GitHub")
with st.expander("Show GitHub Instructions"):
    st.markdown("""
    **Steps:**
    1. Initialize git: `git init`
    2. Add files: `git add .`
    3. Commit: `git commit -m "Initial commit"`
    4. Create repo on GitHub and link:  
       `git remote add origin <your-repo-url>`
    5. Push: `git push -u origin main`
    """)

