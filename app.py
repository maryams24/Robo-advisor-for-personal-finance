import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import numpy as np

# --- 1. Page Configuration and Styling ---

st.set_page_config(page_title="Robo-Advisor", page_icon="📈", layout="centered")

# Custom CSS for a clean look
st.markdown("""
    <style>
        .main {background-color: #f0f4f8; padding: 20px;}
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px;
            padding: 12px 24px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {background-color: #45a049;}
        .stTextInput, .stSelectbox, .stNumberInput {
            border-radius: 8px;
            border: 1px solid #ccc;
            padding: 10px;
        }
        .stTitle {color: #1a73e8; text-align: center;}
        .stSubheader {color: #3367d6;}
        .container {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 12px;
            background-color: white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.title("📈 AI-Powered Financial Advisor")
st.write("Get personalized investment advice based on your profile.")


# --- 2. Model Training and Caching ---

@st.cache_resource
def train_and_cache_model():
    """
    Trains the machine learning model and caches it to prevent retraining on every interaction.
    This function combines the logic from train_and_export_model.py.
    """
    # Create placeholder data for a realistic example.
    data = {
        'gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female'],
        'age': [25, 35, 45, 55, 30, 40, 50, 60, 22, 33],
        'investment_avenues': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
        'mutual_funds': ['1', '1', '0', '1', '1', '0', '1', '0', '1', '0'],
        'equity_market': ['1', '0', '1', '0', '1', '0', '1', '0', '1', '0'],
        'debentures': ['0', '0', '1', '0', '0', '0', '1', '0', '0', '1'],
        'government_bonds': ['0', '1', '1', '0', '0', '1', '1', '0', '1', '0'],
        'fixed_deposits': ['1', '0', '0', '1', '1', '0', '0', '1', '1', '0'],
        'ppf': ['1', '1', '0', '0', '1', '1', '0', '0', '1', '0'],
        'gold': ['2', '1', '2', '2', '1', '2', '2', '1', '1', '2'],
        'stock_marktet': ['No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes'],
        'factor': ['Returns', 'Liquidity', 'Safety', 'Liquidity', 'Returns', 'Returns', 'Safety', 'Safety', 'Liquidity', 'Safety'],
        'objective': ['Capital Appreciation', 'Regular Income', 'Tax Benefits', 'Capital Appreciation', 'Capital Appreciation', 'Tax Benefits', 'Regular Income', 'Capital Appreciation', 'Regular Income', 'Tax Benefits'],
        'purpose': ['Returns', 'Returns', 'Returns', 'Returns', 'Returns', 'Returns', 'Returns', 'Returns', 'Returns', 'Returns'],
        'duration': ['1-3 years', '3-5 years', '1-3 years', 'More than 5 years', '1-3 years', '3-5 years', '1-3 years', 'More than 5 years', '1-3 years', '3-5 years'],
        'invest_monitor': ['Monthly', 'Quarterly', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Weekly', 'Monthly', 'Daily', 'Monthly'],
        'expect': ['10%-20%', '10%-20%', '20%-30%', '5%-10%', '10%-20%', '10%-20%', '20%-30%', '5%-10%', '10%-20%', '10%-20%'],
        'avenue': ['Public Provident Fund', 'Equity Market', 'Gold', 'Fixed Deposits', 'Public Provident Fund', 'Equity Market', 'Gold', 'Fixed Deposits', 'Public Provident Fund', 'Equity Market']
    }
    df = pd.DataFrame(data)

    # Define features (X) and target (y)
    features = ['gender', 'age', 'investment_avenues', 'mutual_funds', 'equity_market', 'debentures', 'government_bonds',
                'fixed_deposits', 'ppf', 'gold', 'stock_marktet', 'factor', 'objective', 'purpose', 'duration',
                'invest_monitor', 'expect']
    target = 'avenue'
    
    X = df[features]
    y = df[target]

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Create and train a machine learning pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X, y)

    return model, features

# Train and load the model at the start of the app
model, model_features = train_and_cache_model()

# --- 3. User Input Form ---

with st.form("advisor_form", clear_on_submit=False):
    st.subheader("Tell us about your investment profile")

    # The form must collect all features used by the model
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Select your gender:", ['Female', 'Male'])
        age = st.number_input("Select your age:", min_value=18, max_value=100, value=25)
        investment_avenues = st.selectbox("Do you invest?", ['Yes', 'No'])
        mutual_funds = st.selectbox("Mutual Funds:", ['1', '0'])
        equity_market = st.selectbox("Equity Market:", ['1', '0'])
        debentures = st.selectbox("Debentures:", ['1', '0'])
        government_bonds = st.selectbox("Government Bonds:", ['1', '0'])
        fixed_deposits = st.selectbox("Fixed Deposits:", ['1', '0'])
        ppf = st.selectbox("PPF:", ['1', '0'])

    with col2:
        gold = st.selectbox("Gold:", ['2', '1'])
        stock_market = st.selectbox("Stock Market:", ['Yes', 'No'])
        factor = st.selectbox("Main investment factor:", ['Returns', 'Liquidity', 'Safety'])
        objective = st.selectbox("Investment objective:", ['Capital Appreciation', 'Regular Income', 'Tax Benefits'])
        purpose = st.selectbox("Investment purpose:", ['Returns'])
        duration = st.selectbox("Investment duration:", ['1-3 years', '3-5 years', 'More than 5 years'])
        invest_monitor = st.selectbox("How often do you monitor:", ['Monthly', 'Quarterly', 'Daily', 'Weekly'])
        expect = st.selectbox("Expected returns:", ['10%-20%', '20%-30%', '5%-10%'])
    
    submitted = st.form_submit_button("Get Personalized Advice")

# --- 4. Prediction and Display ---

if submitted:
    # Create a DataFrame from user inputs to match the model's expected format
    user_data = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'investment_avenues': investment_avenues,
        'mutual_funds': mutual_funds,
        'equity_market': equity_market,
        'debentures': debentures,
        'government_bonds': government_bonds,
        'fixed_deposits': fixed_deposits,
        'ppf': ppf,
        'gold': gold,
        'stock_marktet': stock_market,
        'factor': factor,
        'objective': objective,
        'purpose': purpose,
        'duration': duration,
        'invest_monitor': invest_monitor,
        'expect': expect
    }])

    # Ensure the columns are in the same order as the model's training data
    user_data = user_data[model_features]

    # Make a prediction
    try:
        prediction = model.predict(user_data)
        st.subheader("Your Personalized Investment Recommendation")
        st.success(f"Based on your profile, we recommend **{prediction[0]}**.")

        # Optionally, show the probability of each class
        probabilities = model.predict_proba(user_data)[0]
        class_labels = model.classes_
        prob_df = pd.DataFrame({'Investment Avenue': class_labels, 'Confidence': probabilities})
        prob_df = prob_df.sort_values('Confidence', ascending=False)

        st.info("Confidence Score:")
        st.table(prob_df)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.warning("Please check your input values and try again.")
    
    st.balloons()
