import streamlit as st
import pandas as pd
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

# --- 2. Model and Features Loading ---

@st.cache_resource
def load_model():
    """Loads the pre-trained model and features from disk."""
    try:
        model = joblib.load('robo_advisor_model.joblib')
        features = joblib.load('model_features.joblib')
        return model, features
    except FileNotFoundError:
        st.error("Error: Model files not found. Please run 'train_and_export_model.py' first.")
        return None, None

model, model_features = load_model()

if model is None:
    st.stop()

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
