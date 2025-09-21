import streamlit as st
import pandas as pd
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
    """
    # Create DataFrame from the new data provided by the user
    data_dict = {
        'gender': ['Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Female'],
        'age': [34, 23, 30, 22, 24, 24, 27],
        'investment_avenues': ['Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes'],
        'mutual_funds_rank': [1, 4, 3, 2, 2, 7, 3],
        'equity_market_rank': [2, 3, 6, 1, 1, 5, 6],
        'debentures_rank': [5, 2, 4, 3, 3, 4, 4],
        'government_bonds_rank': [3, 1, 2, 7, 6, 6, 2],
        'fixed_deposits_rank': [7, 5, 5, 6, 4, 3, 5],
        'ppf_rank': [6, 6, 1, 4, 5, 1, 1],
        'gold_rank': [4, 7, 7, 5, 7, 2, 7],
        'stock_market': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes'],
        'factor': ['Returns', 'Locking Period', 'Returns', 'Returns', 'Returns', 'Risk', 'Returns'],
        'objective': ['Capital Appreciation', 'Capital Appreciation', 'Capital Appreciation', 'Income', 'Income', 'Capital Appreciation', 'Capital Appreciation'],
        'purpose': ['Wealth Creation', 'Wealth Creation', 'Wealth Creation', 'Wealth Creation', 'Wealth Creation', 'Wealth Creation', 'Wealth Creation'],
        'duration': ['1-3 years', 'More than 5 years', '3-5 years', 'Less than 1 year', 'Less than 1 year', '1-3 years', '3-5 years'],
        'invest_monitor': ['Monthly', 'Weekly', 'Daily', 'Daily', 'Daily', 'Daily', 'Monthly'],
        'expect': ['20%-30%', '20%-30%', '20%-30%', '10%-20%', '20%-30%', '30%-40%', '20%-30%'],
        'avenue': ['Mutual Fund', 'Mutual Fund', 'Equity', 'Equity', 'Equity', 'Mutual Fund', 'Equity'],
        'savings_objectives': ['Retirement Plan', 'Health Care', 'Retirement Plan', 'Retirement Plan', 'Retirement Plan', 'Retirement Plan', 'Retirement Plan'],
        'reason_equity': ['Capital Appreciation', 'Dividend', 'Capital Appreciation', 'Dividend', 'Capital Appreciation', 'Liquidity', 'Capital Appreciation'],
        'reason_mutual': ['Better Returns', 'Better Returns', 'Tax Benefits', 'Fund Diversification', 'Better Returns', 'Fund Diversification', 'Better Returns'],
        'reason_bonds': ['Safe Investment', 'Safe Investment', 'Assured Returns', 'Tax Incentives', 'Safe Investment', 'Safe Investment', 'Assured Returns'],
        'reason_fd': ['Fixed Returns', 'High Interest Rates', 'Fixed Returns', 'High Interest Rates', 'Risk Free', 'Risk Free', 'High Interest Rates'],
        'info_source': ['Newspapers and Magazines', 'Financial Consultants', 'Television', 'Internet', 'Internet', 'Internet', 'Financial Consultants']
    }
    df = pd.DataFrame(data_dict)

    # Define features (X) and target (y)
    features = ['gender', 'age', 'investment_avenues', 'mutual_funds_rank', 'equity_market_rank', 'debentures_rank', 
                'government_bonds_rank', 'fixed_deposits_rank', 'ppf_rank', 'gold_rank', 'stock_market', 'factor', 
                'objective', 'purpose', 'duration', 'invest_monitor', 'expect']
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

    # Return the trained model, the features, and the full dataframe for lookups
    return model, features, df

# Train and load the model at the start of the app
model, model_features, full_data = train_and_cache_model()


# --- 3. User Input Form ---

with st.form("advisor_form", clear_on_submit=False):
    st.subheader("Tell us about your investment profile")

    # The form must collect all features used by the model
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender:", full_data['gender'].unique())
        age = st.number_input("Age:", min_value=18, max_value=100, value=25)
        investment_avenues = st.selectbox("Do you invest in Investment Avenues?", full_data['investment_avenues'].unique())
        mutual_funds_rank = st.selectbox("Mutual Funds Rank:", sorted(full_data['mutual_funds_rank'].unique()))
        equity_market_rank = st.selectbox("Equity Market Rank:", sorted(full_data['equity_market_rank'].unique()))
        debentures_rank = st.selectbox("Debentures Rank:", sorted(full_data['debentures_rank'].unique()))
        government_bonds_rank = st.selectbox("Government Bonds Rank:", sorted(full_data['government_bonds_rank'].unique()))
        fixed_deposits_rank = st.selectbox("Fixed Deposits Rank:", sorted(full_data['fixed_deposits_rank'].unique()))
        ppf_rank = st.selectbox("Public Provident Fund Rank:", sorted(full_data['ppf_rank'].unique()))
        gold_rank = st.selectbox("Gold Rank:", sorted(full_data['gold_rank'].unique()))
        stock_market = st.selectbox("Do you invest in Stock Market?", full_data['stock_market'].unique())

    with col2:
        factor = st.selectbox("What are the factors considered by you while investing in any instrument?", full_data['factor'].unique())
        objective = st.selectbox("What is your investment objective?", full_data['objective'].unique())
        purpose = st.selectbox("What is your purpose behind investment?", full_data['purpose'].unique())
        duration = st.selectbox("How long do you prefer to keep your money in any investment instrument?", full_data['duration'].unique())
        invest_monitor = st.selectbox("How often do you monitor your investment?", full_data['invest_monitor'].unique())
        expect = st.selectbox("How much return do you expect from any investment instrument?", full_data['expect'].unique())
    
    submitted = st.form_submit_button("Get Personalized Advice")

# --- 4. Prediction and Display ---

if submitted:
    # Create a DataFrame from user inputs to match the model's expected format
    user_data = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'investment_avenues': investment_avenues,
        'mutual_funds_rank': mutual_funds_rank,
        'equity_market_rank': equity_market_rank,
        'debentures_rank': debentures_rank,
        'government_bonds_rank': government_bonds_rank,
        'fixed_deposits_rank': fixed_deposits_rank,
        'ppf_rank': ppf_rank,
        'gold_rank': gold_rank,
        'stock_market': stock_market,
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

        # Show the probability of each class
        probabilities = model.predict_proba(user_data)[0]
        class_labels = model.classes_
        prob_df = pd.DataFrame({'Investment Avenue': class_labels, 'Confidence': probabilities})
        prob_df = prob_df.sort_values('Confidence', ascending=False)
        
        st.info("Confidence Score:")
        st.table(prob_df)

        # Add a bar chart for visual clarity
        st.subheader("Confidence Scores Visualized")
        st.bar_chart(prob_df, x='Investment Avenue', y='Confidence')
        

        # Add contextual information
        st.subheader("Why this recommendation?")
        # Find the row in the original data that most closely matches the user's input
        # This is a simple lookup, not a perfect logic.
        user_input_series = pd.Series(user_data.iloc[0].to_dict())
        match_scores = full_data[model_features].apply(lambda x: (x.astype(str) == user_input_series.astype(str)).sum(), axis=1)
        best_match_index = match_scores.idxmax()
        best_match_row = full_data.loc[best_match_index]

        st.markdown(f"""
        <div class="container">
            <p><strong>Your Investment Objective:</strong> {best_match_row['objective']}</p>
            <p><strong>Reasons for investing in Equity Market:</strong> {best_match_row['reason_equity']}</p>
            <p><strong>Reasons for investing in Mutual Funds:</strong> {best_match_row['reason_mutual']}</p>
            <p><strong>Reasons for investing in Government Bonds:</strong> {best_match_row['reason_bonds']}</p>
            <p><strong>Reasons for investing in Fixed Deposits:</strong> {best_match_row['reason_fd']}</p>
        </div>
        """, unsafe_allow_html=True)
        

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.warning("Please check your input values and try again.")
    
    st.balloons()
