import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import joblib

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
    This function now loads and merges the two provided CSV files.
    """
    try:
        # Load the two provided CSV files with explicit handling for missing values
        na_values = ['', ' ', 'nan', 'N/A', 'NA', 'na']
        original_df = pd.read_csv('Original_data.csv', na_values=na_values)
        finance_df = pd.read_csv('Finance_data.csv', na_values=na_values)
    except FileNotFoundError:
        st.error("Error: CSV data files not found. Please ensure 'Original_data.csv' and 'Finance_data.csv' are in the same directory.")
        st.stop()

    # Standardize column names for merging
    original_df.columns = original_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('?', '').str.replace('[', '').str.replace(']', '').str.replace('.', '_', regex=False)
    finance_df.columns = finance_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('?', '').str.replace('[', '').str.replace(']', '').str.replace('.', '_', regex=False)
    
    # Map the messy column names to a clean, consistent set
    column_mapping = {
        'do_you_invest_in_investment_avenues': 'investment_avenues',
        'what_do_you_think_are_the_best_options_for_investing_your_money_rank_mutual_funds': 'mutual_funds_rank',
        'what_do_you_think_are_the_best_options_for_investing_your_money_rank_equity_market': 'equity_market_rank',
        'what_do_you_think_are_the_best_options_for_investing_your_money_rank_debentures': 'debentures_rank',
        'what_do_you_think_are_the_best_options_for_investing_your_money_rank_government_bonds': 'government_bonds_rank',
        'what_do_you_think_are_the_best_options_for_investing_your_money_rank_fixed_deposits': 'fixed_deposits_rank',
        'what_do_you_think_are_the_best_options_for_investing_your_money_rank_public_provident_fund': 'ppf_rank',
        'what_do_you_think_are_the_best_options_for_investing_your_money_rank_gold': 'gold_rank',
        'do_you_invest_in_stock_market_': 'stock_market',
        'what_are_the_factors_considered_by_you_while_investing_in_any_instrument_': 'factor',
        'what_is_your_investment_objective_': 'objective',
        'what_is_your_purpose_behind_investment_': 'purpose',
        'how_long_do_you_prefer_to_keep_your_money_in_any_investment_instrument_': 'duration',
        'how_often_do_you_monitor_your_investment_': 'invest_monitor',
        'how_much_return_do_you_expect_from_any_investment_instrument_': 'expect',
        'which_investment_avenue_do_you_mostly_invest_in_': 'avenue',
        'what_are_your_savings_objectives_': 'savings_objectives',
        'reasons_for_investing_in_equity_market': 'reason_equity',
        'reasons_for_investing_in_mutual_funds': 'reason_mutual',
        'reasons_for_investing_in_government_bonds': 'reason_bonds',
        'reasons_for_investing_in_fixed_deposits_': 'reason_fd',
        'your_sources_of_information_for_investments_is_': 'info_source'
    }

    # Rename columns in both dataframes
    original_df = original_df.rename(columns=column_mapping)
    finance_df = finance_df.rename(columns={'mutual_funds': 'mutual_funds_rank', 'equity_market': 'equity_market_rank',
                                            'debentures': 'debentures_rank', 'government_bonds': 'government_bonds_rank',
                                            'fixed_deposits': 'fixed_deposits_rank', 'ppf': 'ppf_rank', 'gold': 'gold_rank',
                                            'stock_marktet': 'stock_market', 'invest_monitor': 'invest_monitor', 'expect': 'expect',
                                            'avenue': 'avenue', 'source': 'info_source'})
    
    # Consolidate and drop any duplicate columns
    full_data = pd.concat([original_df, finance_df], ignore_index=True)
    full_data = full_data.loc[:, ~full_data.columns.duplicated()]

    # Define features (X) and target (y)
    features = ['gender', 'age', 'investment_avenues', 'mutual_funds_rank', 'equity_market_rank', 'debentures_rank', 
                'government_bonds_rank', 'fixed_deposits_rank', 'ppf_rank', 'gold_rank', 'stock_market', 'factor', 
                'objective', 'purpose', 'duration', 'invest_monitor', 'expect']
    target = 'avenue'
    
    # --- Robust Data Cleaning ---
    
    # Identify categorical and numerical features
    categorical_features = full_data[features].select_dtypes(include=['object']).columns.tolist()
    rank_cols = [col for col in full_data.columns if '_rank' in col]

    # Explicitly convert categorical features to string to prevent any non-string values
    for col in categorical_features:
        full_data[col] = full_data[col].astype(str)

    # Convert rank columns to numeric, coercing errors to NaN
    for col in rank_cols:
        full_data[col] = pd.to_numeric(full_data[col], errors='coerce')

    # Drop any rows where a feature value is NaN, ensuring a clean dataset for training
    full_data.dropna(subset=features, inplace=True)
    
    X = full_data[features]
    y = full_data[target]
    
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
    return model, features, full_data


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
