
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Finance Advisor", page_icon="💸", layout="centered")

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

# --- BACKEND: EDA, Preprocessing, Feature Engineering, Model Training ---
try:
    finance_df = pd.read_csv('Finance_data.csv')
except Exception as e:
    st.error(f"Error loading data file: {e}")
    st.stop()

# Standardize column names
finance_df.columns = finance_df.columns.str.strip().str.lower().str.replace(' ', '_')

# Fill missing values
finance_df = finance_df.fillna(method='ffill')

# Encode categorical columns (example: gender)
if 'gender' in finance_df.columns:
    finance_df['gender_encoded'] = finance_df['gender'].map({'Female': 0, 'Male': 1}).fillna(0)

# Feature engineering: age group
if 'age' in finance_df.columns:
    finance_df['age_group'] = pd.cut(finance_df['age'].astype(float), bins=[18, 30, 50, 100], labels=['Young', 'Mid', 'Senior'])
    finance_df['age_group_encoded'] = finance_df['age_group'].map({'Young': 0, 'Mid': 1, 'Senior': 2}).fillna(0)

# Select features for model
feature_cols = [
    'gender_encoded', 'age', 'age_group_encoded',
    'do_you_invest_in_investment_avenues',
    'what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[mutual_funds]',
    'what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[equity_market]',
    'what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[debentures]',
    'what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[government_bonds]',
    'what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[fixed_deposits]',
    'what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[public_provident_fund]',
    'what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[gold]',
    'do_you_invest_in_stock_market',
    'what_are_the_factors_considered_by_you_while_investing_in_any_instrument',
    'what_is_your_investment_objective',
    'what_is_your_purpose_behind_investment',
    'how_long_do_you_prefer_to_keep_your_money_in_any_investment_instrument',
    'how_often_do_you_monitor_your_investment',
    'how_much_return_do_you_expect_from_any_investment_instrument',
    'which_investment_avenue_do_you_mostly_invest_in',
    'what_are_your_savings_objectives',
    'reasons_for_investing_in_equity_market',
    'reasons_for_investing_in_mutual_funds',
    'reasons_for_investing_in_government_bonds',
    'reasons_for_investing_in_fixed_deposits',
    'your_sources_of_information_for_investments_is'
]

# Encode all categorical features numerically for model training
for col in feature_cols:
    if col in finance_df.columns and finance_df[col].dtype == 'O':
        finance_df[col] = pd.factorize(finance_df[col])[0]

# Drop rows with missing target ('which_investment_avenue_do_you_mostly_invest_in')
model_df = finance_df.dropna(subset=['which_investment_avenue_do_you_mostly_invest_in'])

X = model_df[feature_cols]
y = model_df['which_investment_avenue_do_you_mostly_invest_in']

# Train model (Random Forest)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- FRONTEND: User Form and Output ---
def get_options(col, default=None):
    opts = finance_df[col].dropna().unique().tolist() if col in finance_df.columns else []
    opts = sorted([str(x) for x in opts])
    if default and str(default) not in opts:
        opts = [str(default)] + opts
    return opts if opts else [str(default)] if default else []

with st.form("advisor_form"):
    gender = st.selectbox("Select your gender:", get_options('gender', 'Female'))
    age = st.number_input("Select your age:", min_value=18, max_value=100, value=21)
    invest_avenues = st.selectbox("Do you invest in Investment Avenues?", get_options('do_you_invest_in_investment_avenues', 'No'))
    mf_rank = st.selectbox("Mutual Funds (Rank in order of preference):", get_options('what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[mutual_funds]', '1'))
    eq_rank = st.selectbox("Equity Market (Rank in order of preference):", get_options('what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[equity_market]', '1'))
    deb_rank = st.selectbox("Debentures (Rank in order of preference):", get_options('what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[debentures]', '1'))
    gb_rank = st.selectbox("Government Bonds (Rank in order of preference):", get_options('what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[government_bonds]', '1'))
    fd_rank = st.selectbox("Fixed Deposits (Rank in order of preference):", get_options('what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[fixed_deposits]', '1'))
    ppf_rank = st.selectbox("Public Provident Fund (Rank in order of preference):", get_options('what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[public_provident_fund]', '1'))
    gold_rank = st.selectbox("Gold (Rank in order of preference):", get_options('what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[gold]', '1'))
    stock_market = st.selectbox("Do you invest in Stock Market?", get_options('do_you_invest_in_stock_market', 'No'))
    factors = st.selectbox("Factors considered while investing:", get_options('what_are_the_factors_considered_by_you_while_investing_in_any_instrument', 'Returns'))
    objective = st.selectbox("Investment objective:", get_options('what_is_your_investment_objective', 'Capital Appreciation'))
    purpose = st.selectbox("Purpose behind investment:", get_options('what_is_your_purpose_behind_investment', 'Wealth Creation'))
    duration = st.selectbox("Preferred duration for investment:", get_options('how_long_do_you_prefer_to_keep_your_money_in_any_investment_instrument', '1-3 years'))
    monitor = st.selectbox("How often do you monitor your investment?", get_options('how_often_do_you_monitor_your_investment', 'Monthly'))
    expected_return = st.selectbox("Expected return:", get_options('how_much_return_do_you_expect_from_any_investment_instrument', '10%-20%'))
    mostly_invest = st.selectbox("Which investment avenue do you mostly invest in?", get_options('which_investment_avenue_do_you_mostly_invest_in', 'Mutual Fund'))
    savings_obj = st.selectbox("What are your savings objectives?", get_options('what_are_your_savings_objectives', 'Retirement Plan'))
    reason_equity = st.selectbox("Reasons for investing in Equity Market:", get_options('reasons_for_investing_in_equity_market', 'Capital Appreciation'))
    reason_mf = st.selectbox("Reasons for investing in Mutual Funds:", get_options('reasons_for_investing_in_mutual_funds', 'Better Returns'))
    reason_gb = st.selectbox("Reasons for investing in Government Bonds:", get_options('reasons_for_investing_in_government_bonds', 'Safe Investment'))
    reason_fd = st.selectbox("Reasons for investing in Fixed Deposits:", get_options('reasons_for_investing_in_fixed_deposits', 'Fixed Returns'))
    info_source = st.selectbox("Your sources of information for investments:", get_options('your_sources_of_information_for_investments_is', 'Internet'))

    submitted = st.form_submit_button("Get Investment Advice")

if submitted:
    gender_encoded = 0 if gender == 'Female' else 1
    age_group = 'Young' if age <= 30 else 'Mid' if age <= 50 else 'Senior'
    age_group_encoded = {'Young': 0, 'Mid': 1, 'Senior': 2}[age_group]

    def factorize_input(col, val):
        vals = finance_df[col].dropna().unique().tolist()
        vals = [str(x) for x in vals]
        try:
            return vals.index(str(val))
        except ValueError:
            return 0

    user_features = [
        gender_encoded,
        age,
        age_group_encoded,
        factorize_input('do_you_invest_in_investment_avenues', invest_avenues),
        factorize_input('what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[mutual_funds]', mf_rank),
        factorize_input('what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[equity_market]', eq_rank),
        factorize_input('what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[debentures]', deb_rank),
        factorize_input('what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[government_bonds]', gb_rank),
        factorize_input('what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[fixed_deposits]', fd_rank),
        factorize_input('what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[public_provident_fund]', ppf_rank),
        factorize_input('what_do_you_think_are_the_best_options_for_investing_your_money_(rank_in_order_of_preference)_[gold]', gold_rank),
        factorize_input('do_you_invest_in_stock_market', stock_market),
        factorize_input('what_are_the_factors_considered_by_you_while_investing_in_any_instrument', factors),
        factorize_input('what_is_your_investment_objective', objective),
        factorize_input('what_is_your_purpose_behind_investment', purpose),
        factorize_input('how_long_do_you_prefer_to_keep_your_money_in_any_investment_instrument', duration),
        factorize_input('how_often_do_you_monitor_your_investment', monitor),
        factorize_input('how_much_return_do_you_expect_from_any_investment_instrument', expected_return),
        factorize_input('which_investment_avenue_do_you_mostly_invest_in', mostly_invest),
        factorize_input('what_are_your_savings_objectives', savings_obj),
        factorize_input('reasons_for_investing_in_equity_market', reason_equity),
        factorize_input('reasons_for_investing_in_mutual_funds', reason_mf),
        factorize_input('reasons_for_investing_in_government_bonds', reason_gb),
        factorize_input('reasons_for_investing_in_fixed_deposits', reason_fd),
        factorize_input('your_sources_of_information_for_investments_is', info_source)
    ]

    user_features = np.array(user_features).reshape(1, -1)
    prediction = model.predict(user_features)
    avenue_vals = finance_df['which_investment_avenue_do_you_mostly_invest_in'].dropna().unique().tolist()
    recommended_avenue = avenue_vals[prediction[0]] if prediction[0] < len(avenue_vals) else str(prediction[0])

    st.success(f"Recommended Investment Avenue: **{recommended_avenue}**")
    st.info("Recommendation is based on your profile and our trained model.")

    # Graph: Distribution of recommended avenues for similar profiles
    age_group_series = pd.cut(finance_df['age'].astype(float), bins=[18, 30, 50, 100], labels=['Young', 'Mid', 'Senior'])
    similar = finance_df[
        (finance_df['gender_encoded'] == gender_encoded) &
        (age_group_series == age_group)
    ]
    if not similar.empty:
        fig, ax = plt.subplots()
        sns.countplot(y='which_investment_avenue_do_you_mostly_invest_in', data=similar,
                      order=similar['which_investment_avenue_do_you_mostly_invest_in'].value_counts().index, ax=ax)
        ax.set_title("Investment Avenue Distribution (Similar Profiles)")
        st.pyplot(fig)
    else:
        st.info("Not enough similar profiles to show a distribution graph.")

    st.balloons()
    st.snow()
