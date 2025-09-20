
import streamlit as st
import pandas as pd

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

# Helper to get options from data
def get_options(col, default=None):
    orig = original_df[col].dropna().unique().tolist() if col in original_df.columns else []
    fin = finance_df[col].dropna().unique().tolist() if col in finance_df.columns else []
    opts = sorted(list(set(orig + fin)))
    if default and default not in opts:
        opts = [default] + opts
    return opts if opts else [default] if default else []

with st.form("advisor_form"):
    gender = st.selectbox("Select your gender:", get_options('gender', 'Female'))
    age = st.text_input("Select your age:", value="21")
    investment_avenues = st.selectbox("Do you invest in Investment Avenues?", get_options('investment_avenues', 'No'))
    mutual_funds = st.selectbox("Do you invest in Mutual Funds?", get_options('mutual_funds', '1'))
    equity_market = st.selectbox("Do you invest in Equity Market?", get_options('equity_market', '1'))
    debentures = st.selectbox("Do you invest in Debentures?", get_options('debentures', '1'))
    government_bonds = st.selectbox("Do you invest in Government Bonds?", get_options('government_bonds', '1'))
    fixed_deposits = st.selectbox("Do you invest in Fixed Deposits?", get_options('fixed_deposits', '1'))
    ppf = st.selectbox("Do you invest in PPF?", get_options('ppf', '1'))
    gold = st.selectbox("Do you invest in Gold?", get_options('gold', '2'))
    stock_market = st.selectbox("Do you invest in Stock Market?", get_options('stock_marktet', 'No'))  # spelling as per your data
    factor = st.selectbox("What is your main investment factor?", get_options('factor', 'Locking Period'))
    objective = st.selectbox("What is your investment objective?", get_options('objective', 'Capital Appreciation'))
    purpose = st.selectbox("What is your investment purpose?", get_options('purpose', 'Returns'))
    duration = st.selectbox("What is your investment duration?", get_options('duration', '1-3 years'))
    invest_monitor = st.selectbox("How often do you monitor investments?", get_options('invest_monitor', 'Monthly'))
    expect = st.selectbox("What returns do you expect?", get_options('expect', '10%-20%'))
    avenue = st.selectbox("Preferred investment avenue?", get_options('avenue', 'Public Provident Fund'))

    submitted = st.form_submit_button("Get Investment Advice")

if submitted:
    # Show user selections
    st.subheader("Your Selections")
    st.markdown(f"""
    - **Gender:** {gender}
    - **Age:** {age}
    - **Investment Avenues:** {investment_avenues}
    - **Mutual Funds:** {mutual_funds}
    - **Equity Market:** {equity_market}
    - **Debentures:** {debentures}
    - **Government Bonds:** {government_bonds}
    - **Fixed Deposits:** {fixed_deposits}
    - **PPF:** {ppf}
    - **Gold:** {gold}
    - **Stock Market:** {stock_market}
    - **Factor:** {factor}
    - **Objective:** {objective}
    - **Purpose:** {purpose}
    - **Duration:** {duration}
    - **Investment Monitoring:** {invest_monitor}
    - **Expected Returns:** {expect}
    - **Preferred Avenue:** {avenue}
    """)

    # Prepare user input for matching
    user_inputs = {
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
        'stock_marktet': stock_market,  # spelling as per your data
        'factor': factor,
        'objective': objective,
        'purpose': purpose,
        'duration': duration,
        'invest_monitor': invest_monitor,
        'expect': expect,
        'avenue': avenue,
    }

    # Matching logic: try to find best match in finance_df
    def find_best_match(finance_df, user_inputs):
        mask = pd.Series([True] * len(finance_df))
        for col, val in user_inputs.items():
            if col in finance_df.columns:
                mask &= (finance_df[col].astype(str).str.lower() == str(val).lower())
        matches = finance_df[mask]
        if not matches.empty:
            return matches.iloc[0]
        # If no exact match, find partial matches (e.g., match on avenue, objective, factor)
        for key in ['avenue', 'objective', 'factor']:
            if key in finance_df.columns and user_inputs[key]:
                partial = finance_df[finance_df[key].astype(str).str.lower() == str(user_inputs[key]).lower()]
                if not partial.empty:
                    return partial.iloc[0]
        # Fallback: first row
        return finance_df.iloc[0]

    match = find_best_match(finance_df, user_inputs)

    # Show advice and suggested avenue
    st.subheader("Personalized Investment Advice")
    advice = match['advice'] if 'advice' in match else None
    suggested_avenue = match['avenue'] if 'avenue' in match else None

    if advice:
        st.success(f"**Advice:** {advice}")
    if suggested_avenue:
        st.info(f"**Suggested Avenue:** {suggested_avenue}")
    if not advice and not suggested_avenue:
        st.warning("No specific advice found. Consider consulting a financial advisor.")

    # Add animation for positive feedback
    st.balloons()
    st.snow()

    # Optionally, show more info if present
    for col in ['reason_equity', 'reason_mutual', 'reason_bonds', 'reason_fd']:
        if col in match and pd.notna(match[col]):
            st.write(f"**{col.replace('_', ' ').title()}:** {match[col]}")
