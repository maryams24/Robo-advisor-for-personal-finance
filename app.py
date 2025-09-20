
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
    age = st.number_input("Select your age:", min_value=18, max_value=100, value=21)
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
        'age': str(age),
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

    # Smarter matching logic: score each row by number of matches
    def score_row(row, user_inputs):
        score = 0
        for col, val in user_inputs.items():
            if col in row.index:
                if str(row[col]).lower() == str(val).lower():
                    score += 1
        return score

    finance_df['match_score'] = finance_df.apply(lambda row: score_row(row, user_inputs), axis=1)
    best_match = finance_df.sort_values('match_score', ascending=False).iloc[0]

    recommended_avenue = best_match['avenue'] if 'avenue' in best_match else None
    advice = best_match['advice'] if 'advice' in best_match else None

    # Compare user preference to recommendation
    if recommended_avenue and avenue and recommended_avenue.lower() == avenue.lower():
        st.success(f"Your preferred investment avenue ({avenue}) matches our recommendation!")
        if advice:
            st.write(f"**Advice:** {advice}")
    elif recommended_avenue:
        st.warning(f"Based on your profile, we recommend **{recommended_avenue}** instead of your selected preference ({avenue}).")
        if advice:
            st.write(f"**Reason:** {advice}")
    else:
        st.warning("No specific avenue recommendation found. Consider consulting a financial advisor.")

    # Show reasoning behind recommendation
    st.info("Recommendation is based on your gender, age, investment objective, purpose, duration, monitoring frequency, expected return, and factors considered.")

    # Optionally, show the full matched row for transparency
    with st.expander("See details of matched profile"):
        st.write(best_match)

    # Add animation for positive feedback
    st.balloons()
    st.snow()

    # Optionally, show more info if present
    for col in ['reason_equity', 'reason_mutual', 'reason_bonds', 'reason_fd']:
        if col in best_match and pd.notna(best_match[col]):
            st.write(f"**{col.replace('_', ' ').title()}:** {best_match[col]}")
