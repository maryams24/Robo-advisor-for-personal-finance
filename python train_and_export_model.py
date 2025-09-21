import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np

# --- 1. Data Loading and Preparation ---

# Create placeholder data for a realistic example.
# In a real project, you would load from your CSV files.
print("Creating placeholder data...")
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

# --- 2. Data Preprocessing & Feature Engineering ---

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create a preprocessor using ColumnTransformer
# It will apply OneHotEncoder to categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# --- 3. Model Training ---

# Create a pipeline that first preprocesses the data and then trains the model.
# Using a RandomForestClassifier, which is suitable for this problem.
print("\nTraining the model...")
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on the training data
model.fit(X_train, y_train)

# --- 4. Model Evaluation ---

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 5. Export Model and Preprocessor ---

# Export the trained pipeline and the list of features.
# The pipeline includes the preprocessor, so it's a single object to save.
print("\nExporting the model and features...")
try:
    joblib.dump(model, 'robo_advisor_model.joblib')
    joblib.dump(features, 'model_features.joblib')
    print("Model and features exported successfully!")
except Exception as e:
    print(f"Error exporting model: {e}")

# Note: The exported files 'robo_advisor_model.joblib' and 'model_features.joblib'
# will be used by the Streamlit application for inference.
