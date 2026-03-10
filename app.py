import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("credit_risk_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Credit Risk Prediction App")

st.write("Enter applicant details to predict credit risk.")

# Inputs
age = st.number_input("Age", 18, 75)

sex = st.selectbox("Sex", ["male", "female"])

job = st.selectbox(
    "Job",
    [
        "0 - unskilled and non-resident",
        "1 - unskilled and resident",
        "2 - skilled",
        "3 - highly skilled",
    ],
)

housing = st.selectbox("Housing", ["own", "rent", "free"])

saving_accounts = st.selectbox(
    "Saving Accounts",
    ["little", "moderate", "quite rich", "rich", "none"],
)

checking_account = st.selectbox(
    "Checking Account",
    ["little", "moderate", "quite rich", "rich", "none"],
)

credit_amount = st.number_input("Credit Amount")

duration = st.number_input("Duration (months)")

purpose = st.selectbox(
    "Purpose",
    [
        "car",
        "furniture/equipment",
        "radio/TV",
        "domestic appliances",
        "repairs",
        "education",
        "business",
        "vacation/others",
    ],
)

# Convert inputs into dataframe
input_data = pd.DataFrame(
    {
        "Age": [age],
        "Sex": [sex],
        "Job": [int(job[0])],
        "Housing": [housing],
        "Saving accounts": [saving_accounts],
        "Checking account": [checking_account],
        "Credit amount": [credit_amount],
        "Duration": [duration],
        "Purpose": [purpose],
    }
)

# Apply same encoding used during training
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=model_columns, fill_value=0)

# Predict
if st.button("Predict Risk"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("High Credit Risk (Bad Borrower)")
    else:
        st.success("Low Credit Risk (Good Borrower)")