#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import joblib
import pandas as pd

# Load the trained XGBoost model
model = joblib.load("fraud_detection_model.pkl")

# Streamlit UI
st.title("🚀 Bank Loan Fraud Detection using XGBoost")
st.write("Enter loan details to predict if it's fraudulent or not.")

# Input fields
amount = st.number_input("Loan Amount", min_value=0)
oldbalanceOrg = st.number_input("Original Balance Before Transaction", min_value=0)
newbalanceOrig = st.number_input("New Balance After Transaction", min_value=0)
transaction_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER"])

# Convert transaction type to numerical
type_mapping = {"CASH_OUT": 1, "TRANSFER": 2}
type_encoded = type_mapping[transaction_type]

# Predict button
if st.button("Predict Fraud"):
    input_data = pd.DataFrame([[amount, oldbalanceOrg, newbalanceOrig, type_encoded]],
                              columns=["amount", "oldbalanceOrg", "newbalanceOrig", "type"])
    
    prediction = model.predict(input_data)
    fraud_prediction = int(prediction[0])  # Convert to binary (0 or 1)
    
    if fraud_prediction == 1:
        st.error("🚨 Fraudulent Loan Transaction Detected!")
    else:
        st.success("✅ Transaction is Legitimate.")


# In[ ]:




