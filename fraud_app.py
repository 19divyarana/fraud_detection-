#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained XGBoost model
model = joblib.load("fraud_detection_model.pkl")

# Streamlit UI
st.title("🚀 Bank Loan Fraud Detection using XGBoost")
st.write("Enter loan details to predict if it's fraudulent or not.")

# Input fields
step = st.number_input("Step (Transaction Time Step)", min_value=0, step=1)
amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
oldbalanceOrg = st.number_input("Original Balance Before Transaction", min_value=0.0, step=0.01)
newbalanceOrig = st.number_input("New Balance After Transaction", min_value=0.0, step=0.01)
oldbalanceDest = st.number_input("Receiver's Balance Before Transaction", min_value=0.0, step=0.01)
newbalanceDest = st.number_input("Receiver's Balance After Transaction", min_value=0.0, step=0.01)
isFlaggedFraud = st.selectbox("Is Transaction Flagged as Fraud?", [0, 1])  # Categorical (0 or 1)
transaction_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER"])

# Convert transaction type to numerical
type_mapping = {"CASH_OUT": 1, "TRANSFER": 2}
type_encoded = type_mapping[transaction_type]

# Ensure input matches training feature count
if st.button("Predict Fraud"):
    try:
        # Convert input to correct format
        input_data = np.array([[step, type_encoded, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFlaggedFraud]])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Convert output to binary (0 = Legit, 1 = Fraud)
        fraud_prediction = int(prediction[0])

        if fraud_prediction == 1:
            st.error("🚨 Fraudulent Loan Transaction Detected!")
        else:
            st.success("✅ Transaction is Legitimate.")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")


# In[ ]:




