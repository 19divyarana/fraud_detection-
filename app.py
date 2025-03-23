#!/usr/bin/env python
# coding: utf-8

# In[7]:


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
amount = st.number_input("Loan Amount", min_value=0.0, step=0.01)
oldbalanceOrg = st.number_input("Original Balance Before Transaction", min_value=0.0, step=0.01)
newbalanceOrig = st.number_input("New Balance After Transaction", min_value=0.0, step=0.01)
transaction_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER"])

# Convert transaction type to numerical
type_mapping = {"CASH_OUT": 1, "TRANSFER": 2}
type_encoded = type_mapping[transaction_type]

# Ensure input is in correct format
if st.button("Predict Fraud"):
    try:
        # Convert input to a NumPy array
        input_data = np.array([[amount, oldbalanceOrg, newbalanceOrig, type_encoded]])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Ensure output is correctly interpreted
        fraud_prediction = int(prediction[0])  # Convert to binary (0 or 1)

        if fraud_prediction == 1:
            st.error("🚨 Fraudulent Loan Transaction Detected!")
        else:
            st.success("✅ Transaction is Legitimate.")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")


# In[ ]:




