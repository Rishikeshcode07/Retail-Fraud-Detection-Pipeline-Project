# ==============================================================================
# WEEK 4 - CELL 2: TRANSACTION TESTING SUITE (NORMAL VS FRAUD SCENARIOS)
# ==============================================================================
# WHAT THIS SCRIPT DOES:
# This script simulates a credit card terminal in the real world. It sends two 
# distinct transactions to our Flask API to prove that the system can accurately 
# differentiate between safe activity and fraudulent activity.

import requests
import json
import pandas as pd

# The local address of our Flask web server
API_URL = "http://127.0.0.1:5000/predict"

print("Loading test dataset...\n")
# We load our testing data to grab a baseline, normal transaction.
df_test = pd.read_csv('X_test_final.csv')


# ==============================================================================
# SCENARIO 1: THE NORMAL TRANSACTION
# ==============================================================================
print("======================================================")
print("SCENARIO 1: PROCESSING A NORMAL, SAFE TRANSACTION")
print("======================================================")

# We take the very first row of our testing data, which represents a standard, low-value purchase.
normal_transaction = df_test.iloc[0].to_dict()

# Send the normal data to the Flask API
print(f"Sending normal transaction data for Account: {normal_transaction.get('acct_num', 'Unknown')}...")
response_normal = requests.post(API_URL, json=normal_transaction)

# Parse and display the result
if response_normal.status_code == 200:
    result_normal = response_normal.json()
    print("\n--- API RESPONSE FOR NORMAL TRANSACTION ---")
    print(json.dumps(result_normal, indent=4))
else:
    print(f"Error connecting to API. Status code: {response_normal.status_code}")
    print(f"Server Reason: {response_normal.text}")


print("\n\n")


# ==============================================================================
# SCENARIO 2: THE FORCED FRAUD TRANSACTION
# ==============================================================================
print("======================================================")
print("SCENARIO 2: PROCESSING A HIGH-RISK FRAUD TRANSACTION")
print("======================================================")

# We take a copy of that same transaction, but act as a hacker and manipulate the values.
fraud_transaction = df_test.iloc[0].copy().to_dict()

# We ONLY change 'amt' because we know this column name is recognized by the ML model.
# By making it massively high, the Isolation Forest and Random Forest will flag it.
# fraud_transaction['amt'] = 45000.00 

fraud_transaction['amt'] = 99999.00
fraud_transaction['customer_num_trans_1_day'] = 500.0
fraud_transaction['customer_num_trans_7_day'] = 1500.0
fraud_transaction['customer_num_trans_30_day'] = 3000.0
fraud_transaction['merchant_risk_1_day'] = 100.0
fraud_transaction['merchant_risk_7_day'] = 100.0
fraud_transaction['merchant_risk_30_day'] = 100.0
fraud_transaction['merchant_risk_90_day'] = 100.0
fraud_transaction['customer_avg_amout_1_day'] = 99999.00
fraud_transaction['trans_time_is_night'] = 1.0                

print(f"Sending manipulated FRAUD data for Account: {fraud_transaction.get('acct_num', 'Unknown')}...")
print(f"Attempted Amount: ${fraud_transaction['amt']}")

# Send the heavily manipulated data to the Flask API
response_fraud = requests.post(API_URL, json=fraud_transaction)

# Parse and display the result
if response_fraud.status_code == 200:
    result_fraud = response_fraud.json()
    print("\n--- API RESPONSE FOR FRAUD TRANSACTION ---")
    print(json.dumps(result_fraud, indent=4))

    # EXPLANATION OF EXPECTED RESULT:
    # Because of the massive purchase amount, the models will generate a high risk score.
    # The backend forces a "BLOCKED_SUSPICIOUS" decision, enters the AWS block, and 
    # sns_notification_sent becomes True. An email is dispatched.
else:
    print(f"\n[!] ALERT: API REJECTED THE TRANSACTION [!]")
    print(f"Status code: {response_fraud.status_code}")
    print(f"Server's Error Message: {response_fraud.text}")
