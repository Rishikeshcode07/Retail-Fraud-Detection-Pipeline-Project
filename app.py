# ==============================================================================
# WEEK 4 - CELL 1: FLASK API WITH ML MODELS & AWS SNS INTEGRATION
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. IMPORTS & SETUP
# ------------------------------------------------------------------------------
# WHY WE NEED THIS: We are building a bridge between our machine learning models and the internet.
# Flask   : A web framework that turns this Python script into a live server listening for data.
# request : Captures the incoming transaction data (the JSON payload) sent to our API.
# jsonify : Converts our Python response back into web-friendly JSON format.
# joblib  : Loads our pre-trained machine learning models quickly from disk.
# pandas  : Structures the incoming web data into a tabular DataFrame so the models can read it.
# boto3   : The official AWS SDK. This is our direct pipeline to AWS SNS for sending emails.
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import boto3

# Initialize the backend application
app = Flask(__name__)

# ------------------------------------------------------------------------------
# 2. AWS SNS CONFIGURATION (THE ALERT SYSTEM)
# ------------------------------------------------------------------------------
# WHAT IS THIS: We are configuring our connection to the AWS Cloud.
# IMPACT: Without this, the system can detect fraud, but it cannot notify human investigators.
SNS_TOPIC_ARN = "arn:aws:sns:eu-north-1:387701295294:FraudAlertTopic"
AWS_REGION = "eu-north-1"

# Active AWS Credentials allowing our app to publish to the SNS Topic
ACCESS_KEY = "*******************************"
SECRET_KEY = "*******************************"

try:
    # We initialize the AWS client. If this connects, our app is capable of sending emails.
    sns_client = boto3.client(
        'sns',
        region_name=AWS_REGION,
        aws_access_key_id=ACCESS_KEY.strip(),
        aws_secret_access_key=SECRET_KEY.strip()
    )
    aws_ready = True
    print("SUCCESS: AWS SNS Client connected. Real-time alerting is ACTIVE.")
except Exception as e:
    aws_ready = False
    print(f"WARNING: AWS SNS Setup failed. System will detect fraud but cannot send emails. Error: {e}")

# ------------------------------------------------------------------------------
# 3. LOAD MACHINE LEARNING MODELS
# ------------------------------------------------------------------------------
# WHY DO WE LOAD HERE?: We load the models outside of the prediction function. 
# IMPACT: If we loaded them inside the function, the server would have to reload heavy files 
# for every single transaction, slowing the system to a crawl. Doing it once here keeps it lightning fast.
print("Loading Random Forest and Isolation Forest models...")
rf_model = joblib.load('random_forest_fraud_model.pkl')
iso_model = joblib.load('isolation_forest_fraud_model.pkl')
print("SUCCESS: Models loaded into memory.")

# ------------------------------------------------------------------------------
# 4. THE API ENDPOINT (THE DECISION ENGINE)
# ------------------------------------------------------------------------------
# WHAT IS THIS: This is the URL route (/predict) that our POS system or test script will send data to.
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # STEP A: DATA INGESTION
        # We receive the transaction data from the internet and convert it into a Pandas DataFrame.
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # STEP B: MODEL EVALUATION
        # We pass the transaction through our two distinct AI models to get three different metrics:

        # 1. Random Forest (Supervised): Looks at historical fraud patterns. Returns 0 (Safe) or 1 (Fraud).
        rf_prediction = int(rf_model.predict(input_df)[0])

        # 2. Random Forest Probability: Returns the exact statistical confidence of fraud (e.g., 22% vs 85%).
        rf_proba = float(rf_model.predict_proba(input_df)[0][1])

        # 3. Isolation Forest (Unsupervised): Looks for bizarre outliers. Returns 1 (Normal) or -1 (Anomaly).
        iso_prediction = int(iso_model.predict(input_df)[0]) 

        # Format the outputs for our logic engine
        is_anomaly = (iso_prediction == -1)
        risk_score = round(rf_proba * 100, 2)

        # STEP C: THE BUSINESS LOGIC (NORMAL VS FRAUD)
        # WHAT HAPPENS HERE: This is the core logic that decides if an email gets sent.
        # If ANY of these three red flags trigger, the transaction is marked as fraud.
        if rf_prediction == 1 or is_anomaly or risk_score > 50.0:
            decision = "BLOCKED_SUSPICIOUS"  # This will trigger the email
        else:
            decision = "APPROVED"            # This is a normal transaction; NO email will be sent

        sns_sent = False
        message_id = None

        # STEP D: AWS CLOUD ALERTING
        # WHY THIS WORKS: This block ONLY executes if the decision is "BLOCKED_SUSPICIOUS".
        # If the transaction is normal ("APPROVED"), this entire block is skipped, which is why 
        # you don't get spammed with emails for normal transactions.
        if decision == "BLOCKED_SUSPICIOUS" and aws_ready:

            # Format a clean, professional email alert payload with the exact transaction details
            alert_payload = (
                f"HIGH RISK FRAUD ALERT DETECTED\n\n"
                f"Target Notification Email: rishicode07@gmail.com\n"
                f"Transaction Amount: ${data.get('amt', 'N/A')}\n"
                f"Account Number: {data.get('acct_num', 'N/A')}\n"
                f"Calculated Risk Score: {risk_score}%\n"
                f"Anomaly Flagged: {is_anomaly}\n"
                f"System Decision: {decision}\n\n"
                f"Action Required: Immediate review needed."
            )

            # Publish sends the message to AWS, which immediately triggers the email to your inbox.
            sns_response = sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=alert_payload,
                Subject="Fraud Alert: Suspicious Transaction Flagged"
            )
            message_id = sns_response.get('MessageId')
            sns_sent = True

        # STEP E: RETURN THE RESULT
        # Send a JSON report back to the system that requested the prediction, detailing exactly what happened.
        return jsonify({
            "status": "success",
            "decision": decision,
            "evaluation_metrics": {
                "supervised_risk_score_percentage": risk_score,
                "unsupervised_anomaly_detected": is_anomaly
            },
            "cloud_alert_system": {
                "sns_notification_sent": sns_sent,
                "aws_message_id": message_id
            }
        }), 200

    except Exception as err:
        return jsonify({"status": "error", "message": str(err)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
