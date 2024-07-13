from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd


def return_prediction(model, sample_json):
    # Extract data from json
    data = [
        sample_json["Age"],
        sample_json["Income"],
        sample_json["LoanAmount"],
        sample_json["CreditScore"],
        sample_json["MonthsEmployed"],
        sample_json["NumCreditLines"],
        sample_json["InterestRate"],
        sample_json["LoanTerm"],
        sample_json["DTIRatio"],
        sample_json["Education"],
        sample_json["EmploymentType"],
        sample_json["MaritalStatus"],
        sample_json["HasMortgage"],
        sample_json["HasDependents"],
        sample_json["LoanPurpose"],
        sample_json["HasCoSigner"]
    ]

    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)

    return int(prediction[0])

# Initialize Flask application
app = Flask(__name__)

@app.route("/")
def index():
    return "<h1>Loan Default Prediction App</h1>"

# Load the model
model = joblib.load("loan_default_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    result = return_prediction(model, content)
    return jsonify({'Default': result})

if __name__ == '__main__':
    app.run(debug=True)
