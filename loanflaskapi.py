from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Initializing Flask Application:
app = Flask(__name__)

# Loading the model and scaler:
model = joblib.load("loanmodel.pkl")
scaler = joblib.load("loanscaler.pkl")

# Creating LabelEncoders for Categorical Variables:
le_education = LabelEncoder()
le_employment = LabelEncoder()
le_marital = LabelEncoder()
le_mortgage = LabelEncoder()
le_dependents = LabelEncoder()
le_purpose = LabelEncoder()
le_cosigner = LabelEncoder()

# Fitting the LabelEncoders with Known Categories:
le_education.fit(['High School', 'Bachelor\'s', 'Master\'s', 'PhD'])
le_employment.fit(['Full-time', 'Part-time', 'Unemployed', 'Self-employed'])
le_marital.fit(['Single', 'Married', 'Divorced'])
le_mortgage.fit(['Yes', 'No'])
le_dependents.fit(['Yes', 'No'])
le_purpose.fit(['Auto', 'Business', 'Education', 'Home', 'Other'])
le_cosigner.fit(['Yes', 'No'])

def return_prediction(model, scaler, sample_json):
    # Extracting data from JSON:
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

    # Encoding Categorical Features:
    data[9] = le_education.transform([data[9]])[0]
    data[10] = le_employment.transform([data[10]])[0]
    data[11] = le_marital.transform([data[11]])[0]
    data[12] = le_mortgage.transform([data[12]])[0]
    data[13] = le_dependents.transform([data[13]])[0]
    data[14] = le_purpose.transform([data[14]])[0]
    data[15] = le_cosigner.transform([data[15]])[0]

    data = np.array(data).reshape(1, -1)
    
    # Scaling Numerical Features:
    scaled_data = scaler.transform(data[:, :9])

    # Combining Scaled Numerical Features and Encoded Categorical Features:
    final_data = np.concatenate([scaled_data, data[:, 9:]], axis=1)

    # Making Predictions:
    prediction = model.predict(final_data)
    probability = model.predict_proba(final_data)[0][1]

    return int(prediction[0]), probability

@app.route("/")
def index():
    return "<h1>Loan Default Prediction App</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    required_fields = [
        "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
        "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio", "Education",
        "EmploymentType", "MaritalStatus", "HasMortgage", "HasDependents",
        "LoanPurpose", "HasCoSigner"
    ]
    
    for field in required_fields:
        if field not in content:
            return jsonify({'error': f'Missing field: {field}'}), 400

    try:
        result, probability = return_prediction(model, scaler, content)
        if result == 1:
            response = {
                'Prediction': 'Will default',
                'Probability': f"The probability of defaulting is: {probability * 100:.2f}%"
            }
        else:
            response = {'Prediction': 'Will not default'}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
