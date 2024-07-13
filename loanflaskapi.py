from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Initialize Flask application
app = Flask(__name__)

# Load the model and scaler
model = joblib.load("loanmodel.pkl")
scaler = joblib.load("loanscaler.pkl")

# Create LabelEncoders for categorical variables
le_education = LabelEncoder()
le_employment = LabelEncoder()
le_marital = LabelEncoder()
le_mortgage = LabelEncoder()
le_dependents = LabelEncoder()
le_purpose = LabelEncoder()
le_cosigner = LabelEncoder()

# Fit the LabelEncoders with known categories
le_education.fit(['High School', 'Bachelor\'s', 'Master\'s', 'PhD'])
le_employment.fit(['Full-time', 'Part-time', 'Unemployed', 'Self-employed'])
le_marital.fit(['Single', 'Married', 'Divorced'])
le_mortgage.fit(['Yes', 'No'])
le_dependents.fit(['Yes', 'No'])
le_purpose.fit(['Auto', 'Business', 'Education', 'Home', 'Other'])
le_cosigner.fit(['Yes', 'No'])

def return_prediction(model, scaler, sample_json):
    # Extract data from JSON
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

    # Encode categorical features
    data[9] = le_education.transform([data[9]])[0]
    data[10] = le_employment.transform([data[10]])[0]
    data[11] = le_marital.transform([data[11]])[0]
    data[12] = le_mortgage.transform([data[12]])[0]
    data[13] = le_dependents.transform([data[13]])[0]
    data[14] = le_purpose.transform([data[14]])[0]
    data[15] = le_cosigner.transform([data[15]])[0]

    data = np.array(data).reshape(1, -1)
    
    # Scale numerical features
    scaled_data = scaler.transform(data[:, :9])

    # Combine scaled numerical features and encoded categorical features
    final_data = np.concatenate([scaled_data, data[:, 9:]], axis=1)

    # Make predictions
    prediction = model.predict(final_data)

    return int(prediction[0])

@app.route("/")
def index():
    return "<h1>Loan Default Prediction App</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    try:
        result = return_prediction(model, scaler, content)
        return jsonify({'Default': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':p
    app.run(debug=True)