import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Loading the trained model and scaler
model = joblib.load('loanmodel.pkl')
scaler = joblib.load('loanscaler.pkl')

# Pre-fitting LabelEncoders for categorical variables
le_education = LabelEncoder()
le_employment = LabelEncoder()
le_marital = LabelEncoder()
le_mortgage = LabelEncoder()
le_dependents = LabelEncoder()
le_purpose = LabelEncoder()
le_cosigner = LabelEncoder()

# Fitting the LabelEncoders with the known categories
le_education.fit(['High School', 'Bachelor\'s', 'Master\'s', 'PhD'])
le_employment.fit(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'])
le_marital.fit(['Single', 'Married', 'Divorced'])
le_mortgage.fit(['Yes', 'No'])
le_dependents.fit(['Yes', 'No'])
le_purpose.fit(['Auto', 'Business', 'Education', 'Home', 'Other'])
le_cosigner.fit(['Yes', 'No'])

# Defining the Streamlit app
def main():
    # Setting the title of the app
    st.title('Loan Defaulter Prediction App')

    # Adding an image
    st.image('loanimage.jpeg', use_column_width=True)

    # Adding a brief description
    st.write('Enter the applicant details below, to get if an applicant will default in paying back obtained loan')

    # Generating two columns
    col1, col2 = st.columns(2)

    # Adding input fields for user to input data in the columns
    with col1:
        age = st.number_input('Age', 18, 100, 30)
        income = st.number_input('Income', value=50000)
        loan_amount = st.number_input('Loan Amount', value=100000)
        credit_score = st.number_input('Credit Score', 300, 950, 600)
        months_employed = st.number_input('Months Employed', 0, 150, 36)
        num_credit_lines = st.number_input('Number of Credit Lines', 0, 20, 5)
        interest_rate = st.number_input('Interest Rate', 0.0, 30.0, 10.0)
        loan_term = st.number_input('Loan Term (months)', 1, 80, 20)
    
    with col2:
        dti_ratio = st.slider('DTI Ratio', 0.0, 1.0, 0.5)
        education = st.selectbox('Education', ['High School', 'Bachelor\'s', 'Master\'s', 'PhD'])
        employment_type = st.selectbox('Employment Type', ['Full-time', 'Part-time', 'Self-employed', 'Unemployed'])
        marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
        has_mortgage = st.selectbox('Has Mortgage', ['Yes', 'No'])
        has_dependents = st.selectbox('Has Dependents', ['Yes', 'No'])
        loan_purpose = st.selectbox('Loan Purpose', ['Auto', 'Business', 'Education', 'Home', 'Other'])
        has_co_signer = st.selectbox('Has Co-Signer', ['Yes', 'No'])

    # Encoding categorical features using the pre-fitted LabelEncoders
    education = le_education.transform([education])[0]
    employment_type = le_employment.transform([employment_type])[0]
    marital_status = le_marital.transform([marital_status])[0]
    has_mortgage = le_mortgage.transform([has_mortgage])[0]
    has_dependents = le_dependents.transform([has_dependents])[0]
    loan_purpose = le_purpose.transform([loan_purpose])[0]
    has_co_signer = le_cosigner.transform([has_co_signer])[0]

    # Creating a button to make predictions
    if st.button('Predict'):
        # Combining input data into a numpy array
        data_array = np.array([[age, income, loan_amount, credit_score, months_employed, num_credit_lines,
                                interest_rate, loan_term, dti_ratio, education, employment_type, marital_status,
                                has_mortgage, has_dependents, loan_purpose, has_co_signer]])

        # Scaling numerical features
        scaled_data = scaler.transform(data_array[:, :9])

        # Combining scaled numerical features and encoded categorical features
        final_data = np.concatenate([scaled_data, data_array[:, 9:]], axis=1)

        # Making predictions
        prediction = model.predict(final_data)

        # Displaying prediction
        st.write(f'Prediction: {"Will Default" if prediction[0] == 1 else "Will Not Default"}')
        st.write(f'Probability of Default: {probability[0][1]:.2f}')
        
# Running the Streamlit app
if __name__ == '__main__':
    main()
