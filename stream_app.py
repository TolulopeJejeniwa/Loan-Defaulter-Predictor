import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('loanmodel.pkl')
scaler = joblib.load('loanscaler.pkl')

# Define the Streamlit app
def main():
    # Set the title of the app
    st.title('Loan Prediction App')

    # Add an image
    st.image('loanimage.jpeg', use_column_width=True)

    # Add a brief description
    st.write('Enter the details below to get loan prediction')

    # Add input fields for user to input data
    age = st.slider('Age', 18, 100, 30)
    income = st.number_input('Income', value=50000)
    loan_amount = st.number_input('Loan Amount', value=100000)
    credit_score = st.slider('Credit Score', 300, 850, 600)
    months_employed = st.slider('Months Employed', 0, 120, 36)
    num_credit_lines = st.slider('Number of Credit Lines', 0, 20, 5)
    interest_rate = st.slider('Interest Rate', 0.0, 30.0, 10.0)
    loan_term = st.slider('Loan Term (years)', 1, 50, 20)
    dti_ratio = st.slider('DTI Ratio', 0.0, 2.0, 0.5)
    education = st.selectbox('Education', ['High School', 'Bachelor\'s', 'Master\'s', 'PhD'])
    employment_type = st.selectbox('Employment Type', ['Full-time', 'Unemployed', 'Self-employed'])
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    has_mortgage = st.selectbox('Has Mortgage', ['Yes', 'No'])
    has_dependents = st.selectbox('Has Dependents', ['Yes', 'No'])
    loan_purpose = st.selectbox('Loan Purpose', ['Auto', 'Business', 'Education', 'Home', 'Other'])
    has_co_signer = st.selectbox('Has Co-Signer', ['Yes', 'No'])

    # Create a button to make predictions
    if st.button('Predict'):
        # Combine input data into a numpy array
        data_array = np.array([[age, income, loan_amount, credit_score, months_employed, num_credit_lines,
                                interest_rate, loan_term, dti_ratio, education, employment_type, marital_status,
                                has_mortgage, has_dependents, loan_purpose, has_co_signer]])

        # Scale numerical features
        scaled_data = scaler.transform(data_array[:, :9])

        # Combine scaled numerical features and encoded categorical features
        final_data = np.concatenate([scaled_data, data_array[:, 9:]], axis=1)

        # Make predictions
        prediction = model.predict(final_data)

        # Display prediction
        st.write(f'Prediction: {"Approved" if prediction[0] == 1 else "Denied"}')

# Run the Streamlit app
if __name__ == '__main__':
    main()

