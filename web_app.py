import numpy as np
import pickle
import streamlit as st

# Load model and scaler
model = pickle.load(open('C:/Users/purna/Desktop/New folder (3)/svc_model.pkl', 'rb'))
scaler = pickle.load(open('C:/Users/purna/Desktop/New folder (3)/scaler.pkl', 'rb'))

# Prediction function
def donor_prediction(input_data):
    query = np.array(input_data)  # input_data is already a list
    query_sqrt = np.sqrt(query).reshape(1, -1)
    query_scaled = scaler.transform(query_sqrt)
    prob = model.predict_proba(query_scaled)[:, 1]
    answer = (prob > 0.44).astype(int)
    if answer == 1:
        return "Donor is likely to donate in March 2007"
    else:
        return "Donor is unlikely to donate in March 2007"

# Streamlit UI
def main():
    st.title('Blood Donor Prediction Interface')

    # Inputs
    months_since_last_donation = st.number_input('Months since last donation', min_value=0)
    months_since_first_donation = st.number_input('Months since first donation', min_value=0)
    number_of_donations = st.number_input('Total number of donations', min_value=0)

    prediction = ''

    # Button
    if st.button('Predict'):
        prediction = donor_prediction([
            months_since_last_donation,
            months_since_first_donation,
            number_of_donations
        ])
        st.success(prediction)

# Main entry point
if __name__ == "__main__":
    main()
