import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load model and scaler from separate files
model = pickle.load(open('C:/Users/purna/Desktop/New folder (3)/svc_model.pkl', 'rb'))
scaler = pickle.load(open('C:/Users/purna/Desktop/New folder (3)/scaler.pkl', 'rb'))

#  Input query
query = np.array([2, 98, 50])  # months_since_last_donation, months_since_first_donation, number_of_donations

#  Preprocessing
query_sqrt = np.sqrt(query).reshape(1, -1)
query_scaled = scaler.transform(query_sqrt)

#  Predict with probability
prob = model.predict_proba(query_scaled)[:, 1]
answer = (prob > 0.44).astype(int)

#  Output
if answer == 1:
    print("Donor is likely to donate in March 2007")
else:
    print("Donor is unlikely to donate in March 2007")
