from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model/autoencoder_model.h5')

# Initialize scaler (ensure the same scaler used during training)
scaler = MinMaxScaler()
# Fit the scaler on the same data used during training
# For demonstration, we simulate fitting on new data
# In practice, save and load the scaler using joblib
# scaler = joblib.load('model/scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            amount = float(request.form['amount'])
            hour = int(request.form['hour'])
            device_change = int(request.form['device_change'])
            location_change = int(request.form['location_change'])
            multiple_attempts = int(request.form['multiple_attempts'])

            # Create input array
            input_data = np.array([[amount, hour, device_change, location_change, multiple_attempts]])

            # Normalize input data
            input_scaled = scaler.transform(input_data)

            # Predict reconstruction error
            reconstruction = model.predict(input_scaled)
            mse = np.mean(np.power(input_scaled - reconstruction, 2), axis=1)

            # Define threshold (this should be determined during model evaluation)
            threshold = 0.01  # Example threshold

            # Determine if transaction is fraudulent
            is_fraud = mse > threshold
            prediction = 'Fraudulent Transaction' if is_fraud else 'Legitimate Transaction'
        except Exception as e:
            prediction = f'Error: {str(e)}'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
