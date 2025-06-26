from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and encoder
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    weather = request.form['weather']
    hour = int(request.form['hour'])
    is_holiday = int(request.form['holiday'])

    weather_encoded = encoder.transform([weather])[0]
    features = np.array([[weather_encoded, hour, is_holiday]])
    prediction = model.predict(features)[0]

    return render_template('output.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
