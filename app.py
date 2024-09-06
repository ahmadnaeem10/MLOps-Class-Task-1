from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        data = request.form['data']
        features = [float(i) for i in data.split(',')]
        prediction = model.predict(np.array(features).reshape(1, -1))[0]
    return render_template('index.html', prediction=prediction)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    features = [float(i) for i in data]
    prediction = model.predict(np.array(features).reshape(1, -1))[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
