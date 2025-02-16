from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        sqft = float(request.form['sqft'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        distance = float(request.form['distance'])
        age = int(request.form['age'])

        # Prepare data for prediction
        input_data = np.array([[sqft, bedrooms, bathrooms, distance, age]])
        predicted_price = model.predict(input_data)[0]

        return render_template('result.html', price=round(predicted_price, 2))
    
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
