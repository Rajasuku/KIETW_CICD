# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model from the saved file
model = joblib.load("iris_model.pkl")

# Initialize the Flask application
app = Flask(__name__)

@app.route("/")
def home():
    return "Iris Classifier API is Running!"  # Simple message to confirm server status

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract JSON data from the POST request
        data = request.get_json(force=True)
        
        # Validate input: ensure 'features' key exists and has exactly 4 values
        if "features" not in data or len(data["features"]) != 4:
            return jsonify({"error": "Exactly 4 numerical features are required"}), 400
        
        # Convert features to a NumPy array and reshape to (1, 4) for prediction
        features = np.array(data["features"], dtype=float).reshape(1, -1)
        
        # Make prediction using the loaded model
        prediction = model.predict(features)[0]
        
        # Map numerical prediction to species name
        classes = ["setosa", "versicolor", "virginica"]
        result = {"prediction": classes[prediction]}
        
        # Return prediction as JSON
        return jsonify(result)
    except Exception as e:
        # Handle errors (e.g., invalid data types) and return error message
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Run on all interfaces for Docker compatibility
