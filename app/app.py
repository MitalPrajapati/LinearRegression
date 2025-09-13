from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "../models/linear_regression.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        # Get input values from form
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        age = int(request.form["age"])

        # Prepare features for model
        features = np.array([[area, bedrooms, age]])
        prediction = model.predict(features)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)