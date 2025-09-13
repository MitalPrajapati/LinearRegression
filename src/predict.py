import pickle
import numpy as np

def predict(model_path, features):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    prediction = model.predict([features])
    return prediction[0]

# Example run
if __name__ == "__main__":
    features = [3000, 4, 15]  # area, bedrooms, age
    result = predict("models/linear_regression.pkl", features)
    print(f"Predicted price: {result}")