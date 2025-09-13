import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(data_path, model_path):
    df = pd.read_csv(data_path)
    X = df[['area', 'bedrooms', 'age']]
    y = df['price']

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    print(f"📊 MSE: {mse:.2f}")
    print(f"📊 R2 Score: {r2:.2f}")