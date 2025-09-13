import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

def train_model(data_path, model_path):
    df = pd.read_csv(data_path)

    X = df[['area', 'bedrooms', 'age']]
    y = df['price']

    model = LinearRegression()
    model.fit(X, y)

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved at", model_path)