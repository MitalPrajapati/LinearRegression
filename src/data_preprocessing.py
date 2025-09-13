import pandas as pd

path = "/Users/himanshuprajapati/Desktop/Mital/Projects/LinearRegression/data/raw/house_prices.csv"

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.dropna()
    return df

def save_processed(df, path):
    df.to_csv(path, index=False)