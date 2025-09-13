from src.data_preprocessing import load_data, clean_data, save_processed
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from utils.helpers import print_banner

if __name__ == "__main__":
    print_banner()

    raw_path = "data/raw/house_prices.csv"
    processed_path = "data/processed/house_prices_clean.csv"
    model_path = "models/linear_regression.pkl"

    # Step 1: Load + Clean Data
    df = load_data(raw_path)
    df_clean = clean_data(df)
    save_processed(df_clean, processed_path)
    print("✅ Data cleaned and saved")

    # Step 2: Train Model
    train_model(processed_path, model_path)

    # Step 3: Evaluate Model
    evaluate_model(processed_path, model_path)