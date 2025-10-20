import argparse
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--model-output-path", type=str, required=True)
    args, _ = parser.parse_known_args()

    print("Starting model training...")

    # Load training data
    train_df = pd.read_csv(os.path.join(args.train_data_path, "train.csv"))

    # Separate features and target
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']

    # Train the model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    print("Model training complete.")

    # Save the model
    os.makedirs(args.model_output_path, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_output_path, "model.joblib"))

    # Corrected the final print statement
    print(f"✅ Model artifact saved to {args.model_output_path}")

