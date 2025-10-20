import argparse
import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-input-path", type=str, required=True)
    parser.add_argument("--test-data-path", type=str, required=True)
    parser.add_argument("--output-evaluation-path", type=str, required=True)
    args, _ = parser.parse_known_args()

    print("Starting model evaluation...")

    # Load the model
    model = joblib.load(os.path.join(args.model_input_path, "model.joblib"))

    # Load the test data
    test_df = pd.read_csv(os.path.join(args.test_data_path, "test.csv"))

    # Separate features and target
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    print(f"Test Accuracy: {accuracy:.4f}")

    # Create evaluation report
    evaluation_data = {
        "metrics": {
            "accuracy": {
                "value": accuracy,
                "standard_deviation": "N/A"
            },
            "classification_report": report_dict
        }
    }

    # Save the evaluation report
    os.makedirs(args.output_evaluation_path, exist_ok=True)
    with open(os.path.join(args.output_evaluation_path, "evaluation.json"), "w") as f:
        json.dump(evaluation_data, f)
    
    print(f"✅ Evaluation report saved to {args.output_evaluation_path}")