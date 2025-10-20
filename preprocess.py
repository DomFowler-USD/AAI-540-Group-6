import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

def clean_col_names(df):
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = col.replace(' ', '_')
        new_col = re.sub(r'[^A-Za-z0-9_]+', '', new_col)
        new_col = new_col.strip('_').lower()
        new_cols.append(new_col)
    df.columns = new_cols
    return df

if __name__ == "__main__":
    
    print("Starting preprocessing...")

    # Define standard SageMaker paths
    base_dir = "/opt/ml/processing"
    input_data_path = os.path.join(base_dir, "input")
    output_train_path = os.path.join(base_dir, "train")
    output_validation_path = os.path.join(base_dir, "validation")
    output_test_path = os.path.join(base_dir, "test")

    # Load data
    df = pd.read_csv(os.path.join(input_data_path, "DevOps AWS Azure Effectiveness Deployment.csv"))
    
    # Clean column names
    df = clean_col_names(df)

    # Feature Engineering
    df['target'] = (df['platform_comparison_index'] > 50).astype(int)
    df_cleaned = df.drop(columns=['platform_comparison_index', 'organization_name'])

    # Split data
    df_modeling, _ = train_test_split(
        df_cleaned, test_size=0.40, random_state=42, stratify=df_cleaned['target']
    )
    df_train, df_temp = train_test_split(
        df_modeling, train_size=(0.4/0.6), random_state=42, stratify=df_modeling['target']
    )
    df_validation, df_test = train_test_split(
        df_temp, test_size=0.50, random_state=42, stratify=df_temp['target']
    )

    print(f"Training set shape: {df_train.shape}")
    print(f"Validation set shape: {df_validation.shape}")
    print(f"Test set shape: {df_test.shape}")

    # Save processed data
    os.makedirs(output_train_path, exist_ok=True)
    os.makedirs(output_validation_path, exist_ok=True)
    os.makedirs(output_test_path, exist_ok=True)

    df_train.to_csv