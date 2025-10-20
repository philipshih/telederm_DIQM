"""
This script trains the image quality assessment model.
It loads the feature matrix, creates the ground truth labels,
trains a RandomForestClassifier, and saves the final model.
"""

import pandas as pd
from pathlib import Path
import argparse
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Creates the binary 'quality_label' from confidence scores."""
    
    def get_confidence(row):
        # Logic to handle confidence scores from different dermatologists
        conf_cols = [f'd{i}_confidence' for i in range(1, 4)]
        conf_vals = [row[c] for c in conf_cols if pd.notna(row[c])]
        return sum(conf_vals) / len(conf_vals) if conf_vals else None

    df['avg_confidence'] = df.apply(get_confidence, axis=1)
    
    # Binary classification: Accept (1) vs. Reject (0)
    # We use a threshold of 2.5 on the average confidence
    df['quality_label'] = df['avg_confidence'].apply(lambda x: 1 if x >= 2.5 else 0)
    
    return df

def main(feature_matrix_path: Path, output_dir: Path):
    """
    Main function to train and save the quality model.
    """
    df = pd.read_csv(feature_matrix_path)
    
    # We only train on the SCIN dataset as it has the confidence labels
    train_df = df[df['dataset'] == 'scin'].copy()
    train_df = create_labels(train_df)
    
    # Drop rows where labels could not be created
    train_df.dropna(subset=['quality_label'], inplace=True)
    
    features = ['sharpness', 'under_exposed', 'over_exposed', 'glare']
    target = 'quality_label'
    
    X = train_df[features]
    y = train_df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Evaluating model on the SCIN test set...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "derm_qc_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the DIQA model.")
    parser.add_argument("--feature_matrix_path", type=Path, default=Path("reports/feature_matrix.csv"), help="Path to the feature matrix CSV.")
    parser.add_argument("--output_dir", type=Path, default=Path("models"), help="Directory to save the trained model.")
    args = parser.parse_args()
    main(args.feature_matrix_path, args.output_dir)
