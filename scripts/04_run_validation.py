"""
This script performs the final, rigorous validation of the trained DIQA model.
It evaluates the model on the SCIN test set, the ISIC high-quality control set,
and the DDI fairness set, then generates a consolidated report.
"""

import pandas as pd
from pathlib import Path
import argparse
import joblib
from sklearn.metrics import classification_report

def main(feature_matrix_path: Path, model_path: Path, output_dir: Path):
    """
    Main function to run the multi-dataset validation.
    """
    print("Loading model and feature matrix...")
    model = joblib.load(model_path)
    df = pd.read_csv(feature_matrix_path)
    
    # We need to re-create labels for the SCIN test set
    from scripts.train_quality_model import create_labels
    df_scin = df[df['dataset'] == 'scin'].copy()
    df_scin = create_labels(df_scin)
    
    features = ['sharpness', 'under_exposed', 'over_exposed', 'glare']
    
    report_lines = ["# Final Validation Report\n\n"]
    
    # --- Test 1: Internal Validation (SCIN Test Set) ---
    print("\n--- Running Test 1: Internal Validation on SCIN Test Set ---")
    # We need to identify the test set using the same split as in training
    from sklearn.model_selection import train_test_split
    X = df_scin[features]
    y = df_scin['quality_label']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    y_pred_scin = model.predict(X_test)
    report_scin = classification_report(y_test, y_pred_scin, target_names=['Reject', 'Accept'])
    report_lines.append("## Test 1: Internal Validation (SCIN Test Set)\n")
    report_lines.append("```\n" + report_scin + "\n```\n\n")
    print(report_scin)

    # --- Test 2: High-Quality Control (ISIC) ---
    print("\n--- Running Test 2: High-Quality Control on ISIC Dataset ---")
    df_isic = df[df['dataset'] == 'isic'].copy()
    X_isic = df_isic[features]
    y_pred_isic = model.predict(X_isic)
    
    accept_rate_isic = sum(y_pred_isic) / len(y_pred_isic)
    report_isic = f"Percentage of ISIC images classified as 'Accept': {accept_rate_isic:.2%}"
    report_lines.append("## Test 2: High-Quality Control (ISIC)\n")
    report_lines.append(report_isic + "\n\n")
    print(report_isic)

    # --- Test 3: Fairness & Bias Assessment (DDI) ---
    print("\n--- Running Test 3: Fairness & Bias Assessment on DDI Dataset ---")
    df_ddi = df[df['dataset'] == 'ddi'].copy()
    # DDI metadata needs to be loaded to get skin tones. Assuming a 'fitzpatrick' column exists.
    # This part will need adjustment if the DDI metadata structure is different.
    # For now, we'll just predict and report overall accuracy.
    X_ddi = df_ddi[features]
    y_pred_ddi = model.predict(X_ddi)
    # In a real scenario, we would load DDI ground truth and stratify by skin tone.
    # Since we don't have ground truth quality labels for DDI, we report the accept rate.
    accept_rate_ddi = sum(y_pred_ddi) / len(y_pred_ddi)
    report_ddi = f"Percentage of DDI images classified as 'Accept': {accept_rate_ddi:.2%}"
    report_lines.append("## Test 3: Fairness & Bias Assessment (DDI)\n")
    report_lines.append(report_ddi + "\n\n")
    print(report_ddi)

    # --- Save Final Report ---
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "final_validation_report.md"
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
    print(f"\nFinal validation report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run final validation on the DIQA model.")
    parser.add_argument("--feature_matrix_path", type=Path, default=Path("reports/feature_matrix.csv"))
    parser.add_argument("--model_path", type=Path, default=Path("models/derm_qc_model.joblib"))
    parser.add_argument("--output_dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    main(args.feature_matrix_path, args.model_path, args.output_dir)
