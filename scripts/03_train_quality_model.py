"""
This script trains the image quality assessment model using VLM-derived labels.
It loads the feature matrix, merges with VLM quality labels,
trains a RandomForestClassifier with class balancing, and saves the final model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import argparse
import joblib
import json
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, mean_squared_error, r2_score
from scripts.utils.run_manager import RunManager

def main(feature_matrix_path: Path, vlm_labels_path: Path, output_dir: Path,
         class_balance: bool = True, quality_threshold: float = 0.6,
         force: bool = False):
    """
    Main function to train and save BOTH QA and QC models using VLM labels (FetMRQC approach).

    FetMRQC trains two separate models:
    - QA model (RandomForestRegressor): Predicts continuous quality score 0-1
    - QC model (RandomForestClassifier): Predicts binary reject/accept

    Args:
        feature_matrix_path: Path to feature matrix CSV
        vlm_labels_path: Path to VLM labels CSV
        output_dir: Directory to save trained models
        class_balance: Whether to use class balancing for QC model
        quality_threshold: Quality score threshold for QC (default 0.6, FetMRQC uses 0.25 on their 0-4 scale)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manager = RunManager(output_dir)
    output_files = [
        "derm_qa_model.joblib",
        "derm_qc_model.joblib",
        "model_metadata.json"
    ]

    if not force:
        should_retrain = any(
            run_manager.should_recompute(name, [feature_matrix_path, vlm_labels_path])
            for name in output_files
        )
        if not should_retrain:
            print(f"Cached models detected in {output_dir}. Skipping training. Use --force to retrain.")
            return
    else:
        should_retrain = True

    if should_retrain:
        archived = run_manager.archive_previous_outputs(output_files)
        if archived:
            print(f"Archived {len(archived)} previous model artifact(s) to {run_manager.old_dir}")

    start_time = time.time()

    print("Loading feature matrix...")
    feature_df = pd.read_csv(feature_matrix_path)

    print("Loading VLM labels...")
    vlm_df = pd.read_csv(vlm_labels_path)

    # Merge feature matrix with VLM labels
    print("Merging features with VLM labels...")
    feature_df['image_path_str'] = feature_df['image_path'].astype(str)
    vlm_df['image_path_str'] = vlm_df['image_path'].astype(str)

    merged_df = pd.merge(
        feature_df,
        vlm_df[['image_path_str', 'quality_label', 'quality_score', 'vlm_confidence']],
        on='image_path_str',
        how='inner'
    )

    # Filter to valid labels and prepare BOTH targets (FetMRQC approach)
    train_df = merged_df[merged_df['quality_score'].notna()].copy()

    # QA target: continuous quality score
    train_df['qa_target'] = train_df['quality_score']

    # QC target: binary classification (FetMRQC: quality < threshold -> reject)
    train_df['qc_target'] = (train_df['quality_score'] >= quality_threshold).astype(int)

    print(f"\nFetMRQC Dual-Task Setup:")
    print(f"  QA Task: Predict continuous quality_score (0.0-1.0)")
    print(f"  QC Task: Binary classification with threshold={quality_threshold}")
    print(f"           quality < {quality_threshold} -> REJECT (0)")
    print(f"           quality >= {quality_threshold} -> ACCEPT (1)")

    print(f"\nTotal samples with valid labels: {len(train_df)}")

    # Feature columns: 16 segmentation-free D-IQMs (FetMRQC-style comprehensive feature set)
    # Covers all quality dimensions from teledermatology literature (2022-2025)
    # + BRISQUE features for state-of-art comparison
    features = [
        # Sharpness/Blur (2)
        'sharpness', 'edge_density',
        # Exposure/Lighting (4)
        'under_exposed', 'over_exposed', 'dynamic_range', 'glare',
        # Color (2 - CRITICAL for dermatology, LAB space)
        'color_variance', 'color_cast',
        # Contrast (2 - CRITICAL for dermatology)
        'global_contrast', 'local_contrast',
        # Noise & Information (4, includes BRISQUE)
        'noise', 'entropy', 'brisque_variance', 'brisque_skewness',
        # Framing (2)
        'lesion_size', 'lesion_centrality'
    ]

    # Remove samples with missing features
    train_df = train_df.dropna(subset=features + ['qa_target', 'qc_target'])
    print(f"Samples after removing missing features: {len(train_df)}")

    # Show data distribution
    print(f"\nQA Target (quality_score) Distribution:")
    print(f"  Mean: {train_df['qa_target'].mean():.3f}")
    print(f"  Std: {train_df['qa_target'].std():.3f}")
    print(f"  Range: [{train_df['qa_target'].min():.3f}, {train_df['qa_target'].max():.3f}]")

    qc_counts = train_df['qc_target'].value_counts()
    rejection_rate = qc_counts.get(0, 0) / len(train_df)
    print(f"\nQC Target (binary) Distribution:")
    print(f"  - Accept (1): {qc_counts.get(1, 0)} ({qc_counts.get(1, 0)/len(train_df):.1%})")
    print(f"  - Reject (0): {qc_counts.get(0, 0)} ({rejection_rate:.1%})")

    # Prepare training data with confidence weighting
    X = train_df[features]
    y_qa = train_df['qa_target']
    y_qc = train_df['qc_target']
    sample_weights = train_df['vlm_confidence']

    # Single train/test split for both models (FetMRQC approach)
    X_train, X_test, y_qa_train, y_qa_test, y_qc_train, y_qc_test, w_train, w_test = train_test_split(
        X, y_qa, y_qc, sample_weights, test_size=0.2, random_state=42, stratify=y_qc
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # ========================================================================
    # TRAIN QA MODEL (RandomForestRegressor) - FetMRQC Task 1
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING QA MODEL (Quality Assessment - Regression)")
    print("=" * 70)

    qa_model_params = {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1,
    }
    qa_model = RandomForestRegressor(**qa_model_params)
    qa_model.fit(X_train, y_qa_train, sample_weight=w_train)
    print("[OK] QA model trained (RandomForestRegressor)")

    # ========================================================================
    # TRAIN QC MODEL (RandomForestClassifier) - FetMRQC Task 2
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING QC MODEL (Quality Control - Binary Classification)")
    print("=" * 70)

    qc_model_params = {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced' if class_balance else None
    }
    qc_model = RandomForestClassifier(**qc_model_params)
    qc_model.fit(X_train, y_qc_train, sample_weight=w_train)
    print("[OK] QC model trained (RandomForestClassifier)")

    # ========================================================================
    # EVALUATE QA MODEL (Regression)
    # ========================================================================
    print("\n" + "=" * 70)
    print("QA MODEL EVALUATION (Regression)")
    print("=" * 70)

    y_qa_pred = qa_model.predict(X_test)

    # Regression metrics
    qa_mse = mean_squared_error(y_qa_test, y_qa_pred)
    qa_rmse = np.sqrt(qa_mse)
    qa_r2 = r2_score(y_qa_test, y_qa_pred)
    qa_mae = np.mean(np.abs(y_qa_test - y_qa_pred))

    print(f"\nRegression Metrics:")
    print(f"  R^2 Score: {qa_r2:.4f}")
    print(f"  RMSE: {qa_rmse:.4f}")
    print(f"  MAE: {qa_mae:.4f}")

    # Feature importance for QA
    print(f"\nQA Model Feature Importance:")
    qa_importances = qa_model.feature_importances_
    for feature, importance in sorted(zip(features, qa_importances),
                                     key=lambda x: x[1], reverse=True):
        print(f"  - {feature:15s}: {importance:.3f}")

    # ========================================================================
    # EVALUATE QC MODEL (Classification)
    # ========================================================================
    print("\n" + "=" * 70)
    print("QC MODEL EVALUATION (Binary Classification)")
    print("=" * 70)

    y_qc_pred = qc_model.predict(X_test)

    # Classification metrics
    n_classes = len(qc_model.classes_)
    if n_classes == 2:
        y_qc_proba = qc_model.predict_proba(X_test)[:, 1]
        qc_auc = roc_auc_score(y_qc_test, y_qc_proba)
    else:
        y_qc_proba = qc_model.predict_proba(X_test)[:, 0]
        print(f"\nWARNING: Model only has {n_classes} class(es).")
        qc_auc = None

    print("\nClassification Report:")
    try:
        print(classification_report(y_qc_test, y_qc_pred, target_names=['Reject', 'Accept']))
    except ValueError:
        print(classification_report(y_qc_test, y_qc_pred))

    if qc_auc is not None and np.isfinite(qc_auc):
        print(f"\nROC AUC Score: {qc_auc:.3f}")
    else:
        if qc_auc is not None:
            qc_auc = None
        print("\nROC AUC Score: N/A (single-class test split)")

    # Confusion matrix
    qc_cm = confusion_matrix(y_qc_test, y_qc_pred)
    print(f"\nConfusion Matrix:")
    if qc_cm.shape == (2, 2):
        print(f"                 Predicted")
        print(f"                 Reject  Accept")
        print(f"Actual  Reject   {qc_cm[0,0]:6d}  {qc_cm[0,1]:6d}")
        print(f"        Accept   {qc_cm[1,0]:6d}  {qc_cm[1,1]:6d}")
    else:
        print(f"Warning: Single class in test set, confusion matrix: {qc_cm}")

    # Feature importance for QC
    print(f"\nQC Model Feature Importance:")
    qc_importances = qc_model.feature_importances_
    for feature, importance in sorted(zip(features, qc_importances),
                                     key=lambda x: x[1], reverse=True):
        print(f"  - {feature:15s}: {importance:.3f}")

    # ========================================================================
    # SAVE BOTH MODELS
    # ========================================================================
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save QA model
    qa_model_path = output_dir / "derm_qa_model.joblib"
    joblib.dump(qa_model, qa_model_path)
    print(f"[OK] QA model saved: {qa_model_path}")

    # Save QC model
    qc_model_path = output_dir / "derm_qc_model.joblib"
    joblib.dump(qc_model, qc_model_path)
    print(f"[OK] QC model saved: {qc_model_path}")

    # Save metadata for both models
    metadata = {
        'methodology': 'FetMRQC dual-task approach',
        'quality_threshold': quality_threshold,
        'features': features,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'confidence_weighted': True,
        'training_timestamp': datetime.now().isoformat(),
        'vlm_labels_source': str(vlm_labels_path),
        'feature_matrix_source': str(feature_matrix_path),
        'qa_model': {
            'model_type': 'RandomForestRegressor',
            'model_path': str(qa_model_path),
            'params': qa_model_params,
            'test_r2': float(qa_r2),
            'test_rmse': float(qa_rmse),
            'test_mae': float(qa_mae),
            'feature_importance': {f: float(imp) for f, imp in zip(features, qa_importances)}
        },
        'qc_model': {
            'model_type': 'RandomForestClassifier',
            'model_path': str(qc_model_path),
            'params': qc_model_params,
            'class_balance_used': class_balance,
            'test_roc_auc': float(qc_auc) if qc_auc is not None else None,
            'feature_importance': {f: float(imp) for f, imp in zip(features, qc_importances)}
        }
    }

    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Metadata saved: {metadata_path}")

    elapsed_time = time.time() - start_time

    # Log run
    run_manager.log_run(
        phase="03_train_quality_model",
        status="completed",
        inputs={
            "feature_matrix": str(feature_matrix_path),
            "vlm_labels": str(vlm_labels_path),
            "n_training": len(X_train),
            "n_test": len(X_test)
        },
        outputs={
            "qa_model": str(qa_model_path),
            "qc_model": str(qc_model_path),
            "metadata": str(metadata_path)
        },
        metadata={
            "qa_r2": float(qa_r2),
            "qa_rmse": float(qa_rmse),
            "qc_auc": float(qc_auc) if qc_auc is not None else None,
            "processing_time_seconds": elapsed_time
        }
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - FetMRQC Dual-Task Approach")
    print("=" * 70)
    print(f"\nQA Model (Regression): R^2={qa_r2:.3f}, RMSE={qa_rmse:.3f}")
    qc_auc_display = f"{qc_auc:.3f}" if qc_auc is not None else "N/A"
    print(f"QC Model (Classification): AUC={qc_auc_display}")

    return

    if False:
        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"\nRegression Metrics:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R^2 Score: {r2:.4f}")
        print(f"  MAE: {np.mean(np.abs(y_test - y_pred)):.4f}")

        # Convert to binary for additional metrics using same threshold
        y_pred_binary = (y_pred >= quality_threshold).astype(int)
        y_test_binary = (y_test >= quality_threshold).astype(int)

        print(f"\nBinary Classification (threshold={quality_threshold}):")
        try:
            print(classification_report(y_test_binary, y_pred_binary, target_names=['Reject', 'Accept']))
        except ValueError:
            print(classification_report(y_test_binary, y_pred_binary))

        # Show rejection rate on test set
        test_rejection_rate = (y_pred < quality_threshold).sum() / len(y_pred)
        print(f"\nTest Set Rejection Rate: {test_rejection_rate:.1%} ({int(test_rejection_rate * len(y_pred))}/{len(y_pred)} images)")

        auc_score = r2  # Use R^2 as main metric for regression

    else:
        # Classification metrics
        # Check if model has both classes
        n_classes = len(model.classes_)
        if n_classes == 2:
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.predict_proba(X_test)[:, 0]
            print(f"\nWARNING: Model only has {n_classes} class(es).")

        print("\nClassification Report:")
        try:
            print(classification_report(y_test, y_pred, target_names=['Reject', 'Accept']))
        except ValueError:
            print(classification_report(y_test, y_pred))

        # AUC score
        try:
            auc_score = roc_auc_score(y_test, y_proba)
            print(f"ROC AUC Score: {auc_score:.3f}")
        except ValueError as e:
            print(f"Cannot compute ROC AUC: {e}")
            auc_score = None

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        if cm.shape == (2, 2):
            print(f"                 Predicted")
            print(f"                 Reject  Accept")
            print(f"Actual  Reject   {cm[0,0]:6d}  {cm[0,1]:6d}")
            print(f"        Accept   {cm[1,0]:6d}  {cm[1,1]:6d}")
        else:
            print(f"Warning: Single class in test set, confusion matrix: {cm}")

    # Feature importance
    print(f"\nFeature Importance:")
    importances = model.feature_importances_
    for feature, importance in sorted(zip(features, importances),
                                     key=lambda x: x[1], reverse=True):
        print(f"  - {feature:15s}: {importance:.3f}")

    # Save model and metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "derm_qc_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save training metadata
    metadata = {
        'model_type': 'RandomForestRegressor' if use_soft_targets else 'RandomForestClassifier',
        'use_soft_targets': use_soft_targets,
        'quality_threshold': quality_threshold,
        'threshold_methodology': 'FetMRQC-style: binary classification with quality < threshold -> reject',
        'model_params': model_params,
        'features': features,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'test_metric': float(auc_score) if auc_score is not None else None,
        'metric_name': 'r2_score' if use_soft_targets else 'roc_auc',
        'class_balance_used': class_balance if not use_soft_targets else None,
        'confidence_weighted': True,
        'feature_importance': {f: float(imp) for f, imp in zip(features, importances)},
        'training_timestamp': datetime.now().isoformat(),
        'vlm_labels_source': str(vlm_labels_path),
        'feature_matrix_source': str(feature_matrix_path)
    }

    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {metadata_path}")
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DIQA models using FetMRQC dual-task approach (QA + QC)."
    )
    parser.add_argument(
        "--feature_matrix_path",
        type=Path,
        default=Path("reports/feature_matrix.csv"),
        help="Path to the feature matrix CSV"
    )
    parser.add_argument(
        "--vlm_labels_path",
        type=Path,
        default=Path("reports/vlm_labels.csv"),
        help="Path to VLM labels CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models"),
        help="Directory to save the trained models"
    )
    parser.add_argument(
        "--no_class_balance",
        action="store_true",
        help="Disable class balancing for QC model"
    )
    parser.add_argument(
        "--quality_threshold",
        type=float,
        default=0.6,
        help="Quality score threshold for QC binary classification (default: 0.6)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain models even if cached artifacts exist"
    )
    args = parser.parse_args()
    main(
        args.feature_matrix_path,
        args.vlm_labels_path,
        args.output_dir,
        class_balance=not args.no_class_balance,
        quality_threshold=args.quality_threshold,
        force=args.force
    )
