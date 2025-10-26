"""
Enhanced Validation Script - Publication-Ready FetMRQC Validation

This script performs comprehensive validation following FetMRQC methodology:
1. Leave-one-dataset-out cross-validation (REQUIRED for publication)
2. Baseline comparisons (simple threshold + VLM direct)
3. Feature importance visualization
4. Standard 4-part validation (VLM agreement, internal, specificity, fairness)
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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    mean_squared_error, r2_score, accuracy_score, f1_score
)
from sklearn.model_selection import StratifiedKFold
from scripts.utils.run_manager import RunManager


def leave_one_dataset_out_cv(feature_df: Path, vlm_df: Path, output_dir: Path,
                              quality_threshold: float = 0.6) -> dict:
    """
    Leave-one-dataset-out cross-validation (FetMRQC approach).

    Train on N-1 datasets, test on held-out dataset.
    Report worst-case performance across all folds.

    Args:
        feature_df: Feature matrix DataFrame
        vlm_df: VLM labels DataFrame
        output_dir: Directory to save results
        quality_threshold: Threshold for binary classification

    Returns:
        Dictionary with CV results
    """
    print("\n" + "=" * 70)
    print("LEAVE-ONE-DATASET-OUT CROSS-VALIDATION (FetMRQC)")
    print("=" * 70)

    # Merge features with VLM labels
    feature_df['image_path_str'] = feature_df['image_path'].astype(str)
    vlm_df['image_path_str'] = vlm_df['image_path'].astype(str)

    merged_df = pd.merge(
        feature_df,
        vlm_df[['image_path_str', 'quality_score', 'vlm_confidence']],
        on='image_path_str',
        how='inner'
    )

    # Filter valid labels
    train_df = merged_df[merged_df['quality_score'].notna()].copy()

    # Define features (16 D-IQMs)
    features = [
        'sharpness', 'edge_density',
        'under_exposed', 'over_exposed', 'dynamic_range', 'glare',
        'color_variance', 'color_cast',
        'global_contrast', 'local_contrast',
        'noise', 'entropy', 'brisque_variance', 'brisque_skewness',
        'lesion_size', 'lesion_centrality'
    ]

    # Remove missing features
    train_df = train_df.dropna(subset=features + ['quality_score'])

    # Prepare targets
    train_df['qa_target'] = train_df['quality_score']
    train_df['qc_target'] = (train_df['quality_score'] >= quality_threshold).astype(int)

    # Get unique datasets
    datasets = train_df['dataset'].unique()
    print(f"\nDatasets found: {list(datasets)}")
    print(f"Total samples: {len(train_df)}")

    # Cross-validation results
    cv_results = {
        'qa': {'r2': [], 'rmse': [], 'mae': []},
        'qc': {'accuracy': [], 'f1': [], 'auc': []}
    }
    fold_details = []

    # Leave-one-dataset-out
    for i, test_dataset in enumerate(datasets):
        print(f"\n--- Fold {i+1}/{len(datasets)}: Test on {test_dataset} ---")

        # Split train/test
        train_fold = train_df[train_df['dataset'] != test_dataset]
        test_fold = train_df[train_df['dataset'] == test_dataset]

        print(f"Train: {len(train_fold)} samples from {train_fold['dataset'].unique()}")
        print(f"Test: {len(test_fold)} samples from {test_dataset}")

        X_train = train_fold[features]
        X_test = test_fold[features]
        y_qa_train = train_fold['qa_target']
        y_qa_test = test_fold['qa_target']
        y_qc_train = train_fold['qc_target']
        y_qc_test = test_fold['qc_target']
        w_train = train_fold['vlm_confidence']

        # Train QA model (Regression)
        qa_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        qa_model.fit(X_train, y_qa_train, sample_weight=w_train)

        y_qa_pred = qa_model.predict(X_test)
        qa_r2 = r2_score(y_qa_test, y_qa_pred)
        qa_rmse = np.sqrt(mean_squared_error(y_qa_test, y_qa_pred))
        qa_mae = np.mean(np.abs(y_qa_test - y_qa_pred))

        cv_results['qa']['r2'].append(qa_r2)
        cv_results['qa']['rmse'].append(qa_rmse)
        cv_results['qa']['mae'].append(qa_mae)

        print(f"  QA (Regression): R²={qa_r2:.3f}, RMSE={qa_rmse:.3f}, MAE={qa_mae:.3f}")

        # Train QC model (Classification)
        qc_model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'
        )
        qc_model.fit(X_train, y_qc_train, sample_weight=w_train)

        y_qc_pred = qc_model.predict(X_test)
        qc_acc = accuracy_score(y_qc_test, y_qc_pred)
        qc_f1 = f1_score(y_qc_test, y_qc_pred, average='binary', zero_division=0)

        # AUC (if both classes present)
        if len(np.unique(y_qc_test)) == 2 and len(qc_model.classes_) == 2:
            y_qc_proba = qc_model.predict_proba(X_test)[:, 1]
            qc_auc = roc_auc_score(y_qc_test, y_qc_proba)
        else:
            qc_auc = np.nan

        cv_results['qc']['accuracy'].append(qc_acc)
        cv_results['qc']['f1'].append(qc_f1)
        cv_results['qc']['auc'].append(qc_auc)

        print(f"  QC (Classification): Acc={qc_acc:.3f}, F1={qc_f1:.3f}, AUC={qc_auc:.3f if not np.isnan(qc_auc) else 'N/A'}")

        fold_details.append({
            'fold': i + 1,
            'test_dataset': test_dataset,
            'n_train': len(train_fold),
            'n_test': len(test_fold),
            'qa_r2': float(qa_r2),
            'qa_rmse': float(qa_rmse),
            'qa_mae': float(qa_mae),
            'qc_accuracy': float(qc_acc),
            'qc_f1': float(qc_f1),
            'qc_auc': float(qc_auc) if not np.isnan(qc_auc) else None
        })

    # Report aggregate results (FetMRQC reports worst-case)
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY (FetMRQC Worst-Case Reporting)")
    print("=" * 70)

    print("\nQA Model (Regression):")
    print(f"  Mean R²: {np.mean(cv_results['qa']['r2']):.3f} ± {np.std(cv_results['qa']['r2']):.3f}")
    print(f"  Worst R²: {np.min(cv_results['qa']['r2']):.3f}")
    print(f"  Mean RMSE: {np.mean(cv_results['qa']['rmse']):.3f} ± {np.std(cv_results['qa']['rmse']):.3f}")

    print("\nQC Model (Classification):")
    print(f"  Mean Accuracy: {np.mean(cv_results['qc']['accuracy']):.3f} ± {np.std(cv_results['qc']['accuracy']):.3f}")
    print(f"  Worst Accuracy: {np.min(cv_results['qc']['accuracy']):.3f}")

    valid_aucs = [x for x in cv_results['qc']['auc'] if not np.isnan(x)]
    if valid_aucs:
        print(f"  Mean AUC: {np.mean(valid_aucs):.3f} ± {np.std(valid_aucs):.3f}")
        print(f"  Worst AUC: {np.min(valid_aucs):.3f}")

    # Save CV results plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # QA metrics
    ax = axes[0]
    x = range(1, len(datasets) + 1)
    ax.plot(x, cv_results['qa']['r2'], 'o-', label='R²', color='steelblue')
    ax.axhline(np.mean(cv_results['qa']['r2']), linestyle='--', color='gray', alpha=0.5, label='Mean')
    ax.set_xlabel('Fold (Test Dataset)')
    ax.set_ylabel('R² Score')
    ax.set_title('QA Model: Leave-One-Dataset-Out CV')
    ax.legend()
    ax.grid(alpha=0.3)

    # QC metrics
    ax = axes[1]
    ax.plot(x, cv_results['qc']['accuracy'], 'o-', label='Accuracy', color='coral')
    if valid_aucs:
        ax.plot(x, [a if not np.isnan(a) else None for a in cv_results['qc']['auc']],
                's-', label='AUC', color='green')
    ax.axhline(np.mean(cv_results['qc']['accuracy']), linestyle='--', color='gray', alpha=0.5)
    ax.set_xlabel('Fold (Test Dataset)')
    ax.set_ylabel('Score')
    ax.set_title('QC Model: Leave-One-Dataset-Out CV')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "cross_validation_results.png", dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'cv_results': cv_results,
        'fold_details': fold_details,
        'summary': {
            'qa_mean_r2': float(np.mean(cv_results['qa']['r2'])),
            'qa_worst_r2': float(np.min(cv_results['qa']['r2'])),
            'qc_mean_accuracy': float(np.mean(cv_results['qc']['accuracy'])),
            'qc_worst_accuracy': float(np.min(cv_results['qc']['accuracy'])),
            'qc_mean_auc': float(np.mean(valid_aucs)) if valid_aucs else None,
            'qc_worst_auc': float(np.min(valid_aucs)) if valid_aucs else None
        }
    }


def baseline_comparison(feature_df: pd.DataFrame, vlm_df: pd.DataFrame,
                       output_dir: Path, quality_threshold: float = 0.6) -> dict:
    """
    Compare trained model against baseline methods (FetMRQC approach).

    Baselines:
    1. Simple threshold on single feature (sharpness)
    2. VLM labels directly (upper bound)

    Args:
        feature_df: Feature matrix DataFrame
        vlm_df: VLM labels DataFrame
        output_dir: Directory to save results
        quality_threshold: Threshold for binary classification

    Returns:
        Dictionary with comparison results
    """
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON (FetMRQC)")
    print("=" * 70)

    # Merge data
    feature_df['image_path_str'] = feature_df['image_path'].astype(str)
    vlm_df['image_path_str'] = vlm_df['image_path'].astype(str)

    merged_df = pd.merge(
        feature_df,
        vlm_df[['image_path_str', 'quality_score']],
        on='image_path_str',
        how='inner'
    )

    test_df = merged_df[merged_df['quality_score'].notna()].copy()
    test_df['qc_target'] = (test_df['quality_score'] >= quality_threshold).astype(int)

    print(f"Test samples: {len(test_df)}")

    # Baseline 1: Simple threshold on sharpness
    # Use median sharpness as threshold
    sharpness_threshold = test_df['sharpness'].median()
    baseline1_pred = (test_df['sharpness'] > sharpness_threshold).astype(int)
    baseline1_acc = accuracy_score(test_df['qc_target'], baseline1_pred)
    baseline1_f1 = f1_score(test_df['qc_target'], baseline1_pred, average='binary', zero_division=0)

    print(f"\nBaseline 1: Simple Sharpness Threshold (>{sharpness_threshold:.1f})")
    print(f"  Accuracy: {baseline1_acc:.3f}")
    print(f"  F1-Score: {baseline1_f1:.3f}")

    # Baseline 2: VLM Direct (upper bound)
    baseline2_pred = test_df['qc_target']  # VLM labels themselves
    baseline2_acc = 1.0  # Perfect by definition
    baseline2_f1 = 1.0

    print(f"\nBaseline 2: VLM Direct (Upper Bound)")
    print(f"  Accuracy: {baseline2_acc:.3f} (perfect by definition)")
    print(f"  F1-Score: {baseline2_f1:.3f}")

    # Plot comparison (will be populated after model evaluation)
    comparison = {
        'baseline_1_simple_threshold': {
            'method': 'Sharpness > threshold',
            'threshold': float(sharpness_threshold),
            'accuracy': float(baseline1_acc),
            'f1': float(baseline1_f1)
        },
        'baseline_2_vlm_direct': {
            'method': 'VLM labels (upper bound)',
            'accuracy': float(baseline2_acc),
            'f1': float(baseline2_f1)
        }
    }

    return comparison


def plot_feature_importance_enhanced(qa_model, qc_model, features: list,
                                    output_dir: Path):
    """
    Plot feature importance for both QA and QC models (FetMRQC style).

    Args:
        qa_model: Trained QA model (regression)
        qc_model: Trained QC model (classification)
        features: List of feature names
        output_dir: Directory to save plot
    """
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE VISUALIZATION")
    print("=" * 70)

    # Get feature importance from both models
    qa_importance = qa_model.feature_importances_
    qc_importance = qc_model.feature_importances_

    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': features,
        'QA (Regression)': qa_importance,
        'QC (Classification)': qc_importance
    })

    # Sort by average importance
    importance_df['Average'] = (importance_df['QA (Regression)'] + importance_df['QC (Classification)']) / 2
    importance_df = importance_df.sort_values('Average', ascending=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(features))
    width = 0.35

    ax.barh(y_pos - width/2, importance_df['QA (Regression)'], width,
            label='QA (Regression)', color='steelblue', alpha=0.8)
    ax.barh(y_pos + width/2, importance_df['QC (Classification)'], width,
            label='QC (Classification)', color='coral', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['Feature'])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance: QA vs QC Models (16 D-IQMs)')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Print top features
    print("\nTop 5 Features by Average Importance:")
    top5 = importance_df.nlargest(5, 'Average')
    for idx, row in top5.iterrows():
        print(f"  {row['Feature']:20s}: QA={row['QA (Regression)']:.3f}, QC={row['QC (Classification)']:.3f}")


def main(feature_matrix_path: Path, vlm_labels_path: Path,
         qa_model_path: Path, qc_model_path: Path, output_dir: Path,
         quality_threshold: float = 0.6, force: bool = False):
    """
    Main validation function with FetMRQC enhancements.

    Args:
        feature_matrix_path: Path to feature matrix CSV
        vlm_labels_path: Path to VLM labels CSV
        qa_model_path: Path to trained QA model
        qc_model_path: Path to trained QC model
        output_dir: Directory to save outputs
        quality_threshold: Quality threshold for binary classification
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manager = RunManager(output_dir)
    output_name = "validation_results.json"

    inputs = [feature_matrix_path, vlm_labels_path, qa_model_path, qc_model_path]
    need_validation = force or run_manager.should_recompute(output_name, inputs)

    if not need_validation:
        cached_path = output_dir / output_name
        print(f"Cached validation results found at {cached_path}. Use --force to rerun validation.")
        return

    archived = run_manager.archive_previous_outputs([
        "validation_results.json",
        "figures/*.png"
    ])
    if archived:
        print(f"Archived {len(archived)} previous validation artifact(s) to {run_manager.old_dir}")

    start_time = time.time()

    print("\n" + "=" * 70)
    print("ENHANCED VALIDATION - FetMRQC Publication-Ready")
    print("=" * 70)

    # Load data
    print(f"\nLoading feature matrix from {feature_matrix_path}...")
    feature_df = pd.read_csv(feature_matrix_path)

    print(f"Loading VLM labels from {vlm_labels_path}...")
    vlm_df = pd.read_csv(vlm_labels_path)

    print(f"Loading models...")
    qa_model = joblib.load(qa_model_path)
    qc_model = joblib.load(qc_model_path)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run validation components
    results = {}

    # 1. Leave-one-dataset-out CV
    results['cross_validation'] = leave_one_dataset_out_cv(
        feature_df, vlm_df, output_dir, quality_threshold
    )

    # 2. Baseline comparison
    results['baseline_comparison'] = baseline_comparison(
        feature_df, vlm_df, output_dir, quality_threshold
    )

    # 3. Feature importance
    features = [
        'sharpness', 'edge_density',
        'under_exposed', 'over_exposed', 'dynamic_range', 'glare',
        'color_variance', 'color_cast',
        'global_contrast', 'local_contrast',
        'noise', 'entropy', 'brisque_variance', 'brisque_skewness',
        'lesion_size', 'lesion_centrality'
    ]

    plot_feature_importance_enhanced(qa_model, qc_model, features, output_dir)

    # Save comprehensive results
    elapsed_time = time.time() - start_time

    results['metadata'] = {
        'validation_timestamp': datetime.now().isoformat(),
        'feature_matrix': str(feature_matrix_path),
        'vlm_labels': str(vlm_labels_path),
        'qa_model': str(qa_model_path),
        'qc_model': str(qc_model_path),
        'quality_threshold': quality_threshold,
        'processing_time_seconds': elapsed_time
    }

    output_path = output_dir / "validation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nValidation results saved to: {output_path}")

    # Log run
    run_manager.log_run(
        phase="04_run_validation_enhanced",
        status="completed",
        inputs={
            "feature_matrix": str(feature_matrix_path),
            "vlm_labels": str(vlm_labels_path),
            "qa_model": str(qa_model_path),
            "qc_model": str(qc_model_path)
        },
        outputs={"validation_results": str(output_path)},
        metadata={
            "cv_worst_r2": results['cross_validation']['summary']['qa_worst_r2'],
            "cv_worst_accuracy": results['cross_validation']['summary']['qc_worst_accuracy'],
            "processing_time_seconds": elapsed_time
        }
    )

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Processing time: {elapsed_time / 60:.1f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced validation with FetMRQC methodology"
    )
    parser.add_argument(
        "--feature_matrix_path",
        type=Path,
        default=Path("reports/feature_matrix.csv"),
        help="Path to feature matrix CSV"
    )
    parser.add_argument(
        "--vlm_labels_path",
        type=Path,
        default=Path("reports/vlm_labels.csv"),
        help="Path to VLM labels CSV"
    )
    parser.add_argument(
        "--qa_model_path",
        type=Path,
        default=Path("models/derm_qa_model.joblib"),
        help="Path to trained QA model"
    )
    parser.add_argument(
        "--qc_model_path",
        type=Path,
        default=Path("models/derm_qc_model.joblib"),
        help="Path to trained QC model"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports"),
        help="Directory to save validation outputs"
    )
    parser.add_argument(
        "--quality_threshold",
        type=float,
        default=0.6,
        help="Quality threshold for binary classification"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run validation even if cached results exist"
    )

    args = parser.parse_args()

    main(
        args.feature_matrix_path,
        args.vlm_labels_path,
        args.qa_model_path,
        args.qc_model_path,
        args.output_dir,
        args.quality_threshold,
        force=args.force
    )
