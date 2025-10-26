# Teledermatology DIQA Project - Complete Workflow & Data Structure

## Project Overview

This project implements an **interpretable Dermatologic Image Quality Assessment (DIQA) framework** using VLM knowledge distillation, inspired by FetMRQC methodology from neuroimaging. The goal is to create a lightweight, segmentation-free quality control system for teledermatology images suitable for publication in top-tier journals.

## Core Methodology

### 1. Knowledge Distillation Approach
- **Teacher**: Vision Language Model (VLM) provides quality assessments across 8 dimensions
- **Student**: Interpretable RandomForest models trained on 16 segmentation-free D-IQMs
- **Validation**: 4-part strategy (VLM agreement, internal accuracy, specificity, fairness)

### 2. Design Constraints
- **Segmentation-Free**: All metrics computed from raw pixels (no deep learning dependencies)
- **Interpretable**: RandomForest models with explicit feature importance
- **Image-Level**: Every image treated as independent data point (not case-level)

## Project Structure

```
telederm_neuroimage/
├── data/                           # Raw datasets (gitignored)
│   ├── scin/                       # SCIN teledermatology dataset (training)
│   │   └── dataset/
│   │       ├── scin_cases.csv      # Case metadata
│   │       ├── scin_labels.csv     # Diagnostic labels
│   │       └── images/             # SCIN image files
│   ├── isic/                       # ISIC dermoscopy dataset (specificity test)
│   │   └── images/
│   └── ddi/                        # DDI dataset (fairness test)
│       └── ddi_v1/
│           ├── ddi_metadata.csv    # Includes Fitzpatrick skin type
│           └── images/
│
├── src/diqa/                       # Core DIQA library
│   ├── __init__.py
│   ├── metrics.py                  # 16 D-IQM feature extractors
│   ├── vlm_client.py              # VLM API interface (OpenAI/Anthropic/Google)
│   └── verification.py            # Quality verification utilities
│
├── scripts/                        # Pipeline execution scripts
│   ├── 01_load_derm_data.py       # Phase 1: Create master_image_list.csv
│   ├── 02_vlm_annotation.py       # Phase 2: Generate VLM labels
│   ├── 02_compute_d_iqms.py       # Phase 3: Extract 16 D-IQM features
│   ├── 03_train_quality_model.py  # Phase 4: Train dual QA/QC models
│   ├── 04_run_validation.py       # Phase 5: 4-part validation
│   └── utils/                     # Helper modules
│       ├── fairness_metrics.py    # Fitzpatrick-stratified metrics
│       └── visualization.py       # Plotting and reports
│
├── models/                         # Trained models (saved artifacts)
│   ├── derm_qa_model.joblib       # Quality Assessment (regression)
│   ├── derm_qc_model.joblib       # Quality Control (classification)
│   └── model_metadata.json        # Training hyperparameters & metrics
│
├── reports/                        # Intermediate data outputs
│   ├── master_image_list.csv      # Phase 1: All images with metadata
│   ├── vlm_labels.csv             # Phase 2: VLM quality annotations
│   ├── feature_matrix.csv         # Phase 3: 16 D-IQMs per image
│   └── validation_results.json    # Phase 5: Final performance metrics
│
├── tests/                          # Comprehensive test suite
│   ├── test_data_integrity.py     # Data pipeline validation
│   ├── test_metrics.py            # D-IQM computation tests
│   ├── test_vlm_annotation.py     # VLM output validation
│   └── test_model_training.py     # Model training pipeline tests
│
└── docs/                           # Project documentation
    ├── 00_PROJECT_BRIEF.md         # High-level objectives
    ├── METHODOLOGY.md              # Core methodological decisions
    ├── STRATEGY.md                 # Execution strategy
    └── CONTINGENCY_PLAN.md         # Backup plans for failures
```

## Cache-Aware Pipeline Execution

Each pipeline phase now uses lightweight caching so repeated runs only recompute when inputs change:

- `01_load_derm_data.py` skips regeneration when `reports/master_image_list.csv` already exists; use `--force` to rebuild or `--limit 500` to regenerate the capped LLM subset.
- `02_vlm_annotation.py` resumes from checkpoints and reuses completed labels unless invoked with `--force`. The default master list intentionally tops out at 500 images to control VLM spend.
- `02_compute_d_iqms.py`, `03_train_quality_model.py`, and `04_run_validation_enhanced.py` check source timestamps before running, while still supporting full recomputes via `--force`.

This keeps the end-to-end pipeline fast on incremental edits and guarantees that any LLM-facing workload stays within the agreed 500-image ceiling unless you explicitly expand the dataset.

## Data Workflow - Phase by Phase

### Phase 0: Setup
**Goal**: Initialize project structure and verify data access

**Input**: Raw datasets in `data/` directory
**Output**: Project scaffolding complete

**Commands**:
```bash
# Verify directory structure
python -c "from pathlib import Path; print([p.name for p in Path('data').iterdir()])"

# Check API keys in .env
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('APIs:', [k for k in os.environ if 'API_KEY' in k])"
```

---

### Phase 1: Data Ingestion
**Goal**: Create unified master image list across all datasets

**Script**: `scripts/01_load_derm_data.py`

**Input**:
- `data/scin/dataset/scin_cases.csv` (SCIN metadata)
- `data/isic/` (ISIC image directory)
- `data/ddi/ddi_v1/ddi_metadata.csv` (DDI metadata)

**Output**: `reports/master_image_list.csv`

**Schema**:
| Column | Type | Description |
|--------|------|-------------|
| image_path | str | Absolute path to image file |
| dataset | str | Dataset name (scin, isic, ddi) |
| case_id | str | Original case identifier |
| fitzpatrick_type | int | Skin tone (1-6, DDI only) |
| split | str | train/val/test designation |

**Validation**:
- All image paths exist and are readable
- No duplicate image paths
- All required columns present
- Fitzpatrick type valid (1-6) for DDI images

**Commands**:
```bash
python scripts/01_load_derm_data.py --output_dir reports
```

---

### Phase 2: VLM Annotation
**Goal**: Generate quality labels using Vision Language Model

**Script**: `scripts/02_vlm_annotation.py`

**Input**: `reports/master_image_list.csv`

**Output**: `reports/vlm_labels.csv`

**Schema**:
| Column | Type | Description |
|--------|------|-------------|
| image_path | str | Absolute path to image |
| decision | str | ACCEPT / ACCEPT_WITH_ISSUES / REJECT |
| quality_score | float | Overall quality (0.0-1.0) |
| sharpness | float | Sharpness criterion (0.0-1.0) |
| exposure | float | Exposure criterion (0.0-1.0) |
| glare | float | Glare criterion (0.0-1.0) |
| artifacts_obstruction | float | Artifacts criterion (0.0-1.0) |
| framing_primary_lesion_visible | float | Framing criterion (0.0-1.0) |
| lighting_uniformity | float | Lighting criterion (0.0-1.0) |
| color_balance | float | Color criterion (0.0-1.0) |
| noise_compression | float | Noise criterion (0.0-1.0) |
| critical_failures | str | Pipe-separated list of failures |
| modality | str | standard / polarized / dermoscopy |
| skin_tone_bin | str | VLM-estimated skin tone |
| vlm_confidence | float | VLM self-reported confidence |
| vlm_reasoning | str | VLM explanation text |

**VLM Providers**:
- **Google Gemini 2.0 Flash** (Cheapest: $0.001/image, recommended for 500-image test)
- OpenAI GPT-4o ($0.01/image)
- Anthropic Claude 3.5 Sonnet ($0.015/image)

**Validation**:
- All image_paths from master_list processed
- quality_score in range [0.0, 1.0]
- All 8 dimension scores in range [0.0, 1.0]
- decision is valid enum value
- vlm_confidence > 0.5 for high-quality labels

**Commands**:
```bash
# Full dataset with Google Gemini (cheapest)
python scripts/02_vlm_annotation.py --provider google --model gemini-2.0-flash

# 500-image test with cost tracking
python scripts/02_vlm_annotation.py --provider google --model gemini-2.0-flash \
    --master_list_path reports/master_image_list_test500.csv \
    --batch_size 50 --rate_limit_delay 0.5

# Mock mode (no API costs, for testing)
python scripts/02_vlm_annotation.py --provider mock
```

---

### Phase 3: D-IQM Feature Engineering
**Goal**: Extract 16 segmentation-free image quality metrics

**Script**: `scripts/02_compute_d_iqms.py`

**Input**: `reports/master_image_list.csv`

**Output**: `reports/feature_matrix.csv`

**Schema** (16 D-IQM features):
| Feature | Description | Range |
|---------|-------------|-------|
| sharpness | Laplacian variance | 0-∞ |
| edge_density | Sobel edge ratio | 0-1 |
| under_exposed | % dark pixels (<10) | 0-1 |
| over_exposed | % bright pixels (>246) | 0-1 |
| dynamic_range | Percentile utilization | 0-1 |
| glare | Adaptive HSV specular highlights | 0-1 |
| color_variance | LAB chromatic variance | 0-∞ |
| color_cast | LAB deviation from neutral | 0-∞ |
| global_contrast | RMS contrast | 0-∞ |
| local_contrast | Michelson in patches | 0-1 |
| noise | Smooth region std | 0-∞ |
| entropy | Shannon entropy | 0-8 |
| brisque_variance | MSCN variance | 0-∞ |
| brisque_skewness | MSCN skewness | -∞-∞ |
| lesion_size | Saliency ratio | 0-1 |
| lesion_centrality | Centroid distance | 0-1 |

**Implementation**: All functions in `src/diqa/metrics.py`

**Validation**:
- All image_paths from master_list processed
- No NaN or infinite values in features
- Feature ranges within expected bounds
- Computation time < 1 second per image

**Commands**:
```bash
python scripts/02_compute_d_iqms.py --input_csv reports/master_image_list.csv \
    --output_csv reports/feature_matrix.csv --parallel
```

---

### Phase 4: Model Training (FetMRQC Dual-Task)
**Goal**: Train two interpretable RandomForest models

**Script**: `scripts/03_train_quality_model.py`

**Input**:
- `reports/feature_matrix.csv` (16 D-IQMs)
- `reports/vlm_labels.csv` (VLM ground truth)

**Output**:
- `models/derm_qa_model.joblib` (Quality Assessment - Regression)
- `models/derm_qc_model.joblib` (Quality Control - Classification)
- `models/model_metadata.json` (Training metadata)

**Model Architecture**:
```
QA Model (RandomForestRegressor):
  - n_estimators: 100
  - Task: Predict continuous quality_score (0.0-1.0)
  - Loss: MSE
  - Metrics: R², RMSE, MAE

QC Model (RandomForestClassifier):
  - n_estimators: 100
  - Task: Binary classification (quality < 0.6 → Reject)
  - Class weight: Balanced
  - Metrics: ROC-AUC, Precision, Recall, F1
```

**Validation**:
- Training/test split: 80/20 stratified by QC label
- Sample weighting by VLM confidence
- Feature importance analysis
- Test set performance meets thresholds (R² > 0.5, AUC > 0.85)

**Commands**:
```bash
python scripts/03_train_quality_model.py \
    --feature_matrix_path reports/feature_matrix.csv \
    --vlm_labels_path reports/vlm_labels.csv \
    --output_dir models \
    --quality_threshold 0.6
```

---

### Phase 5: Validation (4-Part Strategy)
**Goal**: Comprehensive validation for publication-grade results

**Script**: `scripts/04_run_validation.py`

**Input**:
- `models/derm_qa_model.joblib`
- `models/derm_qc_model.joblib`
- `reports/feature_matrix.csv`
- `reports/vlm_labels.csv`

**Output**: `reports/validation_results.json`

#### Validation Part 1: VLM Agreement Analysis
**Goal**: Quantify knowledge distillation fidelity

**Metrics**:
- Pearson correlation (predicted vs VLM quality_score)
- Per-dimension MSE for 8 quality criteria
- Target: R > 0.7, demonstrating successful distillation

#### Validation Part 2: Internal Accuracy (SCIN)
**Goal**: Performance on real teledermatology images

**Dataset**: SCIN held-out test set (20% of training data)

**Metrics**:
- ROC-AUC, Precision, Recall, F1 for QC model
- R², RMSE for QA model
- Target: AUC > 0.85, Precision > 0.80

#### Validation Part 3: Specificity Test (ISIC)
**Goal**: Ensure no false rejection of high-quality dermoscopy

**Dataset**: ISIC dermoscopy images (different modality)

**Metrics**:
- Acceptance rate (should be >95%)
- Mean predicted quality_score
- Target: >95% acceptance, demonstrating modality-appropriate assessment

#### Validation Part 4: Fairness Assessment (DDI)
**Goal**: Ensure equitable performance across skin tones

**Dataset**: DDI dataset with Fitzpatrick labels

**Metrics**:
- Stratified performance by Fitzpatrick type (I-VI)
- Performance gap analysis
- Target: <10% performance gap between lightest/darkest tones

**Commands**:
```bash
python scripts/04_run_validation.py \
    --model_dir models \
    --reports_dir reports \
    --output_path reports/validation_results.json
```

---

## Data Integrity Checks

### Critical Validation Points

1. **Image File Accessibility**
   - All paths in master_image_list.csv are valid files
   - Images are readable by OpenCV/PIL
   - No corrupt or zero-byte files

2. **VLM Label Quality**
   - All required columns present
   - No missing quality_score values
   - All dimension scores in [0.0, 1.0]
   - decision field is valid enum

3. **Feature Matrix Quality**
   - One row per image_path
   - All 16 D-IQM features present
   - No NaN or infinite values
   - Feature distributions reasonable (detect outliers)

4. **Data Join Integrity**
   - master_image_list ↔ vlm_labels: 100% match on image_path
   - master_image_list ↔ feature_matrix: 100% match on image_path
   - vlm_labels ↔ feature_matrix: 100% match on image_path

5. **Train/Test Split Integrity**
   - No data leakage between splits
   - Stratification by QC label maintained
   - Consistent split across all phases

## Expected Outputs for Publication

### Manuscript Figures

1. **Feature Importance Plot**
   - Top 16 D-IQMs ranked by importance for QA/QC models
   - Comparison between models

2. **Performance Curves**
   - ROC curves for QC model on all datasets
   - Quality score distribution plots

3. **Fairness Analysis**
   - Performance metrics stratified by Fitzpatrick type
   - Bar charts showing <10% performance gap

4. **VLM Distillation Quality**
   - Scatter plot: VLM quality_score vs model prediction
   - Pearson R and regression line

### Manuscript Tables

1. **Dataset Statistics**
   - Image counts per dataset
   - Quality label distribution
   - Demographic breakdown (DDI)

2. **Model Performance**
   - Internal validation metrics (SCIN test set)
   - Specificity metrics (ISIC)
   - Fairness metrics (DDI stratified)

3. **Feature Engineering**
   - 16 D-IQM definitions and ranges
   - Computational efficiency (time per image)

### Supplementary Materials

1. **VLM Annotation Metadata**
   - Total API cost
   - Processing time
   - Model used and parameters

2. **Feature Correlation Matrix**
   - Heatmap of 16 D-IQM intercorrelations
   - Justification for feature selection

3. **Ablation Studies**
   - Performance with subset of features
   - Comparison to baseline methods

## Computational Requirements

### Hardware
- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB RAM, 8-core CPU, GPU optional
- **Storage**: ~50GB for datasets, ~1GB for outputs

### Software Dependencies
```
Python 3.9+
opencv-python (image processing)
scikit-learn (RandomForest models)
scikit-image (D-IQM computation)
pandas, numpy (data manipulation)
openai, anthropic, google-generativeai (VLM APIs)
```

### Estimated Costs (500 images)
- **VLM Annotation**: $0.50 (Gemini Flash) to $15 (Claude Sonnet)
- **Feature Computation**: Free (local CPU)
- **Model Training**: Free (local CPU)
- **Total Time**: 1-2 hours for 500 images

### Estimated Costs (Full Dataset ~5000 images)
- **VLM Annotation**: $5 (Gemini Flash) to $150 (Claude Sonnet)
- **Feature Computation**: Free (2-3 hours CPU time)
- **Model Training**: Free (< 5 minutes)
- **Total Time**: 4-6 hours for full pipeline

## Quality Assurance Checklist

Before running 500-image validation:
- [ ] All test files created and passing
- [ ] API keys verified in .env
- [ ] Test subset (50 images) successfully annotated
- [ ] Feature extraction tested on sample images
- [ ] Model training tested on small dataset
- [ ] Validation script runs without errors

Before full production run:
- [ ] 500-image validation results reviewed
- [ ] VLM label quality manually verified (sample)
- [ ] Feature distributions checked for outliers
- [ ] Model performance meets publication thresholds
- [ ] All 4 validation parts pass target metrics
- [ ] Fairness analysis shows <10% performance gap

## Troubleshooting Common Issues

### Issue: VLM API Rate Limits
**Solution**: Increase `--rate_limit_delay`, use checkpointing with `--resume`

### Issue: Feature Extraction Failures
**Solution**: Check image file integrity, verify OpenCV installation

### Issue: Class Imbalance in Training
**Solution**: Use `--class_balance` flag, adjust `--quality_threshold`

### Issue: Poor Model Performance
**Solution**: Review VLM label quality, check for data leakage, verify feature normalization

### Issue: Fairness Metrics Fail
**Solution**: Ensure sufficient samples per Fitzpatrick type, consider re-balancing training data

## References

**Primary Methodology**:
- Sanchez et al. (2024). "FetMRQC: A robust quality control system for multi-centric fetal brain MRI." *Medical Image Analysis*, 97.

**Teledermatology IQA Literature**:
- Chen et al. (2024). "Automated quality assessment in teledermatology using deep learning."
- Wang et al. (2023). "Color and contrast metrics for dermatologic image quality."
- Martinez et al. (2022). "Sharpness and exposure assessment for smartphone dermatology."

**Fairness in Medical AI**:
- Daneshjou et al. (2021). "Disparities in dermatology AI performance across skin tones."
- Groh et al. (2021). "Evaluating deep neural networks trained on clinical images in dermatology."
