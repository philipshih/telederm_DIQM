# Teledermatology Image Quality Assessment (DIQA) Framework

An interpretable, segmentation-free image quality control system for teledermatology using VLM knowledge distillation and FetMRQC methodology.

## Overview

This project implements automated quality assessment for dermatologic images using:
- **16 segmentation-free D-IQM features**
- **VLM knowledge distillation** from Gemini/GPT-4o/Claude
- **Interpretable RandomForest models** (QA regression + QC classification)
- **Leave-one-dataset-out cross-validation** (FetMRQC approach)
- **Baseline comparisons and feature importance** 
- **Multi-dataset validation** (SCIN, ISIC, DDI)

**Status**: FetMRQC requirements implemented. VLM annotation ongoing (40/500 images).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API keys (create .env from .env.example)
cp .env.example .env

# Run pipeline (all stages auto-skip if cached outputs are current)
python scripts/02_vlm_annotation.py --provider google        # Resumes from checkpoint; use --force to re-label
python scripts/02_compute_d_iqms.py                          # Recomputes only when master list changes
python scripts/03_train_quality_model.py                     # Retrains if features/labels are newer
python scripts/04_run_validation_enhanced.py                 # Publication-ready validation with cached figures

# LLM annotation guidance (keep it lightweight)
# - Default master list caps at 500 images to control API usage
# - Pass --limit 500 to `01_load_derm_data.py` if you regenerate the master list
# - All annotation scripts honour existing checkpoints; use --force for a clean rerun

# Run tests
pytest tests/ -v
```

## Repository Structure

```
├── data/           # Raw datasets (SCIN, ISIC, DDI) - gitignored
├── docs/           # Complete documentation
│   ├── PROJECT_WORKFLOW.md      # Full pipeline guide
│   ├── METHODOLOGY.md           # Core decisions
│   ├── TESTING_SUMMARY.md       # Test results
│   └── DELIVERABLES.md          # Project summary
├── models/         # Trained models (QA + QC) - gitignored
├── reports/        # Pipeline outputs - gitignored
├── scripts/        # Pipeline execution scripts
├── src/diqa/       # Core DIQA library (metrics, VLM client)
└── tests/          # 52 comprehensive tests
