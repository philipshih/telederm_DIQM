# Project Brief: Interpretable DIQA Framework

## 1. Objective
The primary goal is to create a robust, interpretable, and fair automated image quality assessment (IQA) framework for dermatology. The final output will be a standalone tool that classifies images as "Accept" or "Reject" for use in clinical AI pipelines.

## 2. Core Methodology
- **Framework:** Adapt the modular `fetmrqc` "filter" methodology.
- **Features:** Engineer a novel lexicon of segmentation-free Dermatologic Image Quality Metrics (D-IQMs).
- **Model:** Train an interpretable `RandomForestClassifier`.
- **Ground Truth:** Use dermatologist diagnostic confidence scores from the SCIN dataset as a proxy for quality labels. This is a "no-annotation" approach.

## 3. Validation Strategy (Non-Negotiable)
The model's success will be judged on a three-part validation:
1.  **Internal Accuracy (SCIN):** The model must be accurate on a held-out test set of real-world images.
2.  **Specificity (ISIC):** The model must not incorrectly reject high-quality, expert-curated images.
3.  **Fairness (DDI):** The model must demonstrate equitable performance across all Fitzpatrick skin tones.

## 4. Phased Execution Plan
The project will be executed in the following phases:
1.  **Data Ingestion:** Consolidate all dataset metadata.
2.  **D-IQM Engineering:** Compute and save the feature matrix for all images.
3.  **Model Training:** Train the classifier on the SCIN dataset.
4.  **Validation:** Run the three-part validation and generate a final report.
