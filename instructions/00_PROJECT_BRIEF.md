# Project Brief: Interpretable DIQA Framework (V2)

## 1. Guiding Philosophy
This project is an exercise in disciplined, AI-assisted development. The goal is not just to produce a model, but to build a clean, maintainable, and reproducible scientific software package. Every step should prioritize clarity, modularity, and robustness.

## 2. Core Objective
The primary goal is to create a robust, interpretable, and fair automated image quality assessment (IQA) framework for dermatology. The final output will be a standalone Python package (`diqa`) and a set of scripts that can classify a given dermatologic image as "Accept" or "Reject" for use in clinical AI pipelines.

## 3. Key Methodological Mandates (Non-Negotiable)
- **Framework:** We will adapt the modular "filter" methodology from neuroimaging. This means IQA is a separate, upstream step from any diagnostic model.
- **Features:** We will engineer a novel lexicon of **segmentation-free** Dermatologic Image Quality Metrics (D-IQMs). This is a critical constraint to ensure the tool is lightweight and has no external model dependencies.
- **Model:** The classifier **must** be an interpretable model, such as a `RandomForestClassifier`. We are prioritizing interpretability over a marginal gain in accuracy from a black-box model.
- **Ground Truth:** We will exclusively use dermatologist diagnostic confidence scores from the SCIN dataset as a proxy for quality labels. This is a "no-annotation" approach, and this constraint must be respected.

## 4. Validation Strategy (The Definition of Success)
The project is only successful if the final model is rigorously validated on three distinct criteria. The final report must contain clear results for each:
1.  **Internal Accuracy (SCIN):** The model must demonstrate high accuracy (AUC, Precision, Recall) on a held-out test set of real-world images from the SCIN dataset.
2.  **Specificity (ISIC):** The model must prove it does not incorrectly penalize high-quality images. It should classify the vast majority (>95%) of images from the ISIC dataset as "Accept".
3.  **Fairness (DDI):** The model's performance metrics **must be stratified by Fitzpatrick skin type** using the DDI dataset. There must be no clinically significant performance degradation for darker skin tones. This is the most critical validation step.

## 5. Phased Execution Plan
The project must be executed in the following sequence. Do not proceed to a phase until the previous one is complete.
1.  **Phase 0: Setup:** Initialize Git, create the directory structure, and establish these instruction files.
2.  **Phase 1: Data Ingestion:** Write a script to parse all dataset metadata and create a single `master_image_list.csv`.
3.  **Phase 2: D-IQM Engineering:** Write a script to compute the feature vector for every image in the master list and save it as `feature_matrix.csv`.
4.  **Phase 3: Model Training:** Write a script to load the feature matrix, train the classifier on the SCIN data, and save the final model as `derm_qc_model.joblib`.
5.  **Phase 4: Validation:** Write a script to load the trained model and run the complete three-part validation, generating a `final_validation_report.md`.
