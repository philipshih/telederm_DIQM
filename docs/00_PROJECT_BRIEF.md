# Project Brief: Interpretable DIQA Framework

## 1. Guiding Philosophy
The goal of this project is to build a clean, maintainable, and reproducible scientific software package. Every step in development should prioritize clarity, modularity, and robustness to ensure a high-quality, reliable final product.

## 2. Core Objective
The primary goal is to create a robust, interpretable, and fair automated image quality assessment (IQA) framework for dermatology. The final output will be a standalone Python package (`diqa`) and a set of scripts that can classify a given dermatologic image as "Accept" or "Reject" for use in clinical AI pipelines.

## 3. Key Methodological Mandates
- **Framework:** The project adapts the modular "filter" methodology from neuroimaging, where IQA is a separate, upstream step from any diagnostic model.
- **Features:** The project uses a novel lexicon of **segmentation-free** Dermatologic Image Quality Metrics (D-IQMs). This is a critical constraint to ensure the tool is lightweight and has no external model dependencies.
- **Model:** The classifier **must** be an interpretable model, such as a `RandomForestClassifier`, as we are prioritizing interpretability over a marginal gain in accuracy from a black-box model.
- **Ground Truth:** The project will exclusively use dermatologist diagnostic confidence scores from the SCIN dataset as a proxy for quality labels, following a "no-annotation" approach.

## 4. Validation Strategy (The Definition of Success)
The project's success is defined by a rigorous, three-part validation of the final model. The final report must contain clear results for each:
1.  **Internal Accuracy (SCIN):** The model must demonstrate high accuracy (AUC, Precision, Recall) on a held-out test set of real-world images.
2.  **Specificity (ISIC):** The model must prove it does not incorrectly penalize high-quality images, classifying the vast majority (>95%) of ISIC images as "Accept".
3.  **Fairness (DDI):** The model's performance metrics **must be stratified by Fitzpatrick skin type** using the DDI dataset, with no clinically significant performance degradation for darker skin tones.

## 5. Phased Execution Plan
The project will be executed in the following sequence:
1.  **Phase 0: Setup:** Initialize the repository and create the directory structure and documentation.
2.  **Phase 1: Data Ingestion:** Parse all dataset metadata to create a single `master_image_list.csv`.
3.  **Phase 2: D-IQM Engineering:** Compute the feature vector for every image and save it as `feature_matrix.csv`.
4.  **Phase 3: Model Training:** Train the classifier on the SCIN data and save the final model.
5.  **Phase 4: Validation:** Run the complete three-part validation and generate a final report.
