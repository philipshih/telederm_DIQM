# Core Methodological Decisions

This document outlines the non-negotiable methodological mandates for this project. These decisions are foundational to the project's design and must be adhered to in all implementation and validation steps.

## 1. Data Handling: Image-Level Independence

- **Mandate:** Every image must be treated as an independent data point.
- **Justification:** We are building an **image** quality assessment tool, not a case-level tool. This approach maximizes the dataset size, simplifies the modeling task, and aligns with standard computer vision practices, leading to a more robust and interpretable result. Case-level aggregation of scores is not permitted.

## 2. Ground Truth: Binary Classification from Proxy

- **Mandate:** The ground truth for model training will be a binary label derived from dermatologist confidence scores in the SCIN dataset.
  - Confidence scores of **1-2** will be mapped to Class **0 (Reject)**.
  - Confidence scores of **3-5** will be mapped to Class **1 (Accept)**.
- **Justification:** This creates a pragmatic, clinically relevant, and actionable output ("Is this image usable, yes or no?"). It simplifies the modeling task and focuses the project on solving the primary problem first, making the results clear and the narrative strong. Regression or multi-class classification is out of scope for the initial model.

## 3. Feature Engineering: Segmentation-Free Mandate

- **Mandate:** All Dermatologic Image Quality Metrics (D-IQMs) must be "no-reference" and calculated from the raw image pixels **without** the use of a segmentation model.
- **Justification:** This is a critical design constraint to ensure the final tool is lightweight, generalizable, and has no external deep learning dependencies. It removes a significant confounding variable (the performance of the segmentation model) and makes the tool far easier for other researchers to adopt and use on any dataset. Metrics requiring a lesion mask are explicitly excluded.

## 4. Validation: Rigorous Multi-Dataset Strategy

- **Mandate:** The final model must be validated using the specific three-part strategy outlined below. A simple train/test split on a single dataset is insufficient.
- **Justification:** This rigorous approach is essential for proving the model's real-world utility and fairness, which is a cornerstone of the project's scientific contribution.
  - **1. Internal Validation (SCIN):** An 80/20 split of the SCIN dataset will be used for training and internal testing to establish baseline performance on real-world data.
  - **2. High-Quality Control (ISIC):** The trained model will be tested on the ISIC dataset to prove high specificity and a low false-positive rate.
  - **3. Fairness & Bias Assessment (DDI):** The trained model will be tested on the DDI dataset, with performance metrics explicitly stratified by Fitzpatrick skin type. This is a non-negotiable step to ensure algorithmic fairness.
