# Contingency Plan & Troubleshooting Guide

This document outlines the pre-planned responses to potential failures or unexpected results during the validation phase. If a validation test fails, this guide must be followed to pivot the project strategy without requiring human intervention.

## 1. Failure Mode 1: The ISIC Specificity Test Fails

-   **Scenario:** The trained model incorrectly classifies a significant portion (>5%) of the high-quality ISIC dataset images as "Reject." This indicates a domain shift problem where a feature common in dermoscopy is being misinterpreted as a quality defect.

-   **Contingency Protocol:**
    1.  **Isolate Failures:** Filter the ISIC prediction results to identify all images that were incorrectly rejected.
    2.  **Analyze Feature Vectors:** Analyze the D-IQM feature vectors for this "failed" subset. Identify which specific metric(s) have outlier scores compared to the correctly accepted ISIC images. (e.g., Do they all have unusually high `hair_edges` scores? Do they all have unusual `color_cast` scores?).
    3.  **Hypothesize the Confounder:** Based on the outlier metric, form a hypothesis. For example, if the `hair_edges` score is the problem, the hypothesis is that the edge detection algorithm is incorrectly identifying dermoscopic rulers or other artifacts as hair.
    4.  **Refine the Metric:** Create a "version 2" of the confounding metric function in `src/diqa/metrics.py`. For example, `compute_hair_edges_v2`. This new function should be modified to be less sensitive to the identified confounder.
    5.  **Re-run and Validate:** Re-run the entire pipeline (feature extraction, training, and validation) using this refined metric. The project narrative will now include this discovery as a key finding.

## 2. Failure Mode 2: The DDI Fairness Test Fails (Critical)

-   **Scenario:** The model shows a clinically significant performance degradation on images of darker skin tones (e.g., Fitzpatrick V-VI) in the DDI dataset. This indicates the model is algorithmically biased. This is the most important potential failure.

-   **Contingency Protocol:** **Do NOT discard the biased model.** This is now the most important result of the study. The project goal immediately pivots from "building a tool" to "building a *fair* tool and demonstrating *why* the naive approach fails."

    1.  **Preserve the Biased Model:** The current model (`derm_qc_model.joblib`) is now "Model V1" or the "Naive Model."
    2.  **Prove the Bias:** The source of the bias is almost certainly a metric that confounds low lesion contrast with poor quality (e.g., `sharpness_laplacian`). You must prove this by generating a statistical analysis (e.g., a plot or table) showing the correlation between the suspected metric's score and the Fitzpatrick skin type in the DDI dataset.
    3.  **Develop a Fairer Metric:** The next step is to engineer a new, "fairness-aware" metric. For example, a contrast-invariant sharpness metric could be developed that is less sensitive to the overall contrast between a lesion and the surrounding skin. This new function will be added to `src/diqa/metrics.py`.
    4.  **Train "Model V2":** Train a new RandomForestClassifier using the exact same procedure as before, but with the new, fairer metric replacing the biased one. This is "Model V2" or the "Fairness-Aware Model."
    5.  **Re-Validate Both Models:** Run *both* Model V1 and Model V2 through the full validation pipeline, especially the DDI fairness test.
    6.  **Pivot the Narrative:** The final report and publication will now be centered on this discovery. The narrative becomes: "We demonstrate that a naive IQA model exhibits significant bias, identify the specific feature (`sharpness_laplacian`) that causes it, and propose a novel, fairness-aware metric that successfully mitigates this bias, leading to an equitable and robust model."

This contingency plan provides a clear, deterministic path forward in the event of the most likely and most important project risks.
