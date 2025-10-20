# Project Vision & Scientific Goals

This document outlines the core vision and scientific contribution of this project. It serves as the strategic guide for development decisions.

## 1. The Clinical Problem
The increasing use of teledermatology has led to a flood of patient-submitted images of highly variable quality. While deep learning models have shown great promise in dermatology, their performance is often brittle when faced with real-world data that does not match the high-quality, curated datasets they were trained on. This creates a significant patient safety risk, as poor-quality images can lead to silent and critical diagnostic errors.

This project aims to address this challenge by creating a foundational, automated tool to serve as a "gatekeeper," ensuring that only images of sufficient quality enter a clinical AI pipeline.

## 2. The Scientific Contribution
This project addresses three key gaps in the current literature:

1.  **Interpretability:** Many existing quality assessment tools are "black-box" models, making it difficult to understand their failure modes. This project prioritizes interpretability by using a feature-based `RandomForestClassifier`, which allows for a clear understanding of why an image is deemed low-quality.
2.  **Fairness:** There is a significant risk of algorithmic bias against darker skin tones in dermatologic AI. This project makes fairness a primary validation endpoint, using the DDI dataset to rigorously test for and mitigate this risk.
3.  **Pragmatism:** High-quality, manually annotated datasets are a major bottleneck in medical AI research. This project introduces a pragmatic "no-annotation" methodology that leverages existing dermatologist confidence scores as a scalable proxy for ground truth, providing a cost-effective blueprint for future research.

## 3. Guiding Principles for Development
All development choices should be guided by the following principles:

*   **Simplicity Over Complexity:** Prioritize simple, direct, and easily understood implementations.
*   **Interpretability Over Marginal Performance:** A small gain in accuracy is not worth sacrificing the model's interpretability.
*   **Robustness Over Speed:** Ensure code is robust and handles errors gracefully.
*   **Generalizability Over Specialization:** The tool must be as general-purpose as possible, hence the "segmentation-free" constraint.
*   **Adherence to the Plan:** Follow the established, phased execution plan to ensure a systematic and reproducible workflow.
