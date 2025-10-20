# Project Provenance: Adaptation Strategy & Data Acquisition

This document details the project's relationship with the `fetmrqc` repository and provides the necessary instructions for acquiring the public datasets.

## 1. Adaptation Strategy from `fetmrqc`

This project adapts the **methodology and modular philosophy** of the `fetmrqc` project, not its literal source code.

**Reference Repository:** [https://github.com/Medical-Image-Analysis-Laboratory/fetmrqc](https://github.com/Medical-Image-Analysis-Laboratory/fetmrqc)

**Justification for Re-implementation:**
We are **not** forking or cloning the `fetmrqc` repository directly for the following strategic reasons:
- **Domain Mismatch:** The `fetmrqc` codebase is highly specialized for 3D/4D neuroimaging data (NIfTI files, BIDS format) and contains complex dependencies (e.g., `nnUNet`, `MONAIfbs`) that are entirely irrelevant to our 2D dermatology image analysis.
- **Simplicity and Minimalism:** Cloning and then surgically removing the large, irrelevant portions of the original repository would be more complex and error-prone than a clean, minimal implementation.
- **Dependency Weight:** The original project recommends a 35GB Docker container. Our re-implementation is lightweight and relies on a small, standard set of data science libraries (`pandas`, `opencv-python`, `scikit-learn`).

Our approach is to honor the scientific contribution of `fetmrqc` by building a new, clean pipeline that follows its successful modular pattern: `Data Ingestion -> Feature Calculation -> Model Training -> Validation`.

## 2. Data Acquisition Instructions

This project requires three public datasets. The following steps must be completed manually due to data usage agreements.

**Step 1: Download Compressed Files**
Visit the following URLs, agree to the respective terms of use, and download the compressed files into the `data/downloads/` directory.

- **SCIN (Skin Condition Image Network):**
  - **URL:** `https://stanfordaimi.azurewebsites.net/datasets/ca03db2b-3c70-405a-8159-355610bbe0ae`
  - **File:** `scin_v1_1.zip`
  - **Action:** Download and place in `data/downloads/scin_v1_1.zip`.

- **DDI (Diverse Dermatology Images):**
  - **URL:** `https://stanfordaimi.azurewebsites.net/datasets/503c76e2-2c63-4351-92a2-b80e55b4b3a7`
  - **File:** `ddi_v1.zip`
  - **Action:** Download and place in `data/downloads/ddi_v1.zip`.

- **ISIC (ISIC 2019 Challenge):**
  - **URL:** `https://challenge.isic-archive.com/data/`
  - **File:** `ISIC_2019_Training_Input.zip`
  - **Action:** Register for an account, download the training input, and place it in `data/downloads/ISIC_2019_Training_Input.zip`.

**Step 2: Unzip Files**
Once downloaded, unzip each file into its corresponding target directory:
- Unzip `scin_v1_1.zip` into `data/scin/`.
- Unzip `ddi_v1.zip` into `data/ddi/`.
- Unzip `ISIC_2019_Training_Input.zip` into `data/isic/`.

After completing these steps, the project will be ready for the execution of the automated pipeline scripts.
