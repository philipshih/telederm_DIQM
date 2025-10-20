# Project Provenance & Data Acquisition

This document details the project's methodological origins and provides instructions for acquiring the necessary public datasets.

## 1. Methodological Provenance

This project adapts the **methodology and modular philosophy** of the `fetmrqc` project, a quality control framework from the field of neuroimaging.

**Reference Publication:**
- **Title:** *FetMRQC: An open-source quality control tool for fetal brain MRI*
- **Repository:** [https://github.com/Medical-Image-Analysis-Laboratory/fetmrqc](https://github.com/Medical-Image-Analysis-Laboratory/fetmrqc)

**Justification for Re-implementation:**
A clean, minimal re-implementation was chosen over a direct fork for several strategic reasons:
- **Domain Mismatch:** The `fetmrqc` codebase is highly specialized for 3D/4D medical imaging data and was not suitable for direct application to 2D dermatology images.
- **Simplicity and Minimalism:** A clean implementation avoids inheriting complex and irrelevant dependencies from the original neuroimaging-focused repository.
- **Dependency Weight:** This project is designed to be lightweight, in contrast to the large containerized environment of the original.

Our approach honors the scientific contribution of `fetmrqc` by applying its successful modular pattern (`Data Ingestion -> Feature Calculation -> Model Training -> Validation`) to a new domain.

## 2. Data Acquisition Instructions

This project requires three public datasets. The following steps must be completed manually due to data usage agreements.

**Step 1: Download Compressed Files**
Visit the following URLs, agree to the respective terms of use, and download the compressed files into the `data/downloads/` directory.

- **SCIN (Skin Condition Image Network):**
  - **URL:** `https://stanfordaimi.azurewebsites.net/datasets/ca03db2b-3c70-405a-8159-355610bbe0ae`
  - **File:** `scin_v1_1.zip`

- **DDI (Diverse Dermatology Images):**
  - **URL:** `https://stanfordaimi.azurewebsites.net/datasets/503c76e2-2c63-4351-92a2-b80e55b4b3a7`
  - **File:** `ddi_v1.zip`

- **ISIC (ISIC 2019 Challenge):**
  - **URL:** `https://challenge.isic-archive.com/data/`
  - **File:** `ISIC_2019_Training_Input.zip`

**Step 2: Unzip Files**
Once downloaded, unzip each file into its corresponding target directory:
- Unzip `scin_v1_1.zip` into `data/scin/`.
- Unzip `ddi_v1.zip` into `data/ddi/`.
- Unzip `ISIC_2019_Training_Input.zip` into `data/isic/`.

After completing these steps, the project will be ready for the execution of the automated pipeline scripts.
