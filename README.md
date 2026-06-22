# Maternal Health Risk Stratification — Two-Cohort ML Analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20792062.svg)](https://doi.org/10.5281/zenodo.20792062)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview

This repository contains the analysis code and figures for:

> Rahman MR, Rose JI, Rahman MR. **Machine learning for maternal health risk stratification using IoT-collected clinical datasets from Bangladesh: a two-cohort comparative analysis.** *Heliyon* (under review).

Six machine learning classifiers (Random Forest, XGBoost, LightGBM, Decision Tree, Logistic Regression, SVM) are applied to two independent IoT-collected Bangladeshi maternal health datasets. SHAP analysis identifies the dominant predictors in each cohort.

## Key Results

| Cohort | Features | Outcome | Best model accuracy | High-risk recall | Top SHAP predictor |
|--------|----------|---------|-------------------|------------------|--------------------|
| UCI (n=1,014) | 6 physiological | 3-class | 85.22% | 96.36% | Blood glucose |
| Mendeley (n=1,186) | 11 (incl. BMI, diabetes, mental health) | Binary | 99.58% | 100.00% | BMI |

Extending the feature set from 6 to 11 parameters raises accuracy by 14 percentage points and eliminates high-risk misclassification.

## Datasets

| Cohort | Source | DOI | n | Features | Outcome |
|--------|--------|-----|---|----------|---------|
| Cohort 1 | UCI ML Repository | [10.24432/C5DP5D](https://doi.org/10.24432/C5DP5D) | 1,014 | 6 | 3-class |
| Cohort 2 | Mendeley Data | [10.17632/p5w98dvbbk.1](https://doi.org/10.17632/p5w98dvbbk.1) | 1,186 | 11 | Binary |

Place downloaded CSV files in the `data/` directory:
- `data/maternal_health_risk_uci.csv`
- `data/maternal_health_risk_mendeley.csv`

## Repository Structure

```
maternal-health-ml-two-cohort/
├── analysis.py          # Full ML pipeline for both cohorts
├── requirements.txt     # Python dependencies
├── CITATION.cff         # Citation metadata
├── LICENSE              # CC BY 4.0
├── figures/             # All manuscript figures (300 DPI PNG)
│   ├── cohort1/         # Cohort 1 figures (UCI)
│   └── cohort2/         # Cohort 2 figures (Mendeley)
└── data/                # Place downloaded datasets here (not included)
```

## Reproducing the Analysis

```bash
# 1. Clone the repository
git clone https://github.com/RafinRahman/maternal-health-ml-two-cohort.git
cd maternal-health-ml-two-cohort

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets and place in data/ directory

# 4. Run analysis
python analysis.py
```

## Environment

- Python 3.12
- scikit-learn 1.4, XGBoost 2.0, LightGBM 4.3, SHAP 0.44
- matplotlib 3.8, seaborn 0.13, pandas 2.1, numpy 1.26

## Authors

- **Md Rafin Rahman** (corresponding) — Institute for Developing Science and Health Initiatives (ideSHi), Dhaka, Bangladesh. ORCID: [0009-0007-5366-0175](https://orcid.org/0009-0007-5366-0175)
- **Jafren Iqbal Rose** — Shaheed Monsur Ali Medical College and Hospital, Dhaka, Bangladesh
- **Md Rofiqur Rahman** — Institute for Developing Science and Health Initiatives (ideSHi), Dhaka, Bangladesh

## License

This work is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use this code, please cite the associated manuscript (DOI to be updated upon publication) and the Zenodo code deposit:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20792062.svg)](https://doi.org/10.5281/zenodo.20792062)
