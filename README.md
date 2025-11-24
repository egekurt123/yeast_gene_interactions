# Yeast Gene Interaction Prediction

Machine-learning workflows for predicting yeast gene essentiality, genetic interaction (GI) scores, and matrix-scale interaction profiles from multi-modal embeddings.

---

## Project Report

The project report is a joint report with my other project: **Predicting Cancer Genes in Homo Sapiens Using Gene Embeddings**, available at: https://github.com/egekurt123/RNA-gene-disease-predictions

---

### Requirements

- Python 3.9+

---

## Environment Setup

1. Create a Python environment and install dependencies:

2. Download Costanzo SGA matrices, Turco gene-expression files, and YeastNet edges into `data/`, then run the wrangling notebooks in `data_wrangling/` to populate `extracted_data/`.

---

## Core Workflows

### 1. Classifying Essential Genes
- Load embeddings (DNALM, YeastNet, Turco expression, or orthogroup features) via the notebooks.
- Use `class_prediction_helpers.get_class_predictions`for standard stratified training or `class_prediction_helpers.get_class_predictions_balanced` for SMOTE/undersampled runs.
- Visualization cells rely on scikit-learn precision/recall metrics and `PrecisionRecallDisplay`.

### 2. Table-Level GI Score Regression
- Combine embeddings through `Table Predictions/from_embeddings/Interaction_Score_Combinations.ipynb` to benchmark Linear Regression, PCA pipelines, Random Forest, and Ridge/Lasso models.
- Specialized notebooks like `GI_Score/DNALM_YeastNet.ipynb` import shared utilities `embedding_prediction_helpers` for consistent preprocessing.

### 3. Matrix-Wide Interaction Prediction
- Train regressors on full Costanzo matrices with `predict_whole_dataset.py` tweak the `predictive_models` list (e.g., `Ridge`, `RandomForestRegressor`, `create_neural_network`) defined in `prediction_helpers`
- Use `iterate_over_proportion_only_interactions` / `iterate_over_proportion_only_embeddings` to perform per-gene R² sweeps and plot distributions via `plot_results`.

