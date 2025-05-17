# Hybrid NN-LGBM for Agricultural Field Performance Prediction

## Project Overview

This project implements a machine learning pipeline to predict the productivity efficiency of agricultural fields. The goal is to classify fields into categories such as "high performing," "moderately performing," or "low performing" based on a variety of input features. The core of the solution involves a hybrid approach, utilizing a Neural Network for feature selection and a LightGBM model for the final classification.

**Notebook:** `code.ipynb`
**Repository Name :** `hybrid-nn-lgbm-agri-insights`
**Description :** `Hybrid Neural Network and LightGBM model for agricultural field performance classification, exploring direct feature learning from data with inherent missingness patterns.`

## Methodology

The `code.ipynb` notebook executes the following pipeline:

1.  **Data Loading and Initial Setup:**
    *   Loads `train.csv` (for training) and `test.csv` (for prediction).
    *   Maps the categorical `Target` labels to numerical representations.
    *   Separates features (X) from the target (y).

2.  **Preprocessing Strategy:**
    *   **Categorical Feature Encoding:** Employs one-hot encoding for categorical features using `pd.get_dummies()`.
    *   **Feature Alignment:** Ensures consistency in feature sets between training and test data using `X.align(X_test, join='left', axis=1, fill_value=0)`.
    *   **Handling of Missing/Blank Values:** This pipeline deliberately avoids explicit imputation of original missing or blank values in numerical features. The rationale is to prevent the introduction of potential bias that can arise from imputation techniques and to allow subsequent models to learn from or handle the inherent patterns of missingness if they are informative.

3.  **Neural Network (NN) for Feature Selection:**
    *   The preprocessed training data is split into training and validation sets for the NN.
    *   Numerical features are scaled using `StandardScaler`. *Note: If the original data (prior to scaling) contains NaNs or features with zero variance, the scaler might produce NaNs, which would then be passed to the NN.*
    *   A Sequential Neural Network (Dense layers with ReLU activation, Softmax output) is trained.
    *   Feature importances are derived from the mean absolute weights of the first dense layer of the trained NN.
    *   The top N (e.g., 40) features are selected based on these derived importances.

4.  **LightGBM for Classification:**
    *   The datasets are filtered to include only the top N features identified by the NN.
    *   A LightGBM classifier (`lgb.LGBMClassifier`) is configured, leveraging GPU acceleration (`device: 'gpu'`) and class balancing (`class_weight: 'balanced'`).
    *   The model's performance is assessed using 5-fold stratified cross-validation, reporting the average macro F1-score.
    *   A final LightGBM model is trained on the complete training dataset using the selected features.

5.  **Prediction and Output:**
    *   The trained LightGBM model generates predictions for the test set.
    *   Numerical predictions are converted back to their original string labels.
    *   A CSV file is produced containing `UID` and the corresponding `Target` predictions.

## Key Technical Aspects

*   **Hybrid Model:** Combines a Neural Network for automated feature selection with a LightGBM model for robust classification.
*   **Missing Data Strategy:** Explores the impact of allowing models to process data with its original patterns of missingness, rather than applying explicit imputation, to potentially avoid imputation-induced biases.
*   **GPU Acceleration:** Utilizes GPU capabilities for both TensorFlow (NN training) and LightGBM training for improved computational efficiency.
*   **Performance Metric:** The development and evaluation implicitly aim to optimize for a high macro F1-score, a suitable metric for multi-class classification, especially with potentially imbalanced classes.

## How to Run

1.  Ensure all Python dependencies listed in the import statements at the beginning of `code.ipynb` are installed (e.g., pandas, numpy, lightgbm, tensorflow, scikit-learn).
2.  Make `train.csv` and `test.csv` available at the paths specified within the notebook (e.g., `/kaggle/input/hackathon/` or adjust as needed).
3.  Execute the cells of the `code.ipynb` Jupyter Notebook sequentially.
4.  The output predictions will be saved to a CSV file (e.g., `output.csv`).
