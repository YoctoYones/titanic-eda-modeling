# Titanic Survival Prediction

Exploratory data analysis and classification modeling on the Kaggle "Titanic: Machine Learning from Disaster" dataset, predicting whether a passenger survived based on their attributes.

## Contents

- `titanic_eda_modeling.ipynb` — full, reproducible notebook: EDA, data cleaning, feature engineering, model training/tuning, and results.

## Approach

1. **EDA** — missing value analysis (including checking whether missingness itself is informative, e.g. `Cabin`), target-vs-feature visualizations, correlation, and chi-square/Cramér's V tests for categorical associations.
2. **Feature engineering** — `Has_Cabin` (from `Cabin` missingness), grouped median imputation for `Age` (by `Pclass`), `FamilyGroup` (bucketed family size), `Title` (extracted from `Name`), and a log-transformed `Fare`.
3. **Modeling** — a `LinearSVC` baseline, tuned via manual sweep, `RandomizedSearchCV`, and `GridSearchCV`, compared against a tuned `RandomForestClassifier`.
4. **Evaluation** — accuracy, precision/recall/F1, confusion matrices, ROC-AUC, and feature importance/coefficients, all validated with stratified k-fold cross-validation.

## Key findings

- **Sex** is the single strongest predictor of survival, followed by passenger class/fare and family size.
- **Feature engineering had a bigger impact than hyperparameter tuning** — adding the `Title` feature improved accuracy more than tuning `LinearSVC`'s hyperparameters did.
- A tuned linear model and a tuned random forest converge to similar performance (~0.83–0.84 cross-validated accuracy), suggesting this is close to the practical ceiling for this feature set.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```

## Usage

Place `train.csv` (from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic)) in the same directory, then run the notebook top to bottom.
