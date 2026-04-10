# Supervised Learning Practice

A collection of machine learning projects implementing and exploring different supervised learning algorithms. Each project focuses on applying a specific model to a real or synthetic dataset.

---

## Repository Folder Structure

> Note: every project has its own README.md

```
supervised-learning-practice/
|
+-- linear_regression/
|   +-- house_price_estimator
|   +-- money_spent_per_customer
|
+-- logistic_regression/
|   +-- ad_click_predictor
|   +-- diabetes_diagnosis
|
+-- KNN/
|   +-- classifier_model_for_artificial_data
|
+-- SVM/
    +-- cancer_predictor
    +-- heart_failure_predictor_using_grid_search
    +-- iris_flower_species_classifier_using_grid_search
```

---

## Projects Overview

### Linear Regression

#### House Price Predictor
**File:** `linear_regression/house_price_estimator/House_price.ipynb`

**Problem Statement:** Predict house sale prices from a large set of structural and quality-based features using a train/validation/test split.

**Overview:** Missing values were handled with domain-aware strategies — quality/condition columns were ordinally mapped using a custom dictionary, and `LotFrontage` was imputed using group medians per `LotShape`. Garage age was discretized into bins using `np.digitize`. Nominal features were encoded with `OneHotEncoder` and scaled with a sparse-safe `StandardScaler`. The model was evaluated on both train and validation sets to diagnose overfitting, and predictions were exported to a submission-ready CSV.

---

#### E-Commerce Money Spent Predictor
**File:** `linear_regression/money_spent_per_customer/money_spent_predictor.ipynb`

**Problem Statement:** Predict the yearly amount spent by e-commerce customers based on their app and website behavior.

**Overview:** EDA was done using pairplots and `lmplot` to visually identify the strongest predictor before training. After fitting the model, feature coefficients were interpreted to answer a business question: should the company invest more in the mobile app or the website? Residuals were plotted to verify the errors were normally distributed and unbiased.

---

### Logistic Regression

#### Ad Click Predictor
**File:** `logistic_regression/ad_click_predictor/ad_click_predictior.ipynb`

**Problem Statement:** Predict whether an internet user will click on an advertisement based on their behavioral and demographic profile.

**Overview:** The `Timestamp` column was parsed into hour, day, day-of-week, and month features, with hour encoded using cyclic sin/cos transformation to preserve its circular nature. High-cardinality columns like `City` and `Country` were dropped to avoid overfitting. Class balance was confirmed with a pie chart before training.

---

#### Diabetes Diagnostic Model
**File:** `logistic_regression/diabetes_diagnosis/diabetes_diagnostic_model.ipynb`

**Problem Statement:** Diagnose whether a patient has diabetes based on medical measurements including Glucose, BMI, Insulin, and Blood Pressure.

**Overview:** Biologically impossible zero values in medical columns were replaced with NaN and imputed with column medians. To handle the dataset's class imbalance, `class_weight='balanced'` was used in the logistic regression model to avoid bias toward the majority class.

---

### K-Nearest Neighbors

#### KNN Classifier
**File:** `KNN/classifier_model_for_artificial_data/knn_classifier.ipynb`

**Problem Statement:** Classify data points from an artificial dataset into binary target classes using KNN.

**Overview:** A pairplot was used to check for outliers and class structure in the synthetic data. The Elbow Method was applied by looping over multiple K values and plotting the error rate to automatically identify the optimal K. StandardScaler was applied before training since KNN is a distance-based algorithm.

---

### Support Vector Machine (SVM)

#### Cancer Predictor
**File:** `SVM/cancer_predictor/cancer_predictor.ipynb`

**Problem Statement:** Classify breast cancer tumors as malignant or benign using features derived from digitized cell nucleus images.

**Overview:** The dataset was loaded directly from `sklearn.datasets` using `load_breast_cancer`. A default `SVC()` was used as a clean baseline with no tuning to demonstrate how well SVM performs out of the box on well-separated data.

---

#### Heart Failure Predictor
**File:** `SVM/heart_failure_predictor_using_grid_search/heart_failure_predictor.ipynb`

**Problem Statement:** Predict the risk of heart disease from clinical features including cholesterol, resting blood pressure, chest pain type, and exercise-induced angina.

**Overview:** Zero values in medical columns were handled using `SimpleImputer`, and categorical features were encoded automatically using `pd.get_dummies` with `select_dtypes`. Features were scaled with `MinMaxScaler`. A `GridSearchCV` was run across three kernels (linear, rbf, poly) with a full parameter grid scored on ROC AUC, and `predict_proba` was enabled to compute the final AUC score on the test set.

---

#### Iris Species Classifier
**File:** `SVM/iris_flower_species_classifier_using_grid_search/iris_species_classifier.ipynb`

**Problem Statement:** Classify iris flowers into three species — Setosa, Versicolor, and Virginica — based on petal and sepal measurements.

**Overview:** The dataset was loaded from `sklearn.datasets` with species names mapped from integer targets for readability. `GridSearchCV` was used to tune C and gamma, with the best parameters printed after fitting. Multi-class classification was handled natively by SVM's one-vs-one strategy.

---

## Technologies Used

| Library | Purpose |
|---------|---------|
| Python 3.x | Core language |
| Jupyter Notebook | Interactive development environment |
| pandas | Data loading, cleaning, and manipulation |
| NumPy | Numerical operations and array handling |
| scikit-learn | ML models, preprocessing, tuning, and metrics |
| Matplotlib | Plotting and visualization |
| Seaborn | Statistical and exploratory visualizations |

---

## Algorithms Summary

| Algorithm              | Task           | Projects                                                        |
|------------------------|----------------|-----------------------------------------------------------------|
| Linear Regression      | Regression     | House Price Prediction, E-Commerce Spending Prediction          |
| Logistic Regression    | Classification | Ad Click Prediction, Diabetes Diagnosis                         |
| K-Nearest Neighbors    | Classification | KNN Classifier (Artificial Data)                                |
| Support Vector Machine | Classification | Cancer Detection, Heart Failure Prediction, Iris Classification |

---

## Author

**Moaaz Ahmed**
- GitHub: [@Moaaz Ahmed](https://github.com/MoaazSalter)