# Heart Disease Predictor

A machine learning project that predicts whether a patient is prone to heart disease based on clinical features. The model is built using a Support Vector Classifier (SVC) with hyperparameter tuning via GridSearchCV, achieving a test ROC AUC of **0.93**.

---

## Dataset

The dataset (`data/heart.csv`) contains **918 patient records** with **11 clinical features** and a binary target label.

| Feature | Description |
|---|---|
| `Age` | Patient age in years |
| `Sex` | Sex (M / F) |
| `ChestPainType` | Type of chest pain (ATA, NAP, ASY, TA) |
| `RestingBP` | Resting blood pressure (mm Hg) |
| `Cholesterol` | Serum cholesterol (mg/dL) |
| `FastingBS` | Fasting blood sugar > 120 mg/dL (1 = true, 0 = false) |
| `RestingECG` | Resting ECG results (Normal, ST, LVH) |
| `MaxHR` | Maximum heart rate achieved |
| `ExerciseAngina` | Exercise-induced angina (Y / N) |
| `Oldpeak` | ST depression induced by exercise |
| `ST_Slope` | Slope of the peak exercise ST segment (Up, Flat, Down) |
| `HeartDisease` | **Target** — 1 = heart disease present, 0 = absent |

---

## Workflow

### 1. Data Loading & Exploration
The dataset is loaded with `pandas` and explored using `.info()`, `.describe()`, and `.head()` to understand the structure and statistics.

### 2. Data Imputation
Zero values in `RestingBP` and `Cholesterol` are biologically implausible and are treated as missing. They are replaced with `NaN` and then imputed using **median imputation** via `sklearn.impute.SimpleImputer`.

### 3. Exploratory Data Analysis (EDA)
Visual analysis is performed on the data to understand feature distributions and relationships with the target variable.

### 4. Preprocessing
Categorical features are encoded, and numerical features are scaled using `StandardScaler` to prepare the data for an SVM model.

### 5. Model Training
A **Support Vector Classifier (SVC)** is trained with an extensive **GridSearchCV** over three kernel types:

- **Linear kernel**: varying `C`
- **RBF kernel**: varying `C` and `gamma`
- **Polynomial kernel**: varying `C`, `gamma`, and `degree`

The search is optimized using **ROC AUC** as the scoring metric.

### 6. Evaluation
The best model is evaluated on a held-out test set.

---

## Results

| Metric | Value |
|---|---|
| Best Parameters | `C=100, gamma=0.01, kernel='rbf'` |
| Best CV ROC AUC | 0.9184 |
| **Test ROC AUC** | **0.9323** |
| Test Accuracy | 88% |
| Train Accuracy | 87.87% |

**Classification Report (Test Set):**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| 0 (No Disease) | 0.89 | 0.82 | 0.85 |
| 1 (Disease) | 0.86 | 0.92 | 0.89 |
| **Weighted Avg** | **0.88** | **0.88** | **0.87** |

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib / seaborn   # for EDA visualizations
jupyter
```

Install dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## Author

**Moaaz Ahmed**
- GitHub: [@Moaaz Ahmed](https://github.com/MomoSalter)