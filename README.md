# Heart Disease Prediction

A beginner-friendly machine learning project that predicts whether a patient has heart disease using Logistic Regression on the Cleveland Heart Disease Dataset.

---

## Project Structure

```
Heart_Disease_Prediction_Project/
│
├── dataset/
│   └── heart_disease_data.csv       # Cleveland Heart Disease Dataset (303 rows, 14 columns)
│
├── outputs/
│   ├── plot1_target_distribution.png   # Bar chart: class distribution
│   ├── plot2_age_distribution.png      # Histogram: age by disease status
│   ├── plot3_correlation_heatmap.png   # Heatmap: feature correlations
│   ├── plot4_heartrate_vs_age.png      # Scatter: max heart rate vs age
│   └── plot5_confusion_matrix.png      # Confusion matrix of test predictions
│
├── Heart_Disease_Prediction.ipynb   # Main Jupyter Notebook
└── README.md                        # Project documentation
```

---

## Dataset

**Source:** Cleveland Heart Disease Dataset (via UCI Machine Learning Repository)  
**File:** `dataset/heart_disease_data.csv`  
**Rows:** 303 | **Columns:** 14 (13 features + 1 target)

| Column | Description |
|--------|-------------|
| `age` | Age of the patient (years) |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment (0–2) |
| `ca` | Number of major vessels coloured by fluoroscopy (0–4) |
| `thal` | Thalassemia type (0–3) |
| `target` | **Heart disease present (1 = yes, 0 = no)** |

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical operations |
| `pandas` | Data loading and preprocessing |
| `matplotlib.pyplot` | Data visualisation |
| `scikit-learn` | ML model, scaling, evaluation |

---

## Workflow

```
1. Import Libraries
       ↓
2. Load Dataset
       ↓
3. Data Preprocessing
   - Shape, info, data types
   - Missing values check
   - Duplicate removal
   - Summary statistics
       ↓
4. EDA (matplotlib.pyplot only)
   - Target class distribution
   - Age distribution by disease status
   - Feature correlation heatmap
   - Max heart rate vs age scatter plot
       ↓
5. Define X and y
       ↓
6. Train-Test Split (80% / 20%, stratified)
       ↓
7. Feature Scaling (StandardScaler)
       ↓
8. Train Logistic Regression Model
       ↓
9. Evaluate (accuracy, confusion matrix, classification report)
       ↓
10. Predictive System (predict for new patient input)
```

---

## Model Details

| Parameter | Value | Reason |
|-----------|-------|--------|
| `solver` | `lbfgs` | Efficient for small/medium datasets |
| `C` | `0.1` | Stronger regularisation to prevent overfitting |
| `max_iter` | `1000` | Ensures full convergence |
| `random_state` | `42` | Reproducibility |

---

## Results

| Metric | Score |
|--------|-------|
| **Train Accuracy** | 84.23% |
| **Test Accuracy** | 77.05% |

**Classification Report (Test Set):**

```
               precision    recall  f1-score   support

   No Disease       0.82      0.64      0.72        28
Heart Disease       0.74      0.88      0.81        33

     accuracy                           0.77        61
    macro avg       0.78      0.76      0.76        61
 weighted avg       0.78      0.77      0.77        61
```

**Confusion Matrix:**
```
              Predicted
              No   Yes
Actual  No  [ 18   10 ]
        Yes [  4   29 ]
```

---

## How to Run

### 1. Clone or download the project folder

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook Heart_Disease_Prediction.ipynb
```

### 4. Run all cells
Go to **Kernel → Restart & Run All**

---

## Notes

- All EDA plots are saved automatically to the `outputs/` folder via `plt.savefig()`.
- `stratify=y` is used in the train-test split to maintain class balance.
- Feature scaling (StandardScaler) is applied **after** the split to prevent data leakage.
- The dataset has **no missing values** and **1 duplicate row** (removed during preprocessing).

