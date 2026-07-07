# Breast Cancer Diagnostic System using Machine Learning

## Overview

The Breast Cancer Diagnostic System is a machine learning application developed in Python to classify breast tumors as either Benign (Non-Cancerous) or Malignant (Cancerous).

The project uses the Wisconsin Breast Cancer Diagnostic Dataset and compares multiple machine learning algorithms to identify the best-performing model. It also provides a simple console-based interface for training, evaluating, and predicting breast cancer diagnoses.

---

## Features

* Data preprocessing and cleaning
* Automatic label encoding
* Feature scaling using StandardScaler
* Training of multiple machine learning models
* Automatic best model selection
* Model performance comparison
* Confusion Matrix
* ROC Curve
* Classification Report
* Interactive console-based prediction system
* Model saving using Joblib

---

## Machine Learning Algorithms

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* Naive Bayes
* Gradient Boosting

---

## Project Structure

```text
Breast_Cancer_Diagnostic/
│
├── dataset/
│   └── breast_cancer.csv
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│   └── model_comparison.csv
│
├── preprocessing.py
├── train.py
├── evaluate.py
├── predict.py
├── interface.py
├── requirements.txt
└── README.md
```

---

## Dataset

Dataset Name:
Wisconsin Breast Cancer Diagnostic Dataset

Target Classes

* M → Malignant
* B → Benign

Number of Features

* 30 Numerical Features

Target Column

* diagnosis

Unused Column

* id

---

## Requirements

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

Required Packages

* numpy
* pandas
* matplotlib
* scikit-learn
* joblib

---

## How to Run

Step 1

Clone the repository.

```bash
git clone https://github.com/yourusername/Breast_Cancer_Diagnostic.git
```

Step 2

Move into the project folder.

```bash
cd Breast_Cancer_Diagnostic
```

Step 3

Install the required libraries.

```bash
pip install -r requirements.txt
```

Step 4

Run the application.

```bash
python interface.py
```

---

## Application Menu

```text
============================================================
        BREAST CANCER DIAGNOSTIC SYSTEM
============================================================
1. Train Machine Learning Models
2. Evaluate Best Model
3. Predict Breast Cancer
4. Exit
============================================================
```

---

## Workflow

1. Load Dataset
2. Preprocess Data
3. Split Training and Testing Data
4. Scale Features
5. Train Multiple Models
6. Compare Model Accuracy
7. Save Best Model
8. Evaluate Performance
9. Predict New Patient Diagnosis

---

## Output

Training

* Trains seven machine learning models
* Displays model accuracies
* Selects the best-performing model
* Saves the trained model

Evaluation

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix
* ROC Curve

Prediction

* Accepts 30 feature values
* Predicts Benign or Malignant
* Displays confidence score

---

## Files Generated

After training, the following files are created automatically.

```text
models/
│
├── best_model.pkl
├── scaler.pkl
├── features.pkl
└── model_comparison.csv
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Joblib

---

## Future Improvements

* Hyperparameter tuning
* Feature selection
* Explainable AI using SHAP or LIME
* Graphical User Interface
* Web application deployment
* Deep Learning model comparison

---

## Author

Ashok Aathreyan U

Artificial Intelligence and Data Science

---

## License

This project is developed for educational and academic purposes.
