# Diabetes Prediction using Random Forest

This project uses the Pima Indians Diabetes dataset to build a classification model that predicts whether a patient has diabetes based on diagnostic measurements.

## 🔍 Problem Statement

Given a dataset of medical attributes, predict whether a patient is likely to develop diabetes (1) or not (0).

## 📊 Dataset

- Source: `diabetes.csv`
- Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- Target: `Outcome` (1 for diabetic, 0 for non-diabetic)

## ⚙️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib/Seaborn (Optional for visualization)
- Jupyter Notebook / Python

## 📌 Steps Performed

1. **Data Cleaning**:
   - Replaced invalid zeros with NaN in relevant columns.
   - Imputed missing values with median.

2. **Preprocessing**:
   - Feature scaling using `StandardScaler`.
   - Stratified train-test split.

3. **Modeling**:
   - `RandomForestClassifier` with `GridSearchCV` for hyperparameter tuning.

4. **Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Feature importance

## 🧪 Results

- Best Parameters: *(Displayed by code)*
- Accuracy: *0.XXXX*
- F1-Score: *0.XXXX*

## 📈 Feature Importance (Top 3)

1. Glucose  
2. BMI  
3. Age  

## 🛠 Installation

```bash
git clone https://github.com/your-username/Diabetes-Prediction.git
cd Diabetes-Prediction-rf
pip install -r requirements.txt
