# 📱 Teen Phone Addiction Prediction

This project predicts the **Addiction Level** of teenagers based on lifestyle and behavioral data using a **regression model** built with **TensorFlow/Keras**. The final model includes hyperparameter tuning, preprocessing, and thorough evaluation.

---

## 📌 Problem Statement

Smartphone addiction is a growing concern, especially among teens. The objective of this project is to:

- Predict the **Addiction Level** (on a scale of 1 to 10)
- Use features like gender, phone usage purpose, screen time, etc.
- Build a robust regression model with good generalization performance

---

## 📂 Dataset

The dataset is sourced from Kaggle:  https://www.kaggle.com/datasets/khushikyad001/teen-phone-addiction-and-lifestyle-survey/data

**Teen Phone Addiction and Lifestyle Survey**  
It contains features such as:

- `Gender`
- `Phone_Usage_Purpose`
- `Daily_Screen_Time`
- `Sleep_Hours`
- `Social_Activity_Score`
- `Addiction_Level` *(Target Variable)*

---

## ⚙️ Workflow Overview

1. **Exploratory Data Analysis (EDA)**
   - Distribution plots, correlations, outlier detection (box plots)

2. **Preprocessing**
   - One-hot encoding of categorical features
   - Feature scaling (StandardScaler)
   - Train-test split

3. **Modeling**
   - Baseline models: Linear Regression, SVM, Random Forest
   - Advanced Model: Neural Network (Keras Sequential)

4. **Evaluation**
   - MAE, MSE, RMSE, R² Score
   - Residual plots, actual vs predicted scatter plots
   - Training vs validation loss visualization

---

## 📈 Model Performance

| Metric      | Value    |
|-------------|----------|
| **MAE**     | 0.4099   |
| **MSE**     | 0.2620   |
| **RMSE**    | 0.5119   |
| **R² Score**| 0.8960   |

✅ Indicates that the model explains ~90% of the variation in addiction levels and performs well on unseen data.

---

## 🧠 Key Concepts Used

- Neural Networks with Keras
- Adam optimizer
- Mean Squared Error (MSE) loss

---

## 📊 Visualizations

- Residual plots
- Actual vs Predicted scatter plot
- Training vs Validation Loss
- Boxplots to detect outliers

---

## ✅ Future Work

- Improve data balance using advanced sampling (SMOGN, KDE).
- Try ensemble models (XGBoost, RandomForestRegressor).
- Convert the regression model into a classification model for thresholds like Low/Moderate/High addiction.

## 🧪 Requirements

- Python 3.7+
- `tensorflow`
- `keras-tuner`
- `matplotlib`, `seaborn`
- `pandas`, `numpy`, `scikit-learn`

Install dependencies with:

```bash
pip install -r requirements.txt
