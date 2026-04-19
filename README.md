# 🏠 House Price Prediction using Machine Learning

This project focuses on predicting housing prices using various machine learning models. It includes data preprocessing, exploratory data analysis (EDA), feature engineering, and model evaluation.

---

## 📂 Project Structure

- `HousePricePrediction.xlsx` – Dataset used for training and evaluation  
- `code.py` – Complete implementation of preprocessing, modeling, and evaluation :contentReference[oaicite:0]{index=0}  
- `correlation_heatmap.png` – Heatmap visualization of feature correlations  

---

## 🚀 Features

- Data preprocessing:
  - Handling missing values (mean for numerical, mode for categorical)
  - Dropping irrelevant features (e.g., `Id`)
- Exploratory Data Analysis:
  - Correlation heatmap
  - Categorical feature distribution analysis
- Feature Engineering:
  - One-Hot Encoding for categorical variables
- Model Training:
  - Linear Regression
  - Random Forest Regressor
  - Support Vector Regressor (SVR)
- Model Evaluation:
  - Mean Absolute Error (MAE)
  - Mean Absolute Percentage Error (MAPE)
  - R² Score

---

## 📊 Workflow

1. Load dataset from Excel file  
2. Perform EDA and visualize correlations  
3. Handle missing values  
4. Encode categorical features using One-Hot Encoding  
5. Split data into training and validation sets  
6. Train multiple regression models  
7. Evaluate and compare model performance  

---

## 🧠 Models Used

### 1. Linear Regression
- Simple baseline model
- Assumes linear relationship between features and target

### 2. Random Forest Regressor
- Ensemble learning method
- Captures non-linear relationships effectively

### 3. Support Vector Regressor (SVR)
- Works well for high-dimensional data
- Requires feature scaling

---

## 📈 Evaluation Metrics

- **MAE (Mean Absolute Error)** – Measures average prediction error  
- **MAPE (Mean Absolute Percentage Error)** – Relative error percentage  
- **R² Score** – Measures goodness of fit  

---

## 📌 Results

The models were compared using MAPE:

| Model                | Performance |
|---------------------|------------|
| Linear Regression   | 0.1963 |
| Random Forest       | 0.2003 |
| SVR                 | 0.1990 |

> Random Forest generally performs better due to its ability to capture non-linear patterns.

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  

---

## ▶️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
