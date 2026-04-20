# HeartDiseasePredictor
A Streamlit-based machine learning web application that predicts the likelihood of heart disease in a patient using a trained K-Nearest Neighbors (KNN) classifier based on clinical data.
# 🫀 Heart Disease Prediction Web App

🔗 Live Demo: https://heartdiseasepredictor-qqdsfqxzgx4uwpbjfiqucy.streamlit.app

An end-to-end Machine Learning web application designed to predict the likelihood of heart disease in a patient based on clinical and physical metrics. Built with **Python**, **Scikit-Learn**, and **Streamlit**, this project bridges the gap between data science and an accessible user interface.

---

## 📂 Repository Structure & File Breakdown

This repository contains all the necessary artifacts, from raw data and exploratory data analysis to the final deployed frontend.

### 🔹 Files Overview

* **`app.py`**
  Main execution script for the Streamlit web application. Handles UI, user inputs, preprocessing, and prediction.

* **`HeartDisease.ipynb`**
  Jupyter Notebook containing EDA, feature engineering, visualization (Seaborn, Matplotlib), and model training (Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM).

* **`heart.csv`**
  Dataset with 11 clinical features (Age, Sex, Chest Pain Type, etc.) and target variable.

* **`KNN_heart_disease.pkl`**
  Final trained KNN model used for predictions.

* **`scaler.pkl`**
  StandardScaler fitted on numeric features to normalize input data.

* **`columns.pkl`**
  Stores encoded column structure to ensure consistency during prediction.

---

## 🛠️ Technology Stack

* **Language:** Python
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn (KNN Classifier)
* **Data Handling:** Pandas, NumPy
* **Serialization:** Joblib

---

## 💻 How to Run Locally

### 📌 Prerequisites

* Python 3.8+
* Git installed

### 🚀 Setup Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

#### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

#### 4. Run the App

```bash
streamlit run app.py
```

The app will open at: **[http://localhost:8501](http://localhost:8501)**

---

## 🧠 Project Workflow

1. **User Input**
   User enters clinical data via Streamlit UI.

2. **Preprocessing**

   * Categorical features → One-hot encoded
   * Numeric features → Scaled using `scaler.pkl`
   * Column alignment ensured via `columns.pkl`

3. **Prediction**
   Processed data is passed to the trained **KNN model**.

4. **Output**
   Displays prediction:

   * `1` → Heart Disease Risk
   * `0` → Normal

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the **MIT License**.

```

## ❤️ Author


