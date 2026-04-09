# Hybrid LSTM-XGBoost Sales Forecasting

## 📌 Overview

This project implements a hybrid machine learning model to forecast sales using a combination of **LSTM (Long Short-Term Memory)** and **XGBoost**.

The goal is to improve prediction accuracy by leveraging:

* LSTM for capturing time-series patterns
* XGBoost for modeling residual errors

---

## 🚀 Key Features

* Multi-store time series forecasting
* Hybrid deep learning + machine learning model
* Automatic feature engineering (lags, date features)
* Residual learning with XGBoost
* Auto-generated visualizations
* Performance comparison using RMSE and MAE

---

## 🧠 Model Architecture

### 🔹 Step 1: LSTM Model

* Learns temporal dependencies in sales data
* Processes sequential input using sliding windows

### 🔹 Step 2: Residual Learning (XGBoost)

* Calculates error from LSTM predictions
* Trains XGBoost on residuals to improve accuracy

### 🔹 Final Output:

Final Prediction = LSTM Prediction + XGBoost Residual Correction

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* XGBoost
* Pandas, NumPy
* Matplotlib

---

## 📊 Results

### 📈 Generated Graphs

* Actual vs Predicted (LSTM vs Hybrid)
* Residual Error Plot
* Training Loss Curve
* Model Comparison (RMSE)
* Zoomed Forecast Visualization

> All graphs are automatically generated and saved in the `images/` folder when the code is executed.

---

## 📁 Project Structure

```
project/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── images/
│   ├── graph1_main.png
│   ├── graph2_residual.png
│   ├── graph3_loss.png
│   ├── graph4_comparison.png
│   ├── graph5_zoom.png
│
├── data/
    └── (dataset - ignored or sample)
```

---

## ▶️ How to Run

1. Clone the repository:

```
git clone https://github.com/Amaanshaikh-89/Hybrid-XGBoost-and-LSTM-model-for-multi-sales-forcasting-.git
cd your-repo-name
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the project:

```
python app.py
```

---

## 📌 Key Insights

* Hybrid models outperform standalone deep learning models
* Residual learning significantly reduces prediction error
* Time-series + ML combination provides better generalization

---

## 🚧 Future Improvements

* Deploy as a web app (Streamlit / Flask)
* Add real-time forecasting
* Hyperparameter tuning
* Use more advanced architectures (Transformer models)

---

## 👤 Author

Amaan Shaikh

---

## ⭐ If you found this useful

Give this repo a star and feel free to fork or contribute!
