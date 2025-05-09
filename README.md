# 📉 Customer Churn Prediction

This project predicts whether a telecom customer will churn or not using various machine learning models. It uses the publicly available **Telco Customer Churn** dataset and compares multiple classification models to identify the most effective one.

---

## 📁 Dataset

- **Source**: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records**: 7,043 customer profiles
- **Target Variable**: `Churn` (Yes/No)
- **Features**: Customer account information, services used, demographics, billing, and contract data.

---

## 🧠 Models Compared

| Model                | Description                              |
|---------------------|------------------------------------------|
| Logistic Regression | Simple, interpretable baseline           |
| Random Forest       | Ensemble model, handles non-linearity    |
| XGBoost             | Highly accurate, robust to overfitting   |
| K-Nearest Neighbors | Distance-based, simple to understand     |
| Support Vector Machine (SVM) | Effective with complex boundaries |

---

## 🧪 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**
- **ROC Curve & AUC**

---

## 🔍 Key Insights

- **XGBoost** outperformed other models in terms of F1 Score and AUC.
- **Contract type**, **tenure**, and **monthly charges** were among the most important features.
- Class imbalance was addressed using `class_weight='balanced'` for SVM and pipeline scaling for accuracy improvement.

---

## 🛠️ Tech Stack

- Python 🐍
- scikit-learn
- XGBoost
- Pandas, NumPy
- Matplotlib, Seaborn

---

## 📊 Feature Importance (Random Forest Example)

![Feature Importance](images/feature_importance.png)

---

## 🧾 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook or script:
   ```bash
   jupyter notebook churn_prediction.ipynb
   ```

---

## 📦 Exported Model

To deploy the model:
```python
import pickle
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)
```

---

## 👨‍💻 Author

**Meet Patel**  
Master's in Engineering & Management – Technische Hochschule Ingolstadt  
[LinkedIn](https://linkedin.com/in/your-link) | [GitHub](https://github.com/yourusername)

---

## 📌 License

MIT License
