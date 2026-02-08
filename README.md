# Customer Churn Prediction Using Artificial Neural Network (ANN)

A deep learning project that predicts customer churn in the telecom industry using an Artificial Neural Network. The model helps identify customers who are likely to leave the business, enabling proactive retention strategies.

## Overview

Customer churn prediction measures why customers are leaving a business. This project focuses on telecom customer churn and builds a deep learning model to predict churn likelihood. Model performance is evaluated using precision, recall, and F1-score metrics.

## Dataset

- **Source:** IBM Telco Customer Churn Dataset
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Records:** ~7,043 customers (7,032 after data cleaning)
- **Target:** `Churn` (Yes/No) — binary classification

### Features

| Category | Features |
|----------|----------|
| **Demographics** | gender, SeniorCitizen, Partner, Dependents |
| **Account** | tenure, Contract, PaperlessBilling, PaymentMethod |
| **Services** | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| **Charges** | MonthlyCharges, TotalCharges |

## Project Structure

```
customer churn prediction nn/
├── churn1.ipynb                          # Main Jupyter notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset
└── README.md                             # This file
```

## Data Preprocessing

1. **Drop** `customerID` column (non-predictive identifier)
2. **Handle missing values** — Remove rows with blank `TotalCharges` (11 rows with tenure=0)
3. **Convert types** — `TotalCharges` from string to numeric
4. **Encode categorical variables:**
   - Binary: Yes/No → 1/0
   - Gender: Female/Male → 1/0
   - Multi-category: One-hot encoding for `InternetService`, `Contract`, `PaymentMethod`
5. **Feature scaling** — StandardScaler on `tenure`, `MonthlyCharges`, `TotalCharges`
6. **Train/test split** — 80/20 stratified split

## Model Architecture

The model is a feedforward neural network built with TensorFlow/Keras:

| Layer | Units | Activation | Input Shape |
|-------|-------|------------|-------------|
| Dense | 26 | ReLU | (26,) |
| Dense | 15 | ReLU | — |
| Dense | 1 | Sigmoid | — |

- **Optimizer:** Adam
- **Loss:** Binary cross-entropy
- **Epochs:** 100
- **Input features:** 26 (after one-hot encoding)

## Results

### Test Set Performance

| Metric | Class 0 (No Churn) | Class 1 (Churn) |
|--------|-------------------|-----------------|
| Precision | 0.83 | 0.63 |
| Recall | 0.86 | 0.56 |
| F1-Score | 0.85 | 0.59 |

- **Accuracy:** ~78%
- **Macro Avg F1:** 0.72
- **Weighted Avg F1:** 0.77

The model performs better at identifying customers who stay (Class 0) than those who churn (Class 1), which is common in imbalanced churn datasets.

## Requirements

```text
pandas
numpy
matplotlib
scikit-learn
tensorflow
```

Install with:
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

## How to Run

1. **Clone or download** this project folder.

2. **Prepare the dataset:**
   - Ensure `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in the project directory.
   - The notebook loads `customer_churn.csv` — either rename the CSV file to `customer_churn.csv` or update the notebook path:
     ```python
     df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
     ```

3. **Open and run** the Jupyter notebook:
   ```bash
   jupyter notebook churn1.ipynb
   ```

4. **Execute all cells** sequentially to load data, preprocess, train the model, and view results.

## Key Insights

- **Tenure:** Customers with shorter tenure tend to churn more.
- **Contract:** Month-to-month contracts are associated with higher churn.
- **Monthly charges:** Higher charges correlate with increased churn risk.
- **Services:** Customers without add-ons (e.g., TechSupport, OnlineSecurity) show higher churn.

## License

This project uses the IBM Telco Customer Churn dataset, which is publicly available for analysis and educational purposes.
