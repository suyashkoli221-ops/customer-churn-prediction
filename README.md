# Customer Churn Prediction and Analysis

A machine learning project that analyzes customer behavior and predicts customer churn using Python. The project includes exploratory data analysis, machine learning model comparison, and an interactive Streamlit dashboard for customer churn predictions.

## Project Overview

Customer churn occurs when customers stop using a company's products or services. This project uses historical customer data to identify churn patterns and build a machine learning model that estimates the likelihood of a customer leaving.

## Features

* Customer data cleaning and preprocessing.
* Exploratory Data Analysis (EDA).
* Customer churn distribution and visualization.
* Analysis of customer tenure, monthly charges, and contract types.
* Comparison of Logistic Regression, Decision Tree, and Random Forest.
* Customer churn prediction using a trained Decision Tree model.
* Interactive Streamlit dashboard.
* Individual customer churn probability and risk category.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Joblib
* Jupyter Notebook

## Dataset

**Dataset:** IBM Telco Customer Churn

* Original records: 7,043
* Features: 20 customer attributes and 1 target column
* Target variable: `Churn`
* Cleaned records used: 7,032

The dataset contains information about customer demographics, subscriptions, services, billing, and churn status.

## Machine Learning Models

Three classification algorithms were trained and evaluated:

* Logistic Regression
* Decision Tree
* Random Forest

### Model Comparison

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   72.57% |    49.01% | 79.68% |   60.69% |  83.51% |
| Decision Tree       |   73.35% |    49.92% | 79.95% |   61.46% |  81.82% |
| Random Forest       |   75.48% |    52.82% | 72.73% |   61.19% |  82.63% |

The Decision Tree was selected by the training script based on its F1 score.

## Dashboard

The Streamlit dashboard contains four sections:

1. **Overview:** Customer statistics, churn rate, and churn distribution.
2. **Customer Analysis:** Visualizations of customer tenure, monthly charges, and contract types.
3. **Model Performance:** Comparison of the three trained classification models.
4. **Predict Churn:** An interactive form for estimating churn probability for an individual customer.

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── customer_churn.csv
│
├── models/
│   ├── churn_model.pkl
│   └── model_comparison.csv
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── src/
│   ├── train_model.py
│   ├── predict.py
│   └── model_evaluation.py
│
├── dashboard/
│   └── app.py
│
├── visualizations/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows:**

```powershell
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the dashboard

```bash
streamlit run dashboard/app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## Run Model Training

To retrain and compare the models:

```bash
python src/train_model.py
```

## Run Individual Prediction

```bash
python src/predict.py
```

## Example Prediction

A sample customer test produced the following result:

* Churn probability: 5.13%
* Risk category: Low
* Prediction: Customer is predicted to stay.

The probability and risk category are model estimates, not guarantees of future behavior.

## Future Improvements

* Hyperparameter tuning and cross-validation.
* More detailed model evaluation and threshold optimization.
* Feature importance analysis.
* Improved dashboard design.
* Deployment of the Streamlit application.
* Automated testing and input validation.

## Author

**Suyash Koli**

Aspiring Data Scientist | Python | Machine Learning | Data Analysis
