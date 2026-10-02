
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from predict import predict_churn

DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
RESULTS_PATH = BASE_DIR / "models" / "model_comparison.csv"

# Page configuration
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )
    data = data.dropna(subset=["TotalCharges"])
    data = data.drop_duplicates()
    return data

@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)

df = load_data()
results = load_results()

# Sidebar navigation
st.sidebar.title("📊 Churn Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Customer Analysis",
        "Model Performance",
        "Predict Churn"
    ]
)

# Main heading
st.title("Customer Churn Prediction")
st.caption(
    "Analyze customer behavior and explore "
    "machine learning churn predictions."
)

# --------------------------------------------------
# PAGE 1: OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.header("Business Overview")

    total_customers = len(df)
    churned = (df["Churn"] == "Yes").sum()
    retained = (df["Churn"] == "No").sum()
    churn_rate = churned / total_customers * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churned Customers",
        f"{churned:,}"
    )

    col3.metric(
        "Retained Customers",
        f"{retained:,}"
    )

    col4.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")

        churn_counts = df["Churn"].value_counts()

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.pie(
            churn_counts,
            labels=churn_counts.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Customer Churn Percentage")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Churn by Contract")

        contract_data = pd.crosstab(
            df["Contract"],
            df["Churn"],
            normalize="index"
        ) * 100

        st.bar_chart(contract_data)

    st.subheader("Customer Dataset")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

# --------------------------------------------------
# PAGE 2: CUSTOMER ANALYSIS
# --------------------------------------------------

elif page == "Customer Analysis":

    st.header("Customer Behavior Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Charges by Churn")

        fig, ax = plt.subplots(figsize=(7, 4))

        df.boxplot(
            column="MonthlyCharges",
            by="Churn",
            ax=ax
        )

        ax.set_title("Monthly Charges vs Churn")
        ax.set_xlabel("Churn")
        ax.set_ylabel("Monthly Charges")

        fig.suptitle("")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Tenure by Churn")

        fig, ax = plt.subplots(figsize=(7, 4))

        for label, group in df.groupby("Churn"):
            ax.hist(
                group["tenure"],
                bins=20,
                alpha=0.6,
                label=label
            )

        ax.set_title("Customer Tenure Distribution")
        ax.set_xlabel("Tenure (Months)")
        ax.set_ylabel("Number of Customers")
        ax.legend(title="Churn")

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Churn Rate by Contract")

    contract_churn = (
        df.assign(
            ChurnFlag=(df["Churn"] == "Yes").astype(int)
        )
        .groupby("Contract")["ChurnFlag"]
        .mean() * 100
    )

    st.bar_chart(contract_churn)

    st.write(
        "These charts show associations in the historical "
        "customer data. They do not establish causation."
    )

# --------------------------------------------------
# PAGE 3: MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("Machine Learning Model Performance")

    st.dataframe(
        results.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}",
            "ROC-AUC": "{:.2%}"
        }),
        use_container_width=True
    )

    st.subheader("Model Comparison")

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]

    chart_data = results.set_index("Model")[metrics]

    st.bar_chart(chart_data)

    best_model = results.sort_values(
        by="F1 Score",
        ascending=False
    ).iloc[0]

    st.info(
        f"Selected model: {best_model['Model']} "
        f"(F1 Score: {best_model['F1 Score']:.2%})"
    )

# --------------------------------------------------
# PAGE 4: PREDICT CHURN
# --------------------------------------------------

elif page == "Predict Churn":

    st.header("Individual Customer Churn Prediction")

    st.write(
        "Enter customer details to generate a prediction "
        "using your trained machine learning model."
    )

    with st.form("customer_form"):

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1]
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            tenure = st.slider(
                "Tenure (Months)",
                0, 72, 12
            )

            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"]
            )

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

        with col2:
            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=200.0,
                value=70.0
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=840.0
            )

        submitted = st.form_submit_button(
            "Predict Customer Churn",
            use_container_width=True
        )

    if submitted:

        customer = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_charges
        }

        try:
            result = predict_churn(customer)

            st.divider()
            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            col1.metric(
                "Churn Probability",
                f"{result['probability']:.2%}"
            )

            col2.metric(
                "Risk Category",
                result["risk"]
            )

            if result["prediction"] == 1:
                st.warning(
                    "The model predicts that this "
                    "customer may churn."
                )
            else:
                st.success(
                    "The model predicts that this "
                    "customer may stay."
                )

            st.caption(
                "This is a model estimate, not a guarantee "
                "of future customer behavior."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")