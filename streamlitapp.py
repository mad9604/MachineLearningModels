import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Mobile Price Classification", layout="wide")
st.title("📱 Mobile Price Range Classification")
st.write("Comparison of multiple ML classification models")

# -----------------------------
# Upload Dataset
# -----------------------------
data_file = st.file_uploader(
    "Upload Mobile Price Dataset (with price_range)",
    type=["csv", "xlsx"]
)

if data_file is None:
    st.warning("Please upload the dataset.")
    st.stop()

# -----------------------------
# Load Data
# -----------------------------
if data_file.name.endswith(".csv"):
    df = pd.read_csv(data_file)
else:
    df = pd.read_excel(data_file)

st.success("Dataset loaded successfully!")
st.dataframe(df.head(), use_container_width=True)

# -----------------------------
# Feature & Target Split
# -----------------------------
if "price_range" not in df.columns:
    st.error("Dataset must contain 'price_range' column.")
    st.stop()

X = df.drop("price_range", axis=1)
y = df["price_range"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

if xgb_available:
    models["XGBoost"] = XGBClassifier(
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42
    )

# -----------------------------
# Model Evaluation
# -----------------------------
results = []

st.subheader("📊 Model Evaluation Metrics")

selected_model_name = st.selectbox(
    "Select a model to view Confusion Matrix & Classification Report",
    list(models.keys())
)

for name, model in models.items():

    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append([
        name,
        round(acc, 4),
        round(auc, 4),
        round(prec, 4),
        round(rec, 4),
        round(f1, 4),
        round(mcc, 4)
    ])

    # Show Confusion Matrix & Classification Report
    if name == selected_model_name:
        st.subheader(f"🔍 {name} – Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader(f"📄 {name} – Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

# -----------------------------
# Results Table
# -----------------------------
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ]
)

st.subheader("📈 Comparison of All Models")
st.dataframe(results_df, use_container_width=True)