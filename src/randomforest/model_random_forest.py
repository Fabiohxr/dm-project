from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import CLEAN_TRANSACTIONS_PATH


TARGET = "Fraud_Label"

# Benutzte Features
FEATURE_COLS = [
    "Transaction_Amount",
    "Amount_to_Balance_Ratio",
    "Transaction_Type",
    "Hour",
    "Is_Weekend",
    "Device_Type",
    "Location",
    "IP_Address_Flag",
    "Previous_Fraudulent_Activity",
    "Failed_Transaction_Count_7d",
    "Card_Type",
    "Card_Age",
    "Transaction_Distance",
    "Authentication_Method",
]

# Feature-Typen
CATEGORICAL_FEATURES = [
    "Transaction_Type",
    "Device_Type",
    "Location",
    "Card_Type",
    "Authentication_Method",
]

NUMERIC_FEATURES = [
    "Transaction_Amount",
    "Amount_to_Balance_Ratio",
    "Hour",
    "Is_Weekend",
    "IP_Address_Flag",
    "Previous_Fraudulent_Activity",
    "Failed_Transaction_Count_7d",
    "Card_Age",
    "Transaction_Distance",
]


def random_forest(path: Path = CLEAN_TRANSACTIONS_PATH):

    df = pd.read_csv(path)

    X = df[FEATURE_COLS].copy()
    y = df[TARGET].astype(int)

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        min_samples_split=10,
        min_samples_leaf=5,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("rf", rf),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    y

