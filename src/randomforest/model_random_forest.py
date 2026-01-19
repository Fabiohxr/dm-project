from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
ROOT = Path(__file__).resolve().parents[2]  # dm-project
TARGET = "Fraud_Label"

# Daten laden
df = pd.read_csv(ROOT / "data" / "cleaned" / "clean_transactions.csv")

# Benutzte Features
feature_cols = [
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

X = df[feature_cols].copy()
y = df[TARGET].astype(int)

# Feature-Typen
categorical_features = [
    "Transaction_Type",
    "Device_Type",
    "Location",
    "Card_Type",
    "Authentication_Method",
]

numeric_features = [
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

def random_forest():
    # OneHotEncoding für kategorische vars
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop"
    )

    # Random Forest Modell
    rf = RandomForestClassifier(
        n_estimators=500,
        min_samples_split=10,
        min_samples_leaf=5,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("rf", rf)
    ])

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Trainieren
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n === Random Forest Evaluation ===")
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n Classification Report:\n", classification_report(y_test, y_pred, digits=4))

    # Feature Importances / Gewichtung
    ohe = model.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(categorical_features)
    all_feature_names = np.concatenate([cat_names, np.array(numeric_features)])

    importances = model.named_steps["rf"].feature_importances_
    fi = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

    print("\n Top 20 Feature Importances:")
    print(fi.head(20))

    # Feature-Namen in Modell-Reihenfolge
    ohe = model.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(categorical_features)
    all_names = list(cat_names) + numeric_features

    fi = pd.Series(model.named_steps["rf"].feature_importances_, index=all_names)

    # Aggregation: Kategorien aufsummieren
    fi_agg = {}

    # kategorische Variablen: summe über alle OneHot-Spalten, die mit "Feature_" anfangen
    for f in categorical_features:
        fi_agg[f] = fi[fi.index.str.startswith(f + "_")].sum()

    # numerische Variablen: genau den Feature-Namen nehmen
    for f in numeric_features:
        fi_agg[f] = fi.get(f, 0.0)

    fi_agg = pd.Series(fi_agg).sort_values(ascending=False)

    print("\n Aggregierte Feature Importances:")
    print(fi_agg)
