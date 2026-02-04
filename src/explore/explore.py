from pathlib import Path

import pandas as pd
import plotly.express as px

from src.config import CLEAN_TRANSACTIONS_PATH

# Einlesen des cleaned Datasets
ROOT = Path(__file__).resolve().parents[2]  # dm-project
# df = pd.read_csv(ROOT / "data" / "cleaned" / "clean_transactions.csv")
# df["Fraud_Label_str"] = df["Fraud_Label"].map({0: "Non-Fraud", 1: "Fraud"})

def explore(path: Path = CLEAN_TRANSACTIONS_PATH):
    df = pd.read_csv(path)
    df["Fraud_Label_str"] = df["Fraud_Label"].map({0: "Non-Fraud", 1: "Fraud"})
    # Imbalance zwischen Fraud und non-Fraud
    fig = px.histogram(
        df,
        x="Fraud_Label_str",
        title="Class Imbalance: Fraud vs. Non-Fraud"
    )
    fig.show()

    # Fraud-Rate nach Location
    if "Location" in df.columns:
        loc = (
            df.groupby("Location", as_index=False)
            .agg(Transactions=("Transaction_ID", "count"),
                 Fraud_Rate=("Fraud_Label", "mean"))
        )

        # Nur Locations mit genügend Daten, sonst sind Raten instabil
        loc = loc[loc["Transactions"] >= 100].sort_values("Fraud_Rate", ascending=False).head(15)

        fig = px.bar(
            loc, x="Fraud_Rate", y="Location",
            orientation="h",
            title="Top Locations nach Fraud-Rate (min. 100 Transaktionen)"
        )
        fig.show()

    # Risk Score nach Fraud
    fig = px.violin(
        df,
        x="Fraud_Label_str",
        y="Risk_Score",
        box=True,
        title="Risk Score Distribution: Fraud vs. Non-Fraud"
    )
    fig.show()

    # Amount to Balance Ratio nach Fraud
    fig = px.box(
        df,
        x="Fraud_Label_str",
        y="Amount_to_Balance_Ratio",
        title="Amount / Balance Ratio: Fraud vs. Non-Fraud"
    )
    fig.show()

    # Vorangegangene betrügerische Aktivitäten nach Fraud
    prev_fraud_agg = (
        df.groupby("Previous_Fraudulent_Activity")["Fraud_Label"]
        .mean()
        .reset_index()
    )

    prev_fraud_agg["Previous_Fraudulent_Activity"] = (
        prev_fraud_agg["Previous_Fraudulent_Activity"]
        .map({0: "No Previous Fraud", 1: "Previous Fraud"})
    )

    fig = px.bar(
        prev_fraud_agg,
        x="Previous_Fraudulent_Activity",
        y="Fraud_Label",
        text="Fraud_Label",
        title="Fraud Rate by Previous Fraudulent Activity"
    )
    fig.update_yaxes(title="Fraud Rate")
    fig.show()

    # Failed Transactions nach Fraud
    fig = px.box(
        df,
        x="Fraud_Label_str",
        y="Failed_Transaction_Count_7d",
        title="Failed Transactions (7d): Fraud vs Non-Fraud"
    )
    fig.show()

    # IP Address Flag nach Fraud
    ip_agg = (
        df.groupby("IP_Address_Flag")["Fraud_Label"]
        .mean()
        .reset_index()
    )

    ip_agg["IP_Address_Flag"] = ip_agg["IP_Address_Flag"].map(
        {0: "Normal IP", 1: "Suspicious IP"}
    )

    fig = px.bar(
        ip_agg,
        x="IP_Address_Flag",
        y="Fraud_Label",
        text="Fraud_Label",
        title="Fraud Rate by IP Address Flag"
    )
    fig.update_yaxes(title="Fraud Rate")
    fig.show()

    # Card Type nach Fraud
    card_agg = (
        df.groupby(["Card_Type", "Fraud_Label_str"])
        .size()
        .reset_index(name="Count")
    )

    fig = px.bar(
        card_agg,
        x="Card_Type",
        y="Count",
        color="Fraud_Label_str",
        barmode="group",
        title="Card Type vs Fraud / Non-Fraud"
    )
    fig.show()










