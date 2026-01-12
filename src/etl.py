import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ============================================================
# EXTRACT: Einlesen der Fraud-Transaktionen aus der CSV-Datei
# ============================================================

fraud_data = pd.read_csv("../data/transactions.csv", sep=",", decimal=".")
print("CSV geladen - Vorschau:")
print(fraud_data.head())
print("\n Info:")
print(fraud_data.info())

# ============================================================
# TRANSFORM
# ============================================================

# Datenbereinigung: Überprüfung auf fehlende Werte
print("\n Fehlende Werte pro Spalte:")
print(fraud_data.isna().sum())

# Datenbereinigung: Validierung binärer Merkmale (0/1)
print("\n", fraud_data["Fraud_Label"].value_counts(dropna=False))
print("\n", fraud_data["Is_Weekend"].value_counts(dropna=False))
print("\n", fraud_data["Previous_Fraudulent_Activity"].value_counts(dropna=False))
print("\n", fraud_data["IP_Address_Flag"].value_counts(dropna=False))

# Datenformatierung: Datumsumwandlung (Timestamp)
fraud_data["Timestamp"] = pd.to_datetime(fraud_data["Timestamp"], errors="coerce", utc=True)

# Datenbereinigung: Entferne Zeilen mit ungültigem Timestamp (Datenqualität)
before_rows = len(fraud_data)
fraud_data = fraud_data[fraud_data["Timestamp"].notna()].copy()
after_rows = len(fraud_data)
print(f"\n Entfernte Zeilen wegen ungültigem Timestamp: {before_rows - after_rows}")

# Datenbereinigung: Deduplication - Transaction_ID eindeutig halten (falls doppelt)
fraud_data.sort_values(by=["Transaction_ID", "Timestamp"], inplace=True)
fraud_data.drop_duplicates(subset=["Transaction_ID"], keep="last", inplace=True)
print("\n Nach Deduplication:", fraud_data.shape)

# Datenformatierung: Numerische Spalten konvertieren
numeric_cols = [
    "Transaction_Amount",
    "Account_Balance",
    "Daily_Transaction_Count",
    "Avg_Transaction_Amount_7d",
    "Failed_Transaction_Count_7d",
    "Card_Age",
    "Transaction_Distance",
    "Risk_Score"
]

for col in numeric_cols:
    if col in fraud_data.columns:
        fraud_data[col] = pd.to_numeric(fraud_data[col], errors="coerce")

# Datenbereinigung: Prüfung ob numerische Werte nach der Formatierung NaNs enthalten
num_nan = fraud_data.select_dtypes(include=["number"]).isna().sum()

print("\n Numerische Spalten mit NaNs:")
print(num_nan[num_nan > 0])

# Datenqualitätsprüfung → Rejects
reject_mask = (
    fraud_data["Transaction_ID"].isna() |
    fraud_data["User_ID"].isna() |
    fraud_data["Timestamp"].isna() |
    fraud_data["Transaction_Amount"].isna() |
    (fraud_data["Transaction_Amount"] < 0) |
    fraud_data["Fraud_Label"].isna()
)
Rejects = fraud_data[reject_mask].copy()
# Speicherung ohne Rejects in Clean_data für weitere Arbeit
Clean_data = fraud_data[~reject_mask].copy()
print(f"\n TRANSFORM – Clean: {len(Clean_data)} | Rejects: {len(Rejects)}")

# Datenbereinigung: Prüfung auf leere oder whitespace-only Strings in Textspalten
obj_cols = Clean_data.select_dtypes(include=["object"]).columns
empty_counts = (
    Clean_data[obj_cols]
    .apply(lambda col: col.astype(str).str.strip().eq(""))
    .sum()
)
print("\n Leere Strings (nicht NaN):")
print(empty_counts[empty_counts > 0])

# ------ Feature Engineering -------

# Datenanreicherung: daytime extrahieren aus timestemp
Clean_data["Hour"] = Clean_data["Timestamp"].dt.hour

# Datenanreicherung + Normierung: Neues Feature -> Transactionsbetrag zu Gesamtaccountbalance Ratio
if "Account_Balance" in Clean_data.columns:
    Clean_data["Amount_to_Balance_Ratio"] = (
        Clean_data["Transaction_Amount"] /
        Clean_data["Account_Balance"].replace(0, np.nan)
    ).fillna(0)

# Löschen von Timestamp, da Aufteilung in hour und isWeekend --> Relevanter für weiteren Prozess
Clean_data.drop(columns=["Timestamp"], inplace=True)

print("\n", Clean_data.info())

# Datenaggregation + Datenanreicherung: Wichtige Userdaten --> Eventuell wichtig für Analysen
user_aggregation = (
    Clean_data
    .groupby("User_ID")
    .agg(
        Total_Transactions=("Transaction_ID", "count"),
        Avg_Transaction_Amount=("Transaction_Amount", "mean"),
        Most_Frequent_Location=("Location", lambda x: x.value_counts().idxmax()),
        Fraud_Rate=("Fraud_Label", "mean")
    )
    .reset_index()
)

print("\n", user_aggregation.head())
print("\n", user_aggregation.info())

# Datenreihenfolge sinnvoll umändern
cols = list(Clean_data.columns)

# Zielpositionen
target_positions = {
    "Amount_to_Balance_Ratio": 5,
    "Hour": 19
}

# Spalten zuerst entfernen
for col in target_positions:
    if col in cols:
        cols.remove(col)

# Spalten in aufsteigender Positionsreihenfolge wieder einfügen
for col, pos in sorted(target_positions.items(), key=lambda x: x[1]):
    if col in Clean_data.columns:
        # Sicherheitscheck: Position darf nicht größer als Spaltenlänge sein
        insert_pos = min(pos, len(cols))
        cols.insert(insert_pos, col)

# Neue Reihenfolge anwenden
Clean_data = Clean_data[cols]

print("\n Neue Spaltenreihenfolge:")
for i, c in enumerate(Clean_data.columns):
    print(i, c)

# ============================================================
# LOAD
# ============================================================

# Sicherstellen, dass Zielverzeichnis existiert (optional, aber sauber)
output_dir = Path("../data/cleaned")
output_dir.mkdir(parents=True, exist_ok=True)

# Bereinigte Transaktionsdaten (Analyse- / Faktentabelle)
Clean_data.to_csv(
    output_dir / "clean_transactions.csv",
    index=False
)

# Abgelehnte Datensätze (Datenqualitäts-Transparenz)
Rejects.to_csv(
    output_dir / "rejects.csv",
    index=False
)

# Aggregierte Kundensicht (Dimension / Analyse)
user_aggregation.to_csv(
    output_dir / "user_aggregation.csv",
    index=False
)

print("\n LOAD abgeschlossen:")
print("- clean_transactions.csv")
print("- rejects.csv")
print("- user_aggregation.csv")

