import pandas as pd
import numpy as np


def _print_missing_values(df: pd.DataFrame) -> None:
    # Datenbereinigung: Überprüfung auf fehlende Werte
    print("\n Fehlende Werte pro Spalte:")
    print(df.isna().sum())


def _print_binary_feature_validation(df: pd.DataFrame) -> None:
    # Datenbereinigung: Validierung binärer Merkmale (0/1)
    print("\n", df["Fraud_Label"].value_counts(dropna=False))
    print("\n", df["Is_Weekend"].value_counts(dropna=False))
    print("\n", df["Previous_Fraudulent_Activity"].value_counts(dropna=False))
    print("\n", df["IP_Address_Flag"].value_counts(dropna=False))


def _convert_and_filter_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    # Datenformatierung: Datumsumwandlung (Timestamp)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)

    # Datenbereinigung: Entferne Zeilen mit ungültigem Timestamp (Datenqualität)
    before_rows = len(df)
    df = df[df["Timestamp"].notna()].copy()
    after_rows = len(df)
    print(f"\n Entfernte Zeilen wegen ungültigem Timestamp: {before_rows - after_rows}")
    return df


def _deduplicate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    # Datenbereinigung: Deduplication - Transaction_ID eindeutig halten (falls doppelt)
    df.sort_values(by=["Transaction_ID", "Timestamp"], inplace=True)
    df.drop_duplicates(subset=["Transaction_ID"], keep="last", inplace=True)
    print("\n Nach Deduplication:", df.shape)
    return df


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
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
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Datenbereinigung: Prüfen, ob numerische Werte nach der Formatierung NaNs enthalten
    num_nan = df.select_dtypes(include=["number"]).isna().sum()

    print("\n Numerische Spalten mit NaNs:")
    print(num_nan[num_nan > 0])

    return df


def _split_rejects_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Datenqualitätsprüfung → Rejects (Nur wirklich unbrauchbare Daten)
    reject_mask = (
            df["Transaction_ID"].isna() |
            df["User_ID"].isna() |
            df["Timestamp"].isna() |
            df["Transaction_Amount"].isna() |
            df["Fraud_Label"].isna() |
            (df["Transaction_Amount"] <= 0) |
            (df["Daily_Transaction_Count"] < 0) |
            (df["Failed_Transaction_Count_7d"] < 0) |
            (df["Avg_Transaction_Amount_7d"] < 0) |
            (df["Card_Age"] < 0) |
            (df["Transaction_Distance"] < 0)
    )
    Rejects = df[reject_mask].copy()
    # Speicherung bereinigter Teil in Clean_data für weitere Arbeit
    Clean_data = df[~reject_mask].copy()
    print(f"\n Reject Mask – Clean: {len(Clean_data)} | Rejects: {len(Rejects)}")
    return Clean_data, Rejects


def _print_empty_string_checks(Clean_data: pd.DataFrame) -> None:
    # Datenbereinigung: Prüfung auf leere oder whitespace-only Strings in Textspalten
    obj_cols = Clean_data.select_dtypes(include=["object", "string"]).columns
    empty_counts = (
        Clean_data[obj_cols]
        .apply(lambda col: col.astype(str).str.strip().eq(""))
        .sum()
    )
    print("\n Leere Strings (nicht NaN):")
    print(empty_counts[empty_counts > 0])


def _normalize_categorical_columns(Clean_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    # Alle Objects, bzw. kategorische vars in lower case
    obj_cols = Clean_data.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        Clean_data[col] = (
            Clean_data[col]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    print("\n Kategoriale Spalten normalisiert:")
    print(obj_cols.tolist())
    return Clean_data, obj_cols


def _feature_engineering(Clean_data: pd.DataFrame) -> pd.DataFrame:
    # Datenanreicherung: daytime extrahieren aus timestemp
    Clean_data["Hour"] = Clean_data["Timestamp"].dt.hour

    # Datenanreicherung + Normierung: Neues Feature -> Transactionsbetrag zu Gesamtaccountbalance Ratio
    if "Account_Balance" in Clean_data.columns:
        Clean_data["Amount_to_Balance_Ratio"] = (
                Clean_data["Transaction_Amount"] /
                Clean_data["Account_Balance"].replace(0, np.nan)
        ).fillna(0)

    # Löschen von Timestamp, da Aufteilung in hour und isWeekend -> Relevanter für weiteren Prozess
    Clean_data.drop(columns=["Timestamp"], inplace=True)

    print("\n", Clean_data.info())
    return Clean_data


def _user_aggregation(Clean_data: pd.DataFrame) -> pd.DataFrame:
    # Datenaggregation + Datenanreicherung: Wichtige Userdaten -> Eventuell wichtig für Analysen
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
    return user_aggregation


def _reorder_columns(Clean_data: pd.DataFrame) -> pd.DataFrame:
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

    return Clean_data

# Main-Funktion für Transform-Prozess
def transform_transactions(df: pd.DataFrame):
    df = df.copy()

    _print_missing_values(df)
    _print_binary_feature_validation(df)

    df = _convert_and_filter_timestamp(df)

    df = _deduplicate_transactions(df)

    df = _convert_numeric_columns(df)

    Clean_data, Rejects = _split_rejects_and_clean(df)

    _print_empty_string_checks(Clean_data)

    Clean_data, _obj_cols = _normalize_categorical_columns(Clean_data)

    Clean_data = _feature_engineering(Clean_data)

    _ = _user_aggregation(Clean_data)

    Clean_data = _reorder_columns(Clean_data)

    return Clean_data, Rejects

