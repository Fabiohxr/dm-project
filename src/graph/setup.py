import pandas as pd
from neo4j import GraphDatabase
from sklearn.preprocessing import StandardScaler

from src.config import (
    NEO4J_IMPORT_PATH,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD, CLEAN_TRANSACTIONS_PATH
)

def import_transactions_to_neo4j(
    input_csv_path: str = CLEAN_TRANSACTIONS_PATH,
    neo4j_import_path: str = NEO4J_IMPORT_PATH,
    neo4j_csv_url: str = "file:///transactions_vec.csv",
    uri: str = NEO4J_URI,
    user: str = NEO4J_USER,
    password: str = NEO4J_PASSWORD
):
    # CSV laden
    df = pd.read_csv(input_csv_path)

    # Feature-Auswahl
    vector_features = [
        "Transaction_Amount",
        "Amount_to_Balance_Ratio",
        "Risk_Score",
        "Failed_Transaction_Count_7d",
        "Transaction_Distance",
        "Card_Age",
        "Hour",
        "Is_Weekend"
    ]

    # Normieren
    X = df[vector_features].astype(float).values
    X = StandardScaler().fit_transform(X)

    # Vektor als einzelne Spalten
    for i in range(X.shape[1]):
        df[f"emb_{i}"] = X[:, i]

    # Export für Neo4j
    df.to_csv(neo4j_import_path, index=False)
    print("CSV für Neo4j exportiert")

    # Verbindung zur Neo4j-Datenbank herstellen
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        # Funktion, um die Abfrage auszuführen
        def execute_query(query, params=None):
            with driver.session() as session:
                session.run(query, params or {}).consume()

        # Vektor Index, damit kNN effizient arbeiten kann später
        def ensure_vector_index():
            q = """
            CREATE VECTOR INDEX tx_embedding_index IF NOT EXISTS
            FOR (t:Transaction) ON (t.embedding)
            OPTIONS {
              indexConfig: {
                `vector.dimensions`: 8,
                `vector.similarity_function`: 'cosine'
              }
            };
            """
            execute_query(q)
            print("✅ Vector Index angelegt/exists")

        # Check des Index
        def check_index_state():
            q = """
            SHOW INDEXES
            YIELD name, state
            WHERE name = 'tx_embedding_index'
            RETURN name, state;
            """
            with driver.session() as session:
                rows = [r.data() for r in session.run(q)]
            print("ℹ️ Index-Status:", rows)

        # Check, ob Daten schon importiert sind
        def has_data():
            q = "MATCH (t:Transaction) RETURN count(t) AS n;"
            with driver.session() as session:
                n = session.run(q).single()["n"]
            return n > 0

        # Die Cypher-Abfrage zum Laden der CSV und Erstellen der Knoten und Beziehungen
        query = f"""
        CALL (){{
          LOAD CSV WITH HEADERS FROM '{neo4j_csv_url}' AS row

          MERGE (u:User {{User_ID: row.User_ID}})
          MERGE (l:Location {{name: row.Location}})
          MERGE (d:DeviceType {{name: row.Device_Type}})
          MERGE (m:MerchantCategory {{name: row.Merchant_Category}})
          MERGE (c:CardType {{name: row.Card_Type}})
          MERGE (a:AuthMethod {{name: row.Authentication_Method}})

          MERGE (t:Transaction {{Transaction_ID: row.Transaction_ID}})
          SET
            t.Transaction_Amount = toFloat(row.Transaction_Amount),
            t.Amount_to_Balance_Ratio = toFloat(row.Amount_to_Balance_Ratio),
            t.Risk_Score = toFloat(row.Risk_Score),
            t.Failed_Transaction_Count_7d = toInteger(row.Failed_Transaction_Count_7d),
            t.Transaction_Distance = toFloat(row.Transaction_Distance),
            t.Card_Age = toFloat(row.Card_Age),
            t.Hour = toInteger(row.Hour),
            t.Is_Weekend = toInteger(row.Is_Weekend),
            t.Fraud_Label = toInteger(row.Fraud_Label),
            t.embedding = [
              toFloat(row.emb_0), toFloat(row.emb_1), toFloat(row.emb_2), toFloat(row.emb_3),
              toFloat(row.emb_4), toFloat(row.emb_5), toFloat(row.emb_6), toFloat(row.emb_7)
            ]

          MERGE (u)-[:MADE]->(t)
          MERGE (t)-[:AT]->(l)
          MERGE (t)-[:USING_DEVICE]->(d)
          MERGE (t)-[:IN_CATEGORY]->(m)
          MERGE (t)-[:PAID_WITH]->(c)
          MERGE (t)-[:AUTHED_VIA]->(a)

          RETURN 1 AS ok
        }}
        IN TRANSACTIONS OF 1000 ROWS
        RETURN count(ok) AS batches;
        """

        # Bestätigen, dass die Daten geladen wurden
        def check_data():
            query_check = """
            MATCH (u:User)-[:MADE]->(t:Transaction)
            RETURN
              u.User_ID            AS User,
              t.Transaction_ID     AS Transaction,
              t.Transaction_Amount AS Amount,
              t.Risk_Score         AS RiskScore,
              t.Fraud_Label        AS FraudLabel
            LIMIT 5;
            """
            with driver.session() as session:
                result = session.run(query_check)
                for record in result:
                    print(record["User"], record["Transaction"], record["Amount"], record["RiskScore"], record["FraudLabel"])

        # Lade nur einmal. Falls Daten schon importiert wurden kein neuer Import.
        if not has_data():
            execute_query(query)
            ensure_vector_index()
            check_index_state()
            check_data()
        else:
            print("Daten sind schon importiert – überspringe Import.")

    finally:
        driver.close()

