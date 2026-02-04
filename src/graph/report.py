from neo4j import GraphDatabase
from src.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD
)

# Verbindung zur Neo4j-Datenbank herstellen
uri = NEO4J_URI
user = NEO4J_USER
password = NEO4J_PASSWORD

from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


def pick_any_txid(driver):
    q = "MATCH (t:Transaction) RETURN t.Transaction_ID AS tx LIMIT 1;"
    with driver.session() as session:
        row = session.run(q).single()
    return row["tx"] if row else None


def knn_by_txid(driver, txid, k=10):
    q = """
    MATCH (t:Transaction {Transaction_ID: $txid})
    WITH t
    CALL db.index.vector.queryNodes('tx_embedding_index', $k, t.embedding)
    YIELD node, score
    WHERE node.Transaction_ID <> t.Transaction_ID
    RETURN node.Transaction_ID AS tx,
           node.Transaction_Amount AS amount,
           node.Risk_Score AS risk,
           node.Fraud_Label AS fraud,
           score
    ORDER BY score DESC
    """
    with driver.session() as session:
        return [r.data() for r in session.run(q, {"txid": txid, "k": k})]


def counts_by_label(driver):
    q = """
    MATCH (n)
    RETURN labels(n)[0] AS label, count(*) AS n
    ORDER BY n DESC
    """
    with driver.session() as session:
        return [r.data() for r in session.run(q)]


def top_users_by_fraud(driver, limit=20, min_tx=10):
    q = """
    MATCH (u:User)-[:MADE]->(t:Transaction)
    WITH u.User_ID AS user,
         count(*) AS total,
         sum(CASE WHEN t.Fraud_Label = 1 THEN 1 ELSE 0 END) AS frauds
    WHERE total >= $min_tx
    RETURN user, total, frauds, toFloat(frauds)/total AS fraud_rate
    ORDER BY frauds DESC, fraud_rate DESC
    LIMIT $limit
    """
    with driver.session() as session:
        return [r.data() for r in session.run(q, {"limit": limit, "min_tx": min_tx})]


def fraud_rate_by_device(driver, limit=20):
    q = """
    MATCH (t:Transaction)-[:USING_DEVICE]->(d:DeviceType)
    WITH d.name AS device,
         count(*) AS n,
         sum(CASE WHEN t.Fraud_Label = 1 THEN 1 ELSE 0 END) AS frauds
    RETURN device, n, frauds, toFloat(frauds)/n AS fraud_rate
    ORDER BY fraud_rate DESC, n DESC
    LIMIT $limit
    """
    with driver.session() as session:
        return [r.data() for r in session.run(q, {"limit": limit})]


def top_failed_transactions(driver, limit=50):
    q = """
    MATCH (t:Transaction)
    WHERE t.Failed_Transaction_Count_7d IS NOT NULL
    RETURN t.Transaction_ID AS tx,
           t.Failed_Transaction_Count_7d AS failed_7d,
           t.Transaction_Amount AS amount,
           t.Risk_Score AS risk,
           t.Fraud_Label AS fraud
    ORDER BY failed_7d DESC, risk DESC
    LIMIT $limit
    """
    with driver.session() as session:
        return [r.data() for r in session.run(q, {"limit": limit})]


def run_demo():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        txid = pick_any_txid(driver)
        print("Nutze TX:", txid)

        neighbors = knn_by_txid(driver, txid, k=10)
        for n in neighbors:
            print(n["tx"], n["amount"], n["risk"], n["fraud"], n["score"])

        print("\nLabel Counts:", counts_by_label(driver))
        print("\nTop Users:", top_users_by_fraud(driver, 10)[:5])
        print("\nFraud by Device:", fraud_rate_by_device(driver, 10)[:5])

        print("\nTop Failed Transactions:")
        for r in top_failed_transactions(driver, 15):
            print(r)

    finally:
        driver.close()

