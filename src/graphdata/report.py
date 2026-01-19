from neo4j import GraphDatabase

# Verbindung zur Neo4j-Datenbank herstellen
uri = "neo4j://127.0.0.1:7687"
user = "neo4j"
password = "babo2003"

# Erstelle eine Neo4j-Session
driver = GraphDatabase.driver(uri, auth=(user, password))

# Zufällige Transaktions ID
def pick_any_txid():
    q = "MATCH (t:Transaction) RETURN t.Transaction_ID AS tx LIMIT 1;"
    with driver.session() as session:
        row = session.run(q).single()
    return row["tx"] if row else None

# k-nearest-neighbour mit TransactionID
def knn_by_txid(txid, k=10):
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
        rows = [r.data() for r in session.run(q, {"txid": txid, "k": k})]
    return rows

# Abfragen
# Aehnliche Transaktionen mit Fraud_Label positiv
def similar_frauds(txid, k=50, top=10):
    q = """
    MATCH (t:Transaction {Transaction_ID: $txid})
    WITH t
    CALL db.index.vector.queryNodes('tx_embedding_index', $k, t.embedding)
    YIELD node, score
    WHERE node.Transaction_ID <> t.Transaction_ID AND node.Fraud_Label = 1
    RETURN node.Transaction_ID AS tx, score
    ORDER BY score DESC
    LIMIT $top
    """
    with driver.session() as session:
        rows = [r.data() for r in session.run(q, {"txid": txid, "k": k, "top": top})]
    return rows

#Random Transaktionen (für Demo)
txid = pick_any_txid()
print("Nutze TX:", txid)

neighbors = knn_by_txid(txid, k=10)
for n in neighbors:
    print(n["tx"], n["amount"], n["risk"], n["fraud"], n["score"])

# Anzahl unterschiedlicher Werte je Label (Kontrolle)
def counts_by_label():
    q = """
    MATCH (n)
    RETURN labels(n)[0] AS label, count(*) AS n
    ORDER BY n DESC
    """
    with driver.session() as session:
        return [r.data() for r in session.run(q)]

# User mit der höchsten Fraud-Rate
def top_users_by_fraud(limit=20, min_tx=10):
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

# Fraud-Rate nach Device
def fraud_rate_by_device(limit=20):
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

# Top Transaktionen mit den Meisten fehlgeschlagenen Transaktionen der vergangenen 7 Tage
def top_failed_transactions(limit=50):
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

#Abbildung in Console
print("\nLabel Counts:", counts_by_label())
print("\nTop Users:", top_users_by_fraud(10)[:5])
print("\nFraud by Device:", fraud_rate_by_device(10)[:5])
print("\nTop Failed Transactions:")
for r in top_failed_transactions(15):
    print(r)

driver.close()
