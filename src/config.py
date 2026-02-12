from pathlib import Path

# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "cleaned"

DATA_PATH = DATA_DIR / "transactions.csv"

CLEAN_TRANSACTIONS_PATH = OUTPUT_DIR / "clean_transactions.csv"
REJECTS_PATH = OUTPUT_DIR / "rejects.csv"
USER_AGG_PATH = OUTPUT_DIR / "user_aggregation.csv"

# Neo4j configuration (Platzhalter ersetzen!)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "<your_neo4j_password>"

# Absolute path to Neo4j import directory
NEO4J_IMPORT_PATH = Path("<path_to_neo4j_import>/transactions_vec.csv")
