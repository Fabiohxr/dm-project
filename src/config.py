from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "transactions.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "cleaned"

CLEAN_TRANSACTIONS_PATH = OUTPUT_DIR / "clean_transactions.csv"
REJECTS_PATH = OUTPUT_DIR / "rejects.csv"
USER_AGG_PATH = OUTPUT_DIR / "user_aggregation.csv"

#Neo4j
NEO4J_IMPORT_PATH = r"/Users/fabioheuser/Library/Application Support/neo4j-desktop/Application/Data/dbmss/dbms-a1255a9a-1d62-48e4-995b-09bd77095057/import/transactions_vec.csv"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "babo2003"