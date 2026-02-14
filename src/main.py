from src.etl.extract import extract_transactions
from src.etl.transform import transform_transactions
from src.etl.load import load_data
from src.graph.setup import import_transactions_to_neo4j
from src.graph.report import run_demo
from src.explore.explore import explore
from src.randomforest.model_random_forest import random_forest
from src.config import OUTPUT_DIR

def run_etl():
    raw_df = extract_transactions()
    clean_df, rejects_df, users_df = transform_transactions(raw_df)
    load_data(clean_df, rejects_df, users_df, OUTPUT_DIR)

def run_neo4j():
    try:
        import_transactions_to_neo4j()
        run_demo()
    except Exception as e:
        print("Neo4j nicht verfügbar – überspringe Graph-Teil.")
        print(e)

def run_explore():
    explore()

def run_random_forest():
    random_forest()

if __name__ == "__main__":
    run_etl()
    run_neo4j()
    run_explore()
    run_random_forest()
