# main.py
from etl.extract import extract_transactions
from etl.transform import transform_transactions
from etl.load import load_data
from graph.setup import import_transactions_to_neo4j
from graph.report import run_demo
from explore.explore import explore
from randomforest.model_random_forest import random_forest
from config import OUTPUT_DIR

def run_etl():
    raw_df = extract_transactions()
    clean_df, rejects_df = transform_transactions(raw_df)
    load_data(clean_df, rejects_df, OUTPUT_DIR)

def run_neo4j():
    import_transactions_to_neo4j()
    run_demo()

def run_explore():
    explore()

def run_random_forest():
    random_forest()

if __name__ == "__main__":
    run_etl()
    run_neo4j()
    run_explore()
    run_random_forest()
