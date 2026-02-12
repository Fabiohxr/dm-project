# Transaction Fraud Detection Pipeline

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate plotlyproj
```


## Configuration

Edit the placeholders in:

```
src/config.py
```

```python
NEO4J_PASSWORD = "<your_neo4j_password>"
NEO4J_IMPORT_PATH = "<path_to_neo4j_import>"
```

If you do not use Neo4j, you can leave these as placeholders.

## Run the Pipeline

From the project root:

```bash
python -m src.main
```

This will:

1. Run the ETL process  
2. Perform exploratory analysis  
3. Train the Random Forest model  

## Neo4j (Optional)

Neo4j is **not required** to run the project.

If Neo4j:
- is not installed
- is not running
- has incorrect credentials

the pipeline will:
- skip the graph step
- still execute ETL, analysis, and the Random Forest model

## Output

After running the pipeline:

```
data/cleaned/
├── clean_transactions.csv
├── rejects.csv
└── user_aggregation.csv
```




