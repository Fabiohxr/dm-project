from pathlib import Path

def load_data(clean, rejects, users, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean.to_csv(output_dir / "clean_transactions.csv", index=False)
    rejects.to_csv(output_dir / "rejects.csv", index=False)
    users.to_csv(output_dir / "user_aggregation.csv", index=False)