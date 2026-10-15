import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAW_PATH = "data/raw/churn.csv"
PROCESSED_PATH = "data/processed/churn_v1.csv"
METADATA_PATH = "data/processed/churn_v1_metadata.json"
DATA_VERSION = "v1"


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    df = pd.read_csv(RAW_PATH)
    rows_before = int(len(df))
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df = df.drop_duplicates()

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    metadata = {
        "data_version": DATA_VERSION,
        "source_path": RAW_PATH,
        "output_path": PROCESSED_PATH,
        "source_sha256": file_sha256(RAW_PATH),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_before_deduplication": rows_before,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "output_sha256": file_sha256(PROCESSED_PATH),
        "description": "First validated and deduplicated version of churn dataset",
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(metadata)


if __name__ == "__main__":
    main()
