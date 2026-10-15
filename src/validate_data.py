import json
from pathlib import Path

import pandas as pd
import yaml

DATA_PATH = "data/raw/churn.csv"
SCHEMA_PATH = "configs/data_schema.yaml"
REPORT_PATH = "reports/data_validation_report.json"


def load_schema(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_data(df: pd.DataFrame, schema: dict) -> dict:
    errors = []
    warnings = []

    required_columns = schema["required_columns"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append({"type": "missing_columns", "columns": missing_columns})

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        warnings.append({"type": "duplicates", "count": duplicate_count})

    normalized_df = df.replace(r"^\s*$", pd.NA, regex=True)
    missing_share = normalized_df.isna().mean().to_dict()
    too_many_missing = {
        col: share for col, share in missing_share.items()
        if share > schema.get("max_missing_share", 0.2)
    }
    if too_many_missing:
        errors.append({"type": "too_many_missing_values", "columns": too_many_missing})

    target = schema["target_column"]
    if target in df.columns:
        allowed = set(schema["allowed_target_values"])
        actual = set(df[target].dropna().unique())
        unexpected = sorted(actual - allowed)
        if unexpected:
            errors.append({"type": "unexpected_target_values", "values": unexpected})

    for col in schema.get("numeric_columns", []):
        if col in df.columns:
            original = df[col].replace(r"^\s*$", pd.NA, regex=True)
            converted = pd.to_numeric(original, errors="coerce")
            invalid_count = int(converted.isna().sum() - original.isna().sum())
            if invalid_count > 0:
                errors.append({"type": "invalid_numeric_values", "column": col, "count": invalid_count})

    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "duplicate_count": duplicate_count,
        "errors": errors,
        "warnings": warnings,
        "status": "passed" if not errors else "failed",
    }


def main():
    df = pd.read_csv(DATA_PATH)
    schema = load_schema(SCHEMA_PATH)
    report = validate_data(df, schema)

    Path("reports").mkdir(exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
