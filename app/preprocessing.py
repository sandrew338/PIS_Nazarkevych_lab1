import pandas as pd


def to_model_input(data: dict, feature_columns: list) -> pd.DataFrame:
    """Перетворює один запит на рядок із тими ж колонками, що були при навчанні.

    get_dummies БЕЗ drop_first (для одного рядка drop_first видалив би єдину категорію),
    далі reindex на колонки навчання: відсутні dummy -> 0, зайві -> відкидаються.
    """
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df.astype(float)
