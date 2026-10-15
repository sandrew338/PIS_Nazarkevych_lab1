# Customer Churn MLOps

Лабораторна робота №1 з дисципліни «Проєктування інформаційних систем».
Стартовий MLOps-репозиторій і базова модель для задачі прогнозування відтоку клієнтів (бінарна класифікація).

## Мета проєкту
Створити відтворюваний мінімальний ML-процес: підготовка даних → навчання
базової моделі (StandardScaler + LogisticRegression) → збереження моделі та метрик.
Це початкова точка порівняння для наступних лабораторних.

## Джерело даних
Telco Customer Churn (Kaggle):
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

7043 клієнти, 21 колонка, цільова змінна `Churn` (Yes/No; ~26.5% — «пішли»).
Файл лежить у `data/raw/churn.csv`. Отримати заново:

```bash
curl -L -o ~/Downloads/telco-customer-churn.zip \
  https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn
unzip -p ~/Downloads/telco-customer-churn.zip > data/raw/churn.csv
```

## Структура директорій
```
customer-churn-mlops/
├── data/
│   ├── raw/           # початкові дані без змін
│   └── processed/     # підготовлені дані
├── models/            # збережені моделі (*.joblib не комітяться)
├── notebooks/         # експерименти та первинний аналіз
├── reports/           # метрики, лог запуску
├── src/
│   ├── data_preparation.py
│   └── train.py
├── tests/
├── README.md
├── requirements.txt
└── .gitignore
```

## Встановлення
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск навчання
```bash
python src/train.py
```
Результат: `models/baseline_model.joblib` та `reports/baseline_metrics.json`.

Перевірка (опційно): `pytest`

## Підготовка даних
- колонку `customerID` видалено (унікальний ідентифікатор, не несе сигналу);
- `TotalCharges` приведено до числа (11 порожніх значень -> 0);
- `Churn`: No -> 0, Yes -> 1;
- категоріальні ознаки закодовано через `pd.get_dummies(drop_first=True)`;
- розбиття train/test 80/20, `random_state=42`, `stratify=y`.

## Отримані метрики (test, 1409 записів)
| Метрика   | Значення |
|-----------|----------|
| accuracy  | 0.807    |
| precision | 0.658    |
| recall    | 0.567    |
| f1        | 0.609    |
| roc_auc   | 0.842    |

## Висновок
Базова логістична регресія дає accuracy ≈ 0.81 та ROC-AUC ≈ 0.84. Accuracy
завищена через дисбаланс класів (~73% клієнтів залишаються): модель, що завжди
прогнозує «залишиться», мала б ≈ 0.73. Recall ≈ 0.57 означає, що майже 43%
клієнтів, які підуть, модель пропускає, тож для бізнесу важливіші recall,
F1 та ROC-AUC. Наступні кроки: валідація даних, підбір порогу, складніші
моделі, трекінг експериментів.
