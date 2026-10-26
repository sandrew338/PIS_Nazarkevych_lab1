# Customer Churn MLOps

Лабораторні роботи №1–2 з дисципліни «Проєктування інформаційних систем».
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

## Лабораторна робота №2: валідація даних і версіонування датасету

Нові файли:
- `configs/data_schema.yaml` – опис очікуваної схеми (обов'язкові колонки, допустимі значення `Churn`, числові колонки, максимальна частка пропусків);
- `src/validate_data.py` – скрипт валідації, звіт `reports/data_validation_report.json`;
- `src/version_data.py` – створення версії даних `data/processed/churn_v1.csv` та метаданих `data/processed/churn_v1_metadata.json`.

Рекомендована послідовність команд:
```bash
pip install -r requirements.txt
python src/validate_data.py
python src/version_data.py
python src/train.py
```

### Валідація
Перевіряються: обов'язкові колонки, дублікати, пропуски (порожні та пробільні рядки вважаються пропусками), допустимі значення цільової змінної, коректність числових колонок.
- `status: passed` – критичних помилок немає (`errors` порожній), можна навчати модель; `warnings` (наприклад, дублікати) лише інформують;
- `status: failed` – є хоча б одна помилка в `errors`; скрипт завершується з кодом 1 і навчання запускати не можна.

Результат для Telco Customer Churn: 7043 рядки, 21 колонка, помилок і дублікатів немає, `status: passed`. 11 порожніх значень `TotalCharges` (≈0,16 %) розцінено як пропуски, що менше порогу 20 %.

### Версія даних
`data/processed/churn_v1.csv` – очищений датасет (пробільні значення -> пропуски, `TotalCharges` -> число, пропуски -> 0, дублікати видалено; 7043 рядки).
Метадані (`churn_v1_metadata.json`) містять версію, шляхи, час створення, кількість рядків і SHA-256 вхідного та вихідного файлів, тому завжди можна перевірити, на яких саме даних навчено модель.

Модель навчається на **`data/processed/churn_v1.csv`**; версія даних записується в `reports/baseline_metrics.json` (`"data_version": "v1"`).
Метрики не змінились порівняно з ЛР1, бо підготовка даних дає ті самі значення, але тепер вони відтворювані й прив'язані до версії даних.

## Лабораторна робота №3: MLflow (experiment tracking і Model Registry)

Залежність: `mlflow` (див. `requirements.txt`). Перевірка: `mlflow --version`.

Запуск експериментів (з кореня проєкту):
```bash
python src/train.py --model-type logistic_regression
python src/train.py --model-type random_forest
```
Кожен запуск логує в MLflow параметри (`model_type`, `data_version`, `test_size`, `random_state`, гіперпараметри),
метрики (accuracy, precision, recall, f1, roc_auc), артефакт `reports/latest_metrics.json` та саму модель.

Перегляд результатів:
```bash
mlflow ui --host 127.0.0.1 --port 5000
```
Відкрити http://127.0.0.1:5000. MLflow UI потрібно запускати з **кореня проєкту** (там, де навчання), інакше він не знайде runs.

- Experiment: `customer-churn-baseline` (список runs, порівняння через Compare).
- Runs і артефакти зберігаються локально в `mlruns/` (у git не комітиться).
- Model Registry: вкладка **Models** в UI, модель `CustomerChurnClassifier`; кожен запуск створює нову версію.
- Вибір найкращої моделі описано в `reports/model_selection.md`.
