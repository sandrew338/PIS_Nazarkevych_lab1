import sys
from pathlib import Path

sys.path.insert(0, "src")


def test_required_files_exist():
    for p in ["README.md", "requirements.txt", ".gitignore",
              "src/data_preparation.py", "src/train.py"]:
        assert Path(p).exists(), p


def test_prepare_features():
    from data_preparation import load_data, prepare_features
    X, y = prepare_features(load_data("data/raw/churn.csv"))
    assert len(X) == len(y) > 0
    assert set(y.unique()) <= {0, 1}
    assert X.shape[1] < 100  # customerID не перетворено на dummy
