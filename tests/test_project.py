
from sklearn.datasets import load_iris
import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "best_model.joblib"
)


def test_iris_dataset_shape():
    iris = load_iris()
    assert iris.data.shape == (150, 4)
    assert len(set(iris.target)) == 3


def test_expected_features_exist():
    iris = load_iris(as_frame=True)
    assert len(iris.feature_names) == 4
    assert all(name in iris.data.columns for name in iris.feature_names)


def test_model_exists():
    assert MODEL_PATH.exists()


def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_prediction():
    model = joblib.load(MODEL_PATH)
    result = model.predict(np.array([[5.1, 3.5, 1.4, 0.2]]))
    assert len(result) == 1
    assert result[0] in [0, 1, 2]

