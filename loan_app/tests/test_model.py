import pickle
import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
from sklearn.dummy import DummyClassifier

# Se um modelo substituto não atingir esses valores, não é aceito.
ACCURACY_MIN  = 0.75
PRECISION_MIN = 0.75
RECALL_MIN    = 0.70
F1_MIN        = 0.72

@pytest.fixture(scope='module')
def model():
    model_path = Path(__file__).resolve().parent.parent / "best_model.pkl"
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@pytest.fixture(scope='module')
def test_data():
    test_path = Path(__file__).resolve().parent / "test_data.pkl"

    with open(test_path, 'rb') as f:
        data = pickle.load(f)
    return data['X_test'], data['y_test']


def test_model_predicts(model, test_data):
    """Modelo retorna predições com shape e valores corretos."""
    X_test, y_test = test_data
    preds = model.predict(X_test)
    assert preds.shape == (len(y_test),)
    assert set(preds).issubset({0, 1})

def test_accuracy(model, test_data):
    """Accuracy >= ACCURACY_MIN."""
    X_test, y_test = test_data
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\nAccuracy: {acc:.4f} (min: {ACCURACY_MIN})")
    assert acc >= ACCURACY_MIN, f"Accuracy {acc:.4f} abaixo do mínimo {ACCURACY_MIN}"

def test_precision(model, test_data):
    """Precision >= PRECISION_MIN."""
    X_test, y_test = test_data
    prec = precision_score(y_test, model.predict(X_test), zero_division=0)
    print(f"\nPrecision: {prec:.4f} (min: {PRECISION_MIN})")
    assert prec >= PRECISION_MIN, f"Precision {prec:.4f} abaixo do mínimo {PRECISION_MIN}"

def test_recall(model, test_data):
    """Recall >= RECALL_MIN."""
    X_test, y_test = test_data
    rec = recall_score(y_test, model.predict(X_test), zero_division=0)
    print(f"\nRecall: {rec:.4f} (min: {RECALL_MIN})")
    assert rec >= RECALL_MIN, f"Recall {rec:.4f} abaixo do mínimo {RECALL_MIN}"

def test_f1(model, test_data):
    """F1-score >= F1_MIN."""
    X_test, y_test = test_data
    f1 = f1_score(y_test, model.predict(X_test), zero_division=0)
    print(f"\nF1: {f1:.4f} (min: {F1_MIN})")
    assert f1 >= F1_MIN, f"F1 {f1:.4f} abaixo do mínimo {F1_MIN}"

def test_dummy_model_fails(test_data):
    """
    Demonstra que o pipeline de validação rejeita modelos ruins.
    Um DummyClassifier que sempre aprova não deve passar nos thresholds.
    Esse teste valida que os critérios tem poder de rejeição.
    """
    X_test, y_test = test_data

    dummy = DummyClassifier(strategy='stratified', random_state=42)
    dummy.fit(X_test, y_test)
    preds = dummy.predict(X_test)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)

    print(f"\nDummy — Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    assert acc < ACCURACY_MIN, "Dummy tem acurácia menor"
    assert prec < PRECISION_MIN, "Dummy tem precision menor"
    assert f1 < F1_MIN, "Dummy tem f1 menor"