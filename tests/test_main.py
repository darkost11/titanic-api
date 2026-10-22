import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

VALID_PASSENGER = {
    "Pclass": 1,
    "Sex": "female",
    "Fare": 80,
    "Age": 20,
    "SibSp": 0,
    "Parch": 0,
    "Embarked": "S",
}


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_valid():
    response = client.post("/predict", json=VALID_PASSENGER)
    assert response.status_code == 200
    data = response.json()
    assert data["survived"] in (0, 1)


@pytest.mark.parametrize(
    "invalid_field",
    [
        {"Age": -6},
        {"Age": 200},
        {"Fare": -100},
        {"Parch": -2},
        {"SibSp": -4},
        {"Pclass": 4},
        {"Sex": "banana"},
        {"Embarked": "somewhere"},
    ],
)
def test_preict_invalid_input(invalid_field):
    bad = {**VALID_PASSENGER, **invalid_field}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


@pytest.mark.parametrize("missing", list(VALID_PASSENGER))
def test_predict_missing_field(missing):
    bad = {k: v for k, v in VALID_PASSENGER.items() if k != missing}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
