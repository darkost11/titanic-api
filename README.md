# Titanic Survival API

A small REST API built with FastAPI that predicts whether a Titanic passenger would have survived, using a scikit-learn pipeline trained on the Titanic dataset.

- **Model:** scikit-learn `Pipeline` (preprocessing + classifier), saved as `model.joblib`
- **Accuracy:** 0.79
- **Stack:** Python, FastAPI, Pydantic, scikit-learn, Docker

## Run with Docker

```bash
docker build -t titanic-api .
docker run -p 8000:8000 titanic-api
```

The API is now at http://127.0.0.1:8000, and the interactive docs are at http://127.0.0.1:8000/docs.

## Run locally (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Run this from the project root (the folder containing `app/`).

## Endpoints

| Method | Path       | Description                    |
|--------|------------|--------------------------------|
| GET    | `/`        | Check that the API is running  |
| POST   | `/predict` | Predict survival for a passenger |
| GET    | `/docs`    | Interactive API documentation  |

### `POST /predict`

**Input fields**

| Field      | Type   | Allowed values                    |
|------------|--------|-----------------------------------|
| `Pclass`   | int    | `1`, `2`, `3`                     |
| `Sex`      | string | `"male"`, `"female"`              |
| `Age`      | float  | 0 to 120                          |
| `SibSp`    | int    | 0 or more (siblings/spouses aboard) |
| `Parch`    | int    | 0 or more (parents/children aboard) |
| `Fare`     | float  | 0 or more                         |
| `Embarked` | string | `"C"`, `"Q"`, `"S"`               |

**Example request**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 1,
    "Sex": "female",
    "Age": 29,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 80,
    "Embarked": "S"
  }'
```

**Example response**

```json
{
  "survived": 1,
  "label": "Survived",
  "survival_probability": 0.9365
}
```

Invalid input (for example `"Sex": "banana"` or a negative age) returns `422` with a description of what is wrong.

## Tests

```bash
pip install pytest httpx
pytest
```

## Notes

- `model.joblib` must be loaded with the same scikit-learn version it was trained with. Versions are pinned in `requirements.txt`.
- The model was trained on historical data and is meant as a learning project, not for real decisions.
