from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
DATA_PATH = BASE_DIR / "titanic-dataset.csv"


def main():
    # Import dataset
    df = pd.read_csv(DATA_PATH)

    # Separate numeric and categorical
    # features
    numeric_features = ["Age", "Parch", "Fare", "SibSp"]

    categorical_features = ["Pclass", "Sex", "Embarked"]

    # Separate features (X) from target (y)
    X = df[numeric_features + categorical_features]
    y = df["Survived"]

    # Create numeric and categorical pipelines,
    # so each pipeline acts on its own set of features.
    # Numeric features need only an imputer (enough for random forest)
    # and categorical features need an imputer and one hot encoding
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="mean"))])

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Preprocessor will apply pipelines to
    # the corresponding features
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )

    # The features will pass through the preprocessor first,
    # then will be fit into the Random Forest
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42),
            ),
        ]
    )

    # Split train data from test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model and display the results
    y_pred = pipeline.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(pipeline, MODEL_PATH)


if __name__ == "__main__":
    main()
