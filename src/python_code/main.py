import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Functions to load training and test data
def load_training_data():
    # build path relative to repo root (the container WORKDIR is /app)
    train_path = os.path.join("..", "data", "train.csv")
    print(f"[INFO] loading training data from: {train_path}")

    # check that file exists
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"[ERROR] could not find file at {train_path}")

    # load data
    df = pd.read_csv(train_path)
    print(f"[INFO] training data loaded. shape={df.shape}")
    print(f"[INFO] columns: {list(df.columns)}")
    return df

def load_test_data():
    test_path = os.path.join("..", "data", "test.csv")
    print(f"[INFO] loading test data from: {test_path}")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"[ERROR] could not find file at {test_path}")

    df = pd.read_csv(test_path)
    print(f"[INFO] test data loaded. shape={df.shape}")
    print(f"[INFO] columns: {list(df.columns)}")
    return df


# Preprocessing functions
def preprocess_titanic_train(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:

    # keep id OUT of features but return it
    passenger_ids = df["PassengerId"].copy() if "PassengerId" in df.columns else None

    # detect train vs test
    is_train = "Survived" in df.columns
    y = df["Survived"].copy() if is_train else None

    # start from a copy so we don't mutate caller's df
    X = df.copy()

    # drop columns we don't want the model to see
    drop_cols = [c for c in ["PassengerId", "Name", "Ticket", "Cabin", "Survived"] if c in X.columns]
    X = X.drop(columns=drop_cols)

    # ---- imputations ----
    # Age
    if "Age" in X.columns and X["Age"].isna().any():
        X["Age"] = X["Age"].fillna(X["Age"].median())

    # Fare (often missing in test)
    if "Fare" in X.columns and X["Fare"].isna().any():
        X["Fare"] = X["Fare"].fillna(X["Fare"].median())

    # Embarked
    if "Embarked" in X.columns and X["Embarked"].isna().any():
        X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

    # ---- encoding ----
    # Sex -> binary
    if "Sex" in X.columns:
        X["Sex"] = X["Sex"].map({"male": 0, "female": 1}).astype(int)

    # engineered features
    if "SibSp" in X.columns and "Parch" in X.columns:
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)

    # one-hot Embarked
    if "Embarked" in X.columns:
        X = pd.get_dummies(X, columns=["Embarked"], drop_first=True, dtype=int)

    # final safety net for numerics
    num_cols = X.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # final safety net for non-numerics (should be none, but just in case)
    non_num_cols = X.select_dtypes(exclude=["number"]).columns
    for col in non_num_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].mode()[0])

    return X, y, passenger_ids



def preprocess_titanic_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    # keep PassengerId for later submission
    passenger_ids = df["PassengerId"].copy() if "PassengerId" in df.columns else None

    # make a working copy
    X = df.copy()

    # drop non-feature columns
    drop_cols = [c for c in ["PassengerId", "Name", "Ticket", "Cabin"] if c in X.columns]
    X = X.drop(columns=drop_cols)

    # ---- imputations ----
    if "Age" in X.columns and X["Age"].isna().any():
        X["Age"] = X["Age"].fillna(X["Age"].median())

    if "Fare" in X.columns and X["Fare"].isna().any():
        X["Fare"] = X["Fare"].fillna(X["Fare"].median())

    if "Embarked" in X.columns and X["Embarked"].isna().any():
        X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

    # ---- encoding ----
    if "Sex" in X.columns:
        X["Sex"] = X["Sex"].map({"male": 0, "female": 1}).astype(int)

    if "SibSp" in X.columns and "Parch" in X.columns:
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)

    if "Embarked" in X.columns:
        X = pd.get_dummies(X, columns=["Embarked"], drop_first=True, dtype=int)

    # safety net: fill remaining numeric NaNs
    num_cols = X.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    non_num_cols = X.select_dtypes(exclude=["number"]).columns
    for col in non_num_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].mode()[0])

    return X, passenger_ids



# Model training function
def train_titanic_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Train a logistic regression model on cleaned Titanic data.
    Returns a fitted sklearn Pipeline (with scaling + logistic regression).
    """
    # define pipeline: scaling + logistic regression
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    return model


# Prediction function
def predict_titanic_survival(model: Pipeline, X_test: pd.DataFrame, test_ids: pd.Series) -> pd.DataFrame:
    """
    Predict Titanic survival on the test set and return submission DataFrame.

    Args:
        model: trained sklearn model
        X_test: cleaned test features
        test_ids: PassengerId column from test set
    
    Returns:
        submission: pd.DataFrame with PassengerId and Survived (0/1)
    """
    preds = model.predict(X_test)
    submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Survived": preds.astype(int)
    })
    return submission


if __name__ == "__main__":

    # Load data

    print("-----------------------------Loading Data-----------------------------")

    train_df = load_training_data()
    test_df = load_test_data() 

    print("-----------------------------Data Loaded-----------------------------\n")

    # Preprocess data

    print("\n---------------------------Preprocessing Data--------------------------")

    X_train, y_train, train_ids = preprocess_titanic_train(train_df)
    print("[INFO] preprocessing training data completed.")
    print(f'the first few rows of X_train are:\n{X_train.head()}')

    X_test, test_ids = preprocess_titanic_test(test_df)
    print("[INFO] preprocessing test data completed.")
    print(f'the first few rows of X_test are:\n{X_test.head()}')

    print("-------------------------Data Preprocessing Done-----------------------\n")

    # Train model
    print("-----------------------------Training Model----------------------------")
    model = train_titanic_model(X_train, y_train)
    print("[INFO] model training completed.")
    print(model)
    print("---------------------------Model Training Done-------------------------\n")

    # Predict on test set
    print("---------------------------Predicting on Test Set----------------------")
    prediction_df = predict_titanic_survival(model, X_test, test_ids)
    print("[INFO] prediction on test set completed.")
    print(f'the first few rows of the predicted dataframe are:\n{prediction_df.head()}')
    print("------------------------Prediction on Test Set Done--------------------\n")

    # Save predictions to CSV
    output_path = os.path.join("..", "data", "survival_predictions.csv")
    prediction_df.to_csv(output_path, index=False)
    print("-----------------------Predictions Saved to CSV------------------------")
    print(f"[INFO] predictions saved to: {output_path}")
    print("----------------------------------------------------------------------")