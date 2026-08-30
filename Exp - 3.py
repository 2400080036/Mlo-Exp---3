import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dagshub.init(
    repo_owner="2400080036",
    repo_name="MLO-DagsHub-Exp-2",
    mlflow=True
)

data = pd.read_csv("data.csv")

X = data[["feature1", "feature2"]]
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():

    model = LogisticRegression()
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, name="model")

    print("Accuracy:", accuracy)
    print("Model trained successfully")