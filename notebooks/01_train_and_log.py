# 01_train_and_log.py
import mlflow
import mlflow.sklearn
from sklearn.metrics import root_mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Put your Experiment path here (workspace path)
mlflow.set_experiment("/Shared/diabetes-demo")

with mlflow.start_run():
    mlflow.sklearn.autolog()  # autolog captures params, metrics, model
    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    mlflow.log_metric("rmse", rmse)

    registered_name = "tejas_catalog.demo_ml.diabetes_rf"
    
    # register model into Model Registry
    mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    registered_model_name=registered_name
)

    print("Completed training. RMSE:", rmse)