# 01_train_and_log.py
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from mlflow.models.signature import infer_signature
import pandas as pd

mlflow.set_experiment("/Shared/diabetes-demo")

with mlflow.start_run():
    mlflow.sklearn.autolog()

    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    mlflow.log_metric("rmse", rmse)

    # --- infer signature & input example required by Unity Catalog ---
    # make sure sample_input is a pandas DataFrame (so column names/types are captured)
    sample_input = X_test.head(5) if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test).head(5)
    sample_preds = model.predict(sample_input)
    sample_preds_df = pd.DataFrame(sample_preds, columns=["prediction"]) if not isinstance(sample_preds, pd.DataFrame) else sample_preds
    signature = infer_signature(sample_input, sample_preds_df)
    input_example = sample_input

    # Use your catalog.schema.model (replace main.demo_ml with your catalog/schema)
    registered_name = "tejas_catalog.demo_ml.diabetes_rf"

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=registered_name,
        signature=signature,
        input_example=input_example
    )

    print("Completed training. RMSE:", rmse)
    print("Registered model as:", registered_name)