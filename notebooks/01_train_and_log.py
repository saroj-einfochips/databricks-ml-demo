# 01_train_and_log.py
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# ----- Configuration -----
# Experiment path (workspace experiment)
EXPERIMENT_PATH = "/Shared/diabetes-demo"

# Replace this with your fully-qualified Unity Catalog name: catalog.schema.model_name
# Example: "main.demo_ml.diabetes_rf"
registered_name = "main.demo_ml.diabetes_rf"

# Fallback artifact path (when registry registration fails)
FALLBACK_ARTIFACT_PATH = "model_artifact_only"

# --------------------------

def main():
    mlflow.set_experiment(EXPERIMENT_PATH)

    with mlflow.start_run() as run:
        # Enable autolog to capture many params/metrics/artifacts automatically
        mlflow.sklearn.autolog()

        # Load data
        data = load_diabetes(as_frame=True)
        X, y = data.data, data.target

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)  # sklearn >=1.4
        mlflow.log_metric("rmse", float(rmse))

        # Prepare signature & input example (required by Unity Catalog)
        # Ensure sample_input is a pandas DataFrame so column names/types are captured
        if isinstance(X_test, pd.DataFrame):
            sample_input = X_test.head(5)
        else:
            sample_input = pd.DataFrame(X_test).head(5)

        sample_preds = model.predict(sample_input)
        # Convert predictions to a DataFrame (clear output schema)
        if isinstance(sample_preds, pd.DataFrame):
            sample_preds_df = sample_preds
        else:
            # single column prediction named "prediction"
            sample_preds_df = pd.DataFrame(sample_preds, columns=["prediction"])

        signature = infer_signature(sample_input, sample_preds_df)
        input_example = sample_input

        # Attempt to register model in Unity Catalog (guarded)
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=registered_name,
                signature=signature,
                input_example=input_example,
                await_registration_for=120,  # seconds to wait for registration (adjust if you want)
            )
            print(f"Successfully registered model as: {registered_name}")

        except MlflowException as me:
            # Known Mlflow/Unity-Catalog registration failure (e.g., credential/access connector issue)
            print("MlflowException during model registration. Falling back to artifact-only logging.")
            print("Error:", str(me))

            # Log the model as an artifact only (no registry registration)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=FALLBACK_ARTIFACT_PATH,
                signature=signature,
                input_example=input_example,
            )
            print(f"Model logged to run artifacts under: {FALLBACK_ARTIFACT_PATH}")
            # Optionally record a param to indicate fallback happened
            mlflow.set_tag("registration_fallback", "true")
            mlflow.log_param("fallback_reason", "MlflowException during registry registration")

        except Exception as e:
            # Any unexpected error: also fallback but include message
            print("Unexpected error during model registration. Falling back to artifact-only logging.")
            print("Error:", str(e))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=FALLBACK_ARTIFACT_PATH,
                signature=signature,
                input_example=input_example,
            )
            print(f"Model logged to run artifacts under: {FALLBACK_ARTIFACT_PATH}")
            mlflow.set_tag("registration_fallback", "true")
            mlflow.log_param("fallback_reason", f"unexpected_error: {str(e)}")

        # Final prints for quick visibility in job output
        print("Run ID:", run.info.run_id)
        print("Experiment path:", EXPERIMENT_PATH)
        print("RMSE:", float(rmse))


if __name__ == "__main__":
    main()