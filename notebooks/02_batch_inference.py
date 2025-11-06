# 02_batch_inference.py
import mlflow
import pandas as pd
from sklearn.datasets import load_diabetes

# Load model from the Model Registry (Production stage)
model_uri = "models:/diabetes_rf/Production"
model = mlflow.pyfunc.load_model(model_uri)

data = load_diabetes(as_frame=True).data.sample(20, random_state=0)
preds = model.predict(data)
out = data.copy()
out['prediction'] = preds

output_path = "/dbfs/tmp/diabetes_predictions.csv"
out.to_csv(output_path, index=False)
print("Wrote predictions to:", output_path)