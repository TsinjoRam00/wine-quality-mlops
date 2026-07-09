import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test-environment")

with mlflow.start_run():
    mlflow.log_param("model", "test")
    mlflow.log_metric("accuracy", 0.95)

print("MLflow test completed")
