from fastapi import FastAPI

app = FastAPI(title="Wine Quality MLOps API")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict():
    return {
        "prediction": "medium",
        "message": "Model not loaded yet"
    }
