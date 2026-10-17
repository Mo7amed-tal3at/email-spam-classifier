import os
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
import  mlflow.sklearn
import mlflow.artifacts
import joblib
from pathlib import Path
from data_prep import EmailToWordCountTransformer, WordCounterToVectorTransformer 

PROJECT_ROOT = Path(__file__).resolve().parent.parent

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(mlflow_uri)

model_url = f"runs:/d21df9f766cb498bbbf4ce07d6a052bd/model"
model = mlflow.sklearn.load_model(model_url)

PIPELINE_RUN_ID= "af4b883b45b6450e81a9530217dcfb2f"
pipeline_path =  mlflow.artifacts.download_artifacts(
    run_id=PIPELINE_RUN_ID,
    artifact_path="preprocess_pipeline.joblib"
)
preprocess_pipeline = joblib.load(pipeline_path)

preprocess_pipeline.named_steps["email_to_wordcount"].input_type = "text"

app = FastAPI(title="Spam Classifier API")

@app.get("/")
def root():
    return {"status ":"ok"}


class EmailRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    is_spam: bool

@app.post("/predict", response_model=PredictionResponse)
def predict(request: EmailRequest):
    vector = preprocess_pipeline.transform([request.text])
    prediction = model.predict(vector)[0]
    return PredictionResponse(
        label="spam" if prediction == 1 else "ham",
        is_spam=bool(prediction),
    )