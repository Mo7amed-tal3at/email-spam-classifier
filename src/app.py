from fastapi import FastAPI
from pydantic import BaseModel
import  mlflow.sklearn

model_url = f"runs:/d21df9f766cb498bbbf4ce07d6a052bd/model"
model = mlflow.sklearn.load_model(model_url)



app= FastAPI()
@app.get("/")
def satuts():
    return {"status ":"ok"}