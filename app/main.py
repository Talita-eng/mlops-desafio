from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import os

# Cria aplicação FastAPI
app = FastAPI(title="MLOps Nível 1 - Inferência")

# Definição do formato de entrada
class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Função para pegar o modelo mais recente
def load_latest_model(models_dir="models"):
    all_models = [d for d in os.listdir(models_dir) if d.startswith("iris_model_")]
    if not all_models:
        raise FileNotFoundError("Nenhum modelo encontrado na pasta 'models'")
    latest_model = max(all_models, key=lambda d: os.path.getmtime(os.path.join(models_dir, d)))
    model_path = os.path.join(models_dir, latest_model)
    print("Carregando modelo mais recente:", model_path)
    return mlflow.pyfunc.load_model(model_path)

# Carrega o modelo mais recente
model = load_latest_model()

# Nomes das classes do Iris
LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

@app.post("/predict")
def predict(req: PredictRequest):
    data = [[
        req.sepal_length,
        req.sepal_width,
        req.petal_length,
        req.petal_width
    ]]
    pred = model.predict(data)
    pred_int = int(pred[0])
    return {"prediction": pred_int, "label": LABELS.get(pred_int, str(pred_int))}
