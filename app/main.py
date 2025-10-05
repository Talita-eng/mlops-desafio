from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Cria aplicação FastAPI
app = FastAPI(title="MLOps Nível 2 – Treinamento sob Demanda")

# Nomes das classes do Iris
LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

# =============================
# Classe para requisição de predição
# =============================
class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    true_label: int | None = None


# =============================
# Função para carregar o modelo mais recente
# =============================
def load_latest_model(models_dir="models"):
    all_models = [d for d in os.listdir(models_dir) if d.startswith("iris_model_")]
    if not all_models:
        raise FileNotFoundError("Nenhum modelo encontrado na pasta 'models'")
    latest_model = max(all_models, key=lambda d: os.path.getmtime(os.path.join(models_dir, d)))
    model_path = os.path.join(models_dir, latest_model)
    print("Carregando modelo mais recente:", model_path)
    return mlflow.pyfunc.load_model(model_path)


# Carrega o modelo inicial
model = load_latest_model()


# =============================
# Endpoint de predição
# =============================
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
    pred_label = LABELS.get(pred_int, str(pred_int))

    response = {
        "prediction": pred_int,
        "label": pred_label
    }

    if req.true_label is not None:
        correct = pred_int == req.true_label
        response["true_label"] = req.true_label
        response["true_label_name"] = LABELS.get(req.true_label, str(req.true_label))
        response["result"] = "Correto" if correct else "Errado"

    return response


# =============================
# Endpoint de TREINAMENTO SOB DEMANDA
# =============================
@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    """
    Endpoint para treinar um novo modelo a partir de um CSV fornecido pelo cliente.
    O CSV deve conter as features e a coluna target (label).
    """
    global model  # permite atualizar o modelo em memória

    # Lê o CSV enviado
    df = pd.read_csv(file.file)

    # Verifica se há uma coluna de target
    if "target" not in df.columns:
        return {"error": "O dataset precisa conter uma coluna chamada 'target'."}

    # Divide em treino e teste
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treina novo modelo
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Avalia o modelo
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Salva no MLflow
    os.makedirs("models", exist_ok=True)
    mlflow.set_experiment("mlops_nivel2")

    with mlflow.start_run() as run:
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("n_estimators", 100)
        mlflow.sklearn.log_model(clf, "model")

        safe_run_id = run.info.run_id[:8]
        serve_path = f"models/iris_model_{safe_run_id}"
        os.makedirs(serve_path, exist_ok=True)
        mlflow.sklearn.save_model(clf, serve_path)

    # Atualiza o modelo em memória para uso nas próximas inferências
    model = mlflow.pyfunc.load_model(serve_path)

    return {
        "message": "Novo modelo treinado e carregado com sucesso!",
        "run_id": run.info.run_id,
        "accuracy": acc,
        "model_path": serve_path
    }
