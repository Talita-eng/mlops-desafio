from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from enum import Enum
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from mlflow import MlflowClient
from datetime import datetime

# =========================================
# Inicialização da aplicação
# =========================================
app = FastAPI(title="MLOps Nível 4 – Listagem de Modelos")

LABELS = {}

# =========================================
# Enum para os tipos de modelo
# =========================================


class ModelType(str, Enum):
    RandomForest = "RandomForest"
    LogisticRegression = "LogisticRegression"
    SVC = "SVC"

# =========================================
# Requisição de predição
# =========================================


class PredictRequest(BaseModel):
    features: list[float]
    true_label: int | None = None

# =========================================
# Função utilitária: carregar modelo mais recente
# =========================================


def load_latest_model(models_dir="models"):
    all_models = [d for d in os.listdir(
        models_dir) if d.startswith("custom_model_")]
    if not all_models:
        raise FileNotFoundError("Nenhum modelo encontrado na pasta 'models'")
    latest_model = max(all_models, key=lambda d: os.path.getmtime(
        os.path.join(models_dir, d)))
    model_path = os.path.join(models_dir, latest_model)
    print("Carregando modelo mais recente:", model_path)
    return mlflow.pyfunc.load_model(model_path)


# =========================================
# Carregar modelo inicial
# =========================================
try:
    model = load_latest_model()
except Exception:
    print("⚠️ Nenhum modelo encontrado ainda. Treine um novo via /train.")
    model = None

# =========================================
# Endpoint de predição
# =========================================


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        return {"error": "Nenhum modelo carregado. Treine um modelo primeiro via /train."}

    data = [req.features]
    pred = model.predict(data)
    pred_int = int(pred[0])
    pred_label = LABELS.get(pred_int, str(pred_int))

    response = {"prediction": pred_int, "label": pred_label}

    if req.true_label is not None:
        correct = pred_int == req.true_label
        response["true_label"] = req.true_label
        response["true_label_name"] = LABELS.get(
            req.true_label, str(req.true_label))
        response["result"] = "Correto" if correct else "Errado"

    return response

# =========================================
# Endpoint de treinamento
# =========================================


@app.post("/train")
async def train_model(
    model_type: ModelType = Form(
        ..., description="Tipo de modelo: RandomForest, LogisticRegression, SVC"),
    file: UploadFile = File(...,
                            description="Arquivo CSV contendo as features e a coluna 'target'")
):
    """
    Treina um modelo a partir de um dataset customizado enviado pelo usuário.
    O CSV deve conter a coluna 'target' como variável alvo.
    """
    global model, LABELS

    # 1️⃣ Carregar dataset customizado
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        return {"error": f"Falha ao ler o arquivo CSV: {str(e)}"}

    if "target" not in df.columns:
        return {"error": "O arquivo CSV precisa conter uma coluna chamada 'target'."}

    X = df.drop("target", axis=1)
    y = df["target"]
    LABELS = {i: str(i) for i in sorted(y.unique())}

    # 2️⃣ Selecionar o tipo de modelo
    if model_type == ModelType.RandomForest:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == ModelType.LogisticRegression:
        clf = LogisticRegression(max_iter=200)
    elif model_type == ModelType.SVC:
        clf = SVC(probability=True)
    else:
        return {"error": "Modelo inválido. Use: RandomForest, LogisticRegression ou SVC."}

    # 3️⃣ Treinar e avaliar
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # 4️⃣ Logar e salvar modelo
    os.makedirs("models", exist_ok=True)
    mlflow.set_experiment("mlops_nivel4_registry")

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")

        # ✅ Registrar o modelo no MLflow Model Registry
        registered_model_name = f"{model_type}_Model"
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name=registered_model_name
        )

        # Salvar modelo localmente
        safe_run_id = run.info.run_id[:8]
        serve_path = f"models/custom_model_{safe_run_id}"
        os.makedirs(serve_path, exist_ok=True)
        mlflow.sklearn.save_model(clf, serve_path)

    # Atualiza o modelo ativo
    model = mlflow.pyfunc.load_model(serve_path)

    return {
        "message": "Novo modelo treinado, registrado e carregado com sucesso!",
        "model_type": model_type,
        "accuracy": acc,
        "run_id": run.info.run_id,
        "model_registry_name": registered_model_name,
        "model_path": serve_path
    }

# =========================================
# ✅ Novo Endpoint: Listagem de Modelos Registrados
# =========================================


@app.get("/models")
def list_models():
    """
    Lista todos os modelos registrados no MLflow Model Registry.
    Retorna nome, versão, estágio e data de criação.
    Compatível com versões antigas do MLflow.
    """
    client = MlflowClient()

    try:
        # Compatibilidade: usa search_registered_models() se list_registered_models() não existir
        if hasattr(client, "list_registered_models"):
            registered_models = client.list_registered_models()
        else:
            registered_models = client.search_registered_models()
    except Exception as e:
        return {"error": f"Falha ao listar modelos: {str(e)}"}

    models_info = []

    for model in registered_models:
        # dependendo da versão, o objeto pode ter estrutura diferente
        latest_versions = getattr(model, "latest_versions", [])
        for version in latest_versions:
            models_info.append({
                "name": model.name,
                "version": version.version,
                "stage": version.current_stage,
                "creation_date": datetime.fromtimestamp(
                    version.creation_timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                "run_id": version.run_id
            })

    if not models_info:
        return {"message": "Nenhum modelo registrado no MLflow ainda."}

    return {"registered_models": models_info}
