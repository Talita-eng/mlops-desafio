from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from enum import Enum
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import os
import json
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
app = FastAPI(title="MLOps Nível 6")

LABELS = {}
model = None
# 🔹 Novo: guarda informações sobre o modelo ativo (n_features, etc.)
MODEL_METADATA = {}

# =========================================
# Enum para tipos de modelos
# =========================================


class ModelType(str, Enum):
    RandomForest = "RandomForest"
    LogisticRegression = "LogisticRegression"
    SVC = "SVC"

# =========================================
# Estrutura de requisição de predição
# =========================================


class PredictRequest(BaseModel):
    features: list[float]
    true_label: int | None = None

# =========================================
# Utilitários de modelo
# =========================================


def load_latest_model(models_dir="models"):
    """Carrega o modelo mais recente salvo localmente."""
    all_models = [d for d in os.listdir(
        models_dir) if d.startswith("custom_model_")]
    if not all_models:
        raise FileNotFoundError("Nenhum modelo encontrado na pasta 'models'")
    latest_model = max(all_models, key=lambda d: os.path.getmtime(
        os.path.join(models_dir, d)))
    model_path = os.path.join(models_dir, latest_model)
    print("Carregando modelo mais recente:", model_path)
    return model_path, mlflow.pyfunc.load_model(model_path)


def load_model_metadata(model_path):
    """Carrega metadados salvos junto ao modelo (número de features, nomes, etc.)."""
    global MODEL_METADATA
    metadata_path = os.path.join(model_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            MODEL_METADATA = json.load(f)
        print(f"Metadados carregados: {MODEL_METADATA}")
    else:
        MODEL_METADATA = {}
        print("⚠️ Nenhum metadata.json encontrado para este modelo.")


# =========================================
# Tenta carregar modelo inicial
# =========================================
try:
    model_path, model = load_latest_model()
    load_model_metadata(model_path)
except Exception:
    print("⚠️ Nenhum modelo encontrado ainda. Treine um novo via /train.")
    model = None

# =========================================
# Endpoint de predição (Nível 6)
# =========================================


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(
            status_code=400, detail="Nenhum modelo carregado. Treine ou selecione um modelo primeiro via /train ou /use-model.")

    # 🔹 Validação 1: quantidade de parâmetros
    expected_features = MODEL_METADATA.get("n_features")
    if expected_features and len(req.features) != expected_features:
        example = [0.0] * expected_features
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Número incorreto de parâmetros: esperado {expected_features}, recebido {len(req.features)}.",
                "expected_format": {"features": example},
            }
        )

    # 🔹 Validação 2: tipo dos dados
    if not all(isinstance(x, (int, float)) for x in req.features):
        raise HTTPException(
            status_code=400,
            detail="Os valores em 'features' devem ser numéricos (int ou float)."
        )

    # ✅ Realiza predição
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
# Endpoint de treinamento (Níveis 2–5)
# =========================================


@app.post("/train")
async def train_model(
    model_type: ModelType = Form(
        ..., description="Tipo de modelo: RandomForest, LogisticRegression, SVC"),
    file: UploadFile = File(...,
                            description="Arquivo CSV com as features e a coluna 'target'")
):
    global model, LABELS, MODEL_METADATA

    # 1️⃣ Carregar dataset
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        return {"error": f"Falha ao ler o arquivo CSV: {str(e)}"}

    if "target" not in df.columns:
        return {"error": "O arquivo CSV precisa conter uma coluna chamada 'target'."}

    X = df.drop("target", axis=1)
    y = df["target"]
    LABELS = {i: str(i) for i in sorted(y.unique())}

    # 2️⃣ Escolher tipo de modelo
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
    mlflow.set_experiment("mlops_nivel6_registry")

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")

        # Registrar modelo no MLflow Model Registry
        registered_model_name = f"{model_type}_Model"
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model", name=registered_model_name)

        # Salvar modelo localmente
        safe_run_id = run.info.run_id[:8]
        serve_path = f"models/custom_model_{safe_run_id}"
        os.makedirs(serve_path, exist_ok=True)
        mlflow.sklearn.save_model(clf, serve_path)

        # 🔹 Novo: salvar metadados
        metadata = {"n_features": X.shape[1], "feature_names": list(X.columns)}
        with open(os.path.join(serve_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    # Atualiza modelo ativo e metadados
    model = mlflow.pyfunc.load_model(serve_path)
    MODEL_METADATA = metadata

    return {
        "message": "Novo modelo treinado, registrado e carregado com sucesso!",
        "model_type": model_type,
        "accuracy": acc,
        "run_id": run.info.run_id,
        "model_registry_name": registered_model_name,
        "model_path": serve_path,
        "metadata": metadata
    }

# =========================================
# Listagem de modelos (Nível 4)
# =========================================


@app.get("/models")
def list_models():
    client = MlflowClient()
    try:
        if hasattr(client, "list_registered_models"):
            registered_models = client.list_registered_models()
        else:
            registered_models = client.search_registered_models()
    except Exception as e:
        return {"error": f"Falha ao listar modelos: {str(e)}"}

    models_info = []
    for model_obj in registered_models:
        latest_versions = getattr(model_obj, "latest_versions", [])
        for version in latest_versions:
            models_info.append({
                "name": model_obj.name,
                "version": version.version,
                "stage": version.current_stage,
                "creation_date": datetime.fromtimestamp(version.creation_timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                "run_id": version.run_id
            })

    if not models_info:
        return {"message": "Nenhum modelo registrado no MLflow ainda."}

    return {"registered_models": models_info}

# =========================================
# Troca de modelo ativo (Nível 5)
# =========================================


@app.post("/use-model")
def use_model(model_name: str = Form(...), version: str = Form(...)):
    global model, MODEL_METADATA

    client = MlflowClient()
    try:
        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(
            f"✅ Modelo carregado em memória: {model_name} (versão {version})")

        # Tenta carregar metadados locais se existir
        model_dir = f"models/{model_name}_v{version}"
        if os.path.exists(model_dir):
            load_model_metadata(model_dir)

        return {"message": f"Modelo {model_name} (versão {version}) carregado com sucesso e agora é o ativo."}
    except Exception as e:
        return {"error": f"Falha ao carregar modelo {model_name} versão {version}: {str(e)}"}
