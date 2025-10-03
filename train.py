import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from abc import ABC, abstractmethod


# =============================
# 1) Inversão de Dependência (SOLID - I)
# =============================
class IModelTrainer(ABC):
    """Interface para treino de modelos."""

    @abstractmethod
    def train(self, X_train, y_train):
        """Treina o modelo."""
        pass

    @abstractmethod
    def predict(self, X_test):
        """Realiza predição."""
        pass


class RandomForestTrainer(IModelTrainer):
    """Treinador usando RandomForest."""

    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


# =============================
# 2) Classe principal com várias funções (em vez de função única)
# =============================
class IrisModelPipeline:
    """Pipeline para treinar, avaliar e salvar modelo de Iris usando MLflow."""

    def __init__(self, trainer: IModelTrainer, experiment_name="mlops_nivel1"):
        """
        Args:
            trainer (IModelTrainer): Instância de treinador que segue a interface IModelTrainer.
            experiment_name (str): Nome do experimento no MLflow.
        """
        self.trainer = trainer
        mlflow.set_experiment(experiment_name)

    def carregar_dados(self):
        """Carrega o dataset Iris e divide em treino e teste.
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X, y = load_iris(return_X_y=True)
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def treinar_modelo(self, X_train, y_train):
        """Treina o modelo."""
        self.trainer.train(X_train, y_train)

    def avaliar_modelo(self, X_test, y_test):
        """Avalia o modelo e retorna métricas.
        
        Args:
            X_test (ndarray): Dados de teste.
            y_test (ndarray): Rótulos de teste.
        
        Returns:
            tuple: (accuracy, confusion_matrix, y_pred)
        """
        y_pred = self.trainer.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return acc, cm, y_pred

    def salvar_modelo(self, acc, cm):
        """Loga e salva modelo no MLflow."""
        with mlflow.start_run() as run:
            # Parâmetros e métricas
            mlflow.log_param("n_estimators", self.trainer.model.n_estimators)
            mlflow.log_metric("accuracy", acc)

            # Salva matriz de confusão como artefato
            cm_path = "confusion_matrix.txt"
            with open(cm_path, "w") as f:
                f.write(str(cm))
            mlflow.log_artifact(cm_path)

            # Loga modelo no MLflow
            mlflow.sklearn.log_model(self.trainer.model, artifact_path="model")

            # Salva modelo local com run_id seguro
            safe_run_id = run.info.run_id[:8]
            serve_path = f"models/iris_model_{safe_run_id}"
            os.makedirs(serve_path, exist_ok=True)
            mlflow.sklearn.save_model(sk_model=self.trainer.model, path=serve_path)

            print("Modelo salvo em:", serve_path)
            print("Run ID:", run.info.run_id)
            print(f"Acurácia: {acc:.4f}")
            print("Matriz de confusão:\n", cm)


# =============================
# 3) Execução
# =============================
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    trainer = RandomForestTrainer(n_estimators=100, random_state=42)
    pipeline = IrisModelPipeline(trainer)

    X_train, X_test, y_train, y_test = pipeline.carregar_dados()
    pipeline.treinar_modelo(X_train, y_train)
    acc, cm, _ = pipeline.avaliar_modelo(X_test, y_test)
    pipeline.salvar_modelo(acc, cm)
