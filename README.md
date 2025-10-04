# mlops-desafio


🚀 Desafio de MLOps – Progressivo por Nível de Dificuldade

Este desafio foi criado para guiar o aprendizado em MLOps, aumentando gradualmente o nível de complexidade.
O objetivo é construir uma aplicação de serving de modelos de machine learning utilizando Flask ou FastAPI, com gerenciamento de modelos pelo MLflow.


---

🔹 Nível 1 – Aplicação Básica de Inferência

📌 Objetivo:

Criar uma aplicação Flask/FastAPI que:

Carregue em memória um modelo de classificação treinado com alguma biblioteca do scikit-learn.

Disponibilize um endpoint POST /predict que receba dados e retorne a inferência.




---

🔹 Nível 2 – Treinamento sob Demanda

📌 Objetivo:

Adicionar um novo endpoint POST /train que permita treinar o mesmo modelo, mas utilizando outro dataset fornecido pelo cliente.

O modelo treinado deve ser atualizado na aplicação e utilizado nas próximas inferências.



---

🔹 Nível 3 – Opções de Modelos e Datasets

📌 Objetivo:

Expandir o endpoint /train para permitir que o cliente escolha:

O tipo de modelo de classificação (ex.: RandomForestClassifier, LogisticRegression, SVC, etc.).

O dataset a ser utilizado.


O MLflow deve ser integrado para versionar e registrar os experimentos de treinamento.



---

🔹 Nível 4 – Listagem de Modelos

📌 Objetivo:

Criar um endpoint GET /models que liste todos os modelos já treinados e armazenados no MLflow.

A resposta deve incluir informações como: nome, versão e data de criação.



---

🔹 Nível 5 – Troca de Modelo em Memória

📌 Objetivo:

Criar um endpoint POST /use-model que permita ao usuário carregar em memória qualquer modelo listado no MLflow.

Após a troca, o endpoint /predict deve utilizar o novo modelo ativo para realizar inferências.



---

🔹 Nível 6 – Validação de Parâmetros de Inferência

📌 Objetivo:

No endpoint /predict, validar se o cliente está enviando os parâmetros corretos para o modelo carregado.

Caso os parâmetros estejam incorretos, retornar um erro 400 – Bad Request com um exemplo do formato esperado.



---

⚡ Dicas Técnicas:

Use MLflow Tracking para registrar e versionar modelos.

Use pydantic (se optar por FastAPI) para validação de payloads.

Estruture a aplicação em camadas para facilitar a evolução entre os níveis.
