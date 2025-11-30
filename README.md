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

🔹 Nível 7 – Deploy e Arquitetura em Produção (AWS)

📌 **Objetivo:**  
Explicar onde e como servir a aplicação em produção, apresentando opções reais de deploy na AWS e o desenho arquitetural recomendado.

Para servir o modelo em produção, a arquitetura ideal depende do volume de requisições, custo esperado e necessidade de escalabilidade.  
Seguindo boas práticas do **AWS Well-Architected Framework**, o fluxo comum inclui um endpoint HTTP (API Gateway) chamando uma camada de computação que executa a inferência.

🔸 **Opção 1 — AWS Lambda + API Gateway (baixa/média demanda)**  
Solução mais simples e barata, executa sob demanda.

**Quando usar:**  
- payload pequeno  
- inferência rápida (< 10–15 s)  
- carga esporádica  

**Fluxo:**  
**Cliente → API Gateway → Lambda → Modelo no S3/MLflow**

**Vantagens:**  
- Escalabilidade automática  
- Custo por execução  
- Zero gestão de servidores  

**Limitações:**  
- Máximo **10 GB** em `/tmp`  
- Máximo **15 min** de execução  
- Cold start com modelos grandes  

---

🔸 **Opção 2 — ECS Fargate + API Gateway (produção contínua)**  
O modelo roda em um container FastAPI sempre ativo.

**Quando usar:**  
- volume moderado/alto  
- modelo precisa ficar carregado em memória  
- latência baixa é prioridade  

**Fluxo:**  
**Cliente → API Gateway → ECS/Fargate → Container FastAPI → MLflow/S3**

**Vantagens:**  
- Baixa latência  
- Escala automática  
- Ótimo para cargas constantes  

---

🔸 **Opção 3 — Amazon SageMaker (MLOps avançado)**  
Solução completa para todo o ciclo de Machine Learning.

**Quando usar:**  
- monitoramento e drift detection  
- autoscaling especializado  
- deploy blue/green  
- versionamento robusto  

**Fluxo:**  
**Cliente → API Gateway → SageMaker Endpoint**

**Vantagens:**  
- Autoscaling de ML nativo  
- Métricas integradas  
- Deploy profissional sem esforço  

---

🔹 Nível 8 – Separação de Treinamento e Inferência + AWS Lambda

📌 **Objetivo:**  
Mostrar como separar corretamente **treinamento** e **inferência**, requisitos mínimos e como rodar inferência em Lambda.

---

🔸 **Separação entre Treinamento e Inferência**

🧠 Treinamento (Training Pipeline)  
Exige mais CPU/GPU/memória e não deve rodar na aplicação de inferência.

**Serviços recomendados:**  
- AWS SageMaker Training Jobs  
- AWS Batch  
- EC2 Spot (barato)  
- ECS Fargate (menos comum)

**Saídas do treinamento:**  
- ✔ Modelo final (`.pkl` ou diretório MLflow)  
- ✔ Metadados  
- ✔ Registro no MLflow Model Registry  
- ✔ Upload no S3  

---

⚡ Inferência (Serving)  
Precisa ser rápida, estável e de baixo custo.  
Nunca deve treinar nada — apenas carregar versões do S3/MLflow.

**Serviços recomendados:**  
- AWS Lambda  
- ECS Fargate  
- SageMaker Endpoint  

---

 **Requisitos mínimos para Inferência**

**Lambda**  
- **512 MB – 1024 MB** RAM recomendados  
- modelos menores que **200 MB**  
- inferência média < **3 s**  

**ECS Fargate**  
- 0.5 vCPU + **1 GB RAM** mínimo  
- ideal para modelo sempre carregado  

**SageMaker Endpoint**  
- instância mínima: **ml.t2.medium**  
- ideal para baixa latência  

---

🔸 **Como usar Lambda para este projeto**

Lambda executa sua FastAPI usando ferramentas como:  
- Mangum  
- AWS Lambda Powertools  
- Zappa  
- Lambyda  

**Fluxo:**  
**Treinamento → S3/MLflow Registry → API Gateway → Lambda → Modelo carregado do S3**

**Passos:**  
1. Empacotar FastAPI + MLflow + dependências (ZIP ou container).  
2. Lambda baixa o modelo para `/tmp`.  
3. O modelo é carregado na primeira execução (cold start).  
4. API Gateway expõe o endpoint para o cliente.  

**Vantagens:**  
- Infra barata  
- Escalabilidade automática  
- Simples de manter  

**Desvantagens:**  
- Cold start  
- Limite de memória e tempo  

---

🔹 **Resumo dos Níveis 7 e 8**

**Nível 7 – Deploy**  
- Deploy recomendado: **API Gateway + Lambda/ECS/SageMaker**  
- **Baixa demanda → Lambda**  
- **Demanda contínua → ECS**  
- **MLOps completo → SageMaker**  

**Nível 8 – Arquitetura**  
- Separar completamente **treinamento** e **inferência**  
- Inferência leve: **Lambda (512–1024MB)** ou **ECS (1GB RAM)**  
- Lambda usa FastAPI + modelo carregado do S3/MLflow via API Gateway  

---

⚡ Dicas Técnicas:

Use MLflow Tracking para registrar e versionar modelos.

Use pydantic (se optar por FastAPI) para validação de payloads.

Estruture a aplicação em camadas para facilitar a evolução entre os níveis.
