# 🛠️ Industrial Predictive Maintenance Lab

> **AVISO DE ISENÇÃO DE RESPONSABILIDADE (DISCLAIMER):**  
> Este projeto é estritamente experimental e educacional. Todos os dados gerados pelas simulações de sensores são sintéticos e os modelos estatísticos/ML não devem ser utilizados em ambientes de produção industrial sem prévia validação e calibração com dados reais.

## 📌 Visão Geral
O **Industrial Predictive Maintenance Lab** é uma plataforma virtual modular voltada para Engenharia de Computação, projetada para simular ativos industriais, capturar telemetria de sensores em tempo real, detectar anomalias operacionais, calcular a saúde da máquina (Health Score), diagnosticar causas raízes de falha e estimar a vida útil restante (RUL).

---

## 🏗️ Arquitetura do Sistema

[Virtual Machine / Sensores] ──> [Feature Engineering (Pandas)] ──> [Database (SQLite)]
│
├──> [Anomaly Detection (Isolation Forest)]
├──> [Health Score Engine]
├──> [Failure Classifier (Random Forest)]
└──> [RUL Estimator]
│
┌────────────────────┴────────────────────┐
▼                                         ▼
[FastAPI REST API]                     [Streamlit Dashboard]


### Divisão de Camadas:
- **Simulation Layer:** Simulação de estados da máquina e sensores físicos com ruído e degradação progressiva.
- **Data Layer:** Persistência relacional isolada via banco SQLite.
- **ML Layer:** Algoritmos de aprendizado não-supervisionado e supervisionado para diagnóstico preditivo.
- **API Layer:** Interface RESTful em FastAPI para consumo externo.
- **UI Layer:** Dashboard interativo em Streamlit com suporte a monitoramento em tempo real.

---

## 🧰 Tecnologias Utilizadas

- **Linguagem:** Python 3.10+
- **Processamento de Dados:** NumPy, Pandas
- **Machine Learning:** Scikit-learn (Isolation Forest, Random Forest)
- **Backend & Persistence:** FastAPI, Uvicorn, SQLite3, Pydantic
- **Visualização & Dashboard:** Streamlit, Plotly
- **Testes & Qualidade:** Pytest, HTTPX

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
- Python 3.10 ou superior instalados
- Pip e Git

### Instalação Local

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/seu-usuario/industrial-predictive-maintenance.git](https://github.com/seu-usuario/industrial-predictive-maintenance.git)
   cd industrial-predictive-maintenance



