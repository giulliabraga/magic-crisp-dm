# README: Estrutura do Projeto

Este documento descreve a estrutura do projeto, detalhando a função de cada diretório e arquivo na hierarquia apresentada.

---


# Grupo 01

* Giullia Braga
* Thifany Souza
* Luís Amaral
* Manuel Ferreira-Junior


---


## Diretórios Principais

1. `data`
Contém os arquivos de dados utilizados no projeto.
    - **`teste.csv`**: Dados de teste para validação dos modelos.
    - **`train_winsor_1_norm.csv`**: Dados de treinamento, pré-processados com Winsorização e normalização.
    - **`validation_winsor_1_norm.csv`**: Dados de validação, pré-processados de maneira semelhante aos de treinamento.

2. `metrics_correct`
Armazena métricas de desempenho dos modelos após cross-validation, considerando a aplicação do ADASYN.
    - **`metrics_adasyn_ANNs_cv.csv`**: Métricas do modelo de Redes Neurais Artificiais (ANNs).
    - **`metrics_adasyn_DTR_cv.csv`**: Métricas do modelo Decision Tree Regressor (DTR).
    - **`metrics_adasyn_Heterogêneo_cv.csv`**: Métricas para o cenario de modelo heterogeneo.
    - **`metrics_adasyn_KNN_cv.csv`**: Métricas do modelo K-Nearest Neighbors (KNN).
    - **`metrics_adasyn_LGBM_cv.csv`**: Métricas do modelo LightGBM.
    - **`metrics_adasyn_LVQ_cv.csv`**: Métricas do modelo Learning Vector Quantization (LVQ).
    - **`metrics_adasyn_MLP_cv.csv`**: Métricas do modelo Multi-Layer Perceptron (MLP).
    - **`metrics_adasyn_RF_cv.csv`**: Métricas do modelo Random Forest (RF).
    - **`metrics_adasyn_SVM_cv.csv`**: Métricas do modelo Support Vector Machine (SVM).
    - **`metrics_adasyn_XGB_cv.csv`**: Métricas do modelo XGBoost (XGB).

3. `modules`
Contém os módulos Python com funções reutilizáveis para o pipeline do projeto.
    - **`best_models.py`**: Identificação dos melhores modelos com base em métricas definidas.
    - **`cross_validation.py`**: Implementação do processo de cross-validation.
    - **`lvq_classifier.py`**: Implementação do modelo LVQ.
    - **`model_selector.py`**: Funções para seleção e comparação de modelos.
    - **`optimizer.py`**: Algoritmos de otimização de hiperparâmetros.
    - **`results_visualization.py`**: Geração de gráficos e visualizações de métricas.
    - **`statistical_methods.py`**: Métodos estatísticos auxiliares para análise.

4. `notebooks`
Reúne notebooks Jupyter para experimentação e documentação interativa.
    
    4.1. **`eda.ipynb`**: Análise exploratória dos dados (EDA).
    
    4.2. **`preprocessing.ipynb`**: Pipeline de pré-processamento dos dados.

    4.3. **`optimization.ipynb`**: Otimização de hiperparâmetros dos modelos.
    
    4.4. **`cross_validation.ipynb`**: Execução e análise detalhada de cross-validation.
    
    4.5. **`cv_results.ipynb`**: Análise dos resultados obtidos com cross-validation.
    
    4.6. **`stress_testing.ipynb`**: Testes de robustez e desempenho dos modelos.

5. `outputs`
Destinado aos resultados gerados pelo pipeline, como gráficos, relatórios e modelos salvos.

---

## Como Utilizar
1. Certifique-se de ter as dependências instaladas.
2. Explore os notebooks para reproduzir as análises ou rodar novos experimentos.
3. Utilize os módulos Python para expandir ou adaptar o pipeline do projeto conforme necessário.

---
