Esta pasta contém 5 (cinco) scripts em Python utilzados para modelagem e otimização do algoritmo de classificação. São eles:

`bibliotecas_model`

* Importa todas as bibliotecas necessárias para execução dos demais scripts

`treino_teste_06.py`

* Converte `advertising_processed.csv` em dataframe

* Quantifica valores únicos das variáveis categóricas

* Divide a base em atributos e target

* Verifica se o target está desbalanceado (proporção de cliques e não cliques)

* Separa conjuntos de treino (70%) e de teste (30%) com estratificação

* Instancia o splitter **K-fold** com `k=10`

`07_knn.py`

* Instancia o algoritmo **K Nearest Neighbours (KNN)** com `k=5`

* Cria pipeline com Min-max Scaler e KNN

* Realiza validação cruzada com a métrica `precision`

* Treina e testa o modelo

* Salva a matriz de confusão correspondente

* Retorna o desempenho do modelo segundo as métricas `accuracy`, `precision`, `recall` e `f-score`.

`08_arvore_de_decisao.py`

* Instancia o algoritmo de **Árvore de Decisão** com `max_depth=10`

* Cria pipeline com Min-max Scaler e Árvore de Decisão

* Realiza validação cruzada com a métrica `precision`

* Treina e testa o modelo

* Salva a matriz de confusão correspondente

* Retorna o desempenho do modelo segundo as métricas `accuracy`, `precision`, `recall` e `f-score`.

`regressao_logistica_09.py`

* Instancia o algoritmo de **Regressão Logística**

* Cria pipeline com Min-max Scaler e Regressão Logística

* Realiza validação cruzada com `scoring='precision'`

* Treina e testa o modelo

* Salva a matriz de confusão correspondente

* Retorna o desempenho do modelo segundo as métricas `accuracy`, `precision`, `recall` e `f-score`.

`10_otimizacao.py`

* Define hiperparâmetros a serem otimizados (`C`, `penalty`, `solver`) no modelo de **Regressão Logística**

* Instancia o método de otimização **Random Search** com a métrica `precision` e com `n_iter=10`

* Retorna os melhores hiperparâmetros

* Treina e testa o modelo otimizado

* Salva a matriz de confusão correspondente

* Salva a curva *Receiver Operating Characteristic* (ROC) e retorna o valor da área sob a curva (AUC)
