![Capa](.\5_Imagens\Capa.jpg)

# ::computer_mouse::computer_mouse:**Modelo Preditivo de Classificação de Clique em Anúncios**


O objetivo deste projeto é aplicar os conceitos aprendidos durante a nossa jornada na **Ada** para desenvolver um modelo de aprendizado supervisionado, com a finalidade de classificar o conjunto de dados de publicidade(*advertising*). Este conjunto de dados indica se um usuário específico da *internet* clicou ou não em um anúncio de um site não especificado nesta análise. Vamos trabalhar para criar um modelo que preveja se o usuário realizará o clique com base em suas característica (*features*)


**Equipe** :
* Camila - https://github.com/7cami
* Gisele - https://github.com/xlSilva
* Nathália - https://github.com/martinsnathalia
* Sabrina - https://github.com/abyss-child
* Stefhani - https://github.com/StefhaniAlkin


**Linguagem de programação** : Python                                                    
**Programa**: Quero Ser Data Analytics                                                                           
**Conjunto de dados de origem** : CSV - [Advertising.full](0_Dados\advertising_raw.csv)

**Índice**

1. [Introdução à base](#1-Introducao-a-base)

2. [Bibliotecas Utilizadas](#2-Bibliotecas-utilizadas)

3. [Análise descritiva Exploratória (EDA)](#3-EDA)

   3.1 [Visualização dos dados](#32-Visualizacao-dos-dados)

    3.2 [Análise de valores negativos](#32-analise-de-valores-negativos)

    3.3 [Análise de outliers](#33-analise-de-outliers)

    3.4 [Distribuição das variáveis numéricas](#34-distribuicao-das-variaveis-numericas)

    3.5 [Correlação](#35-correlacao)

4. [Separação dos conjuntos de treinamento e de teste](#110-treinamento-e-teste)

5.  [Modelagem de dados](#5-Modelagem-de-dados)

    5.1 [K Nearest Neighbors](#51-knearestneighbors)

    5.2 [Árvore de decisão](#52-arvore-de-decisao)

    5.3 [Regressão Logística](#53-regressao-logistica)

6. [Otimização do modelo (Regressão Logistica)](#6-otimizacao-do-modelo)


------------------------------------------

## 1. [Introdução à base](#1-Introducao-a-base)

A base de dados fornecida contém um conjunto de dados relacionado à publicidade online, possivelmente para análise de comportamento do usuário ou modelagem preditiva. No que se refere a cada coluna, temos: no que se refere a cada coluna, temos:

|Coluna | Descrição|
|-------|-----------|
|Daily Time Spent on Site| Variável contínua que representa a quantidade de tempo que um usuário passa no site diariamente|
|Age| Variável discreta (inteira) que representa a idade do usuário.|
|Area Income| Variável contínua que pode representar a renda média na área geográfica do usuário.|
|Daily Internet Usage| Variável contínua que representa a quantidade de tempo que um usuário passa na internet diariamente.|
|Ad Topic Line| Variável categórica (qualitativa nominal) que representa o tópico ou o título do anúncio.|
|City| Variável categórica (qualitativa nomial) que representa a cidade do usuário.|
|Male| Variável quantitativa discreta (binária; 0 ou 1) que indica o gênero do usuário.| Ela carrega consigo uma informação qualitativa, em que o valor 1 corresponde ao gênero masculino e 0 ao gênero feminino.|
|Country| Variável categórica (qualitativa nominal) que representa o país do usuário.|
|Timestamp| Variável quantitativa contínua (medida dentro de um intervalo de tempo), que representa quando o usuário interagiu com o anúncio.|
|Clicked on Ad| Variável quantitativa discreta (binária; 0 ou 1).Ela carrega consigo uma informação qualitativa, em que o valor 1 indica se o usuário clicou no anúncio e o 0 se não clicou. Esta é a variável alvo em nosso problema de modelagem preditiva.|

## 2. [Bibliotecas Utilizadas](#2-Bibliotecas-utilizadas)

```´
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc
from scipy.stats import loguniform
```
