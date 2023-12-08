Esta pasta contém 3 (três) scripts em Python utilizados para Análise Exploratória dos Dados (EDA). A EDA proporciona a investigação e o entendimento iniciais dos dados, assim como as relações existentes entre eles.

`03_analise_outliers.py`

* Leitura do conjunto de dados

* Remoção de outliers na variável `idade`

* Remoção de dados relacionados a valores específicos da variável `idade`

* Geração de estatísticas descritivas

* Exportação do dataframe gerado após estes processamentos

`04_analise_variaveis.py`

* Leitura do conjunto de dados

* Exploração da distribuição das variáveis numéricas através gráficos e estatísticas descritivas

* Análise da variável `timestamp` por meio de gráficos de barras

* Análise da variável `daily_time_spent_on_site`, remoção dos dados relacionados a valores negativos desta variável e sua relação com as variáveis `age_group`, `Activity Level` e `clicked_on_ad`

* Análise das variáveis `country`, `daily_internet_usage`, `clicked_on_ad`, `ad_topic_line`, `city`, `male` e `clicked_on_ad` e suas relações com outras variáveis

`05_correlacao.py`

* Leitura do conjunto de dados

* Geração do mapa de calor das correlações