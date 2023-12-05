from bibliotecas_eda import *

# DISTRIBUIÇÃO DAS VARIÁVEIS NUMÉRICAS

df_adv_copy = pd.read_csv('./0_Dados/advertising_processed.csv', parse_dates=['timestamp'])

numeric_var = df_adv_copy[['daily_time_spent_on_site', 'age', 'area_income', 'daily_internet_usage']]
numeric_var.hist(bins=30, color = 'teal')
plt.suptitle("Distribuição das Variáveis Numéricas")
plt.savefig('./2_EDA/variaveis_numericas.png')
plt.close()

# TIMESTAMP 

#Separação por turno, dia, mês e ano
def shift(timestamp):
    hour = timestamp.hour
    if 0 <= hour < 6:
        return '1'
    elif 6 <= hour < 12:
        return '2'
    elif 12 <= hour < 18:
        return '3'
    else:
        return '4'

df_adv_copy['Shift'] = df_adv_copy['timestamp'].apply(shift)
df_adv_copy['Year'] = df_adv_copy['timestamp'].dt.year
df_adv_copy['Month'] = df_adv_copy['timestamp'].dt.month
df_adv_copy['Day'] = df_adv_copy['timestamp'].dt.day

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
df_adv_copy['Shift'].value_counts().sort_index().plot(kind='bar', color='teal')
plt.title('Distribuição por Turno')
plt.xlabel('Shift')
plt.ylabel('Acessos')
plt.xticks(rotation=0)

plt.subplot(2, 2, 2)
df_adv_copy['Year'].value_counts().sort_index().plot(kind='bar', color='teal')
plt.title('Distribuição por Ano')
plt.ylabel('Acessos')
plt.xticks(rotation=0)

plt.subplot(2, 2, 3)
df_adv_copy['Month'].value_counts().sort_index().plot(kind='bar', color='teal')
plt.title('Distribuição por Mês')
plt.ylabel('Acessos')
plt.xticks(rotation=0)

plt.subplot(2, 2, 4)
df_adv_copy['Day'].value_counts().sort_index().plot(kind='bar', color='teal')
plt.title('Distribuição por Dia')
plt.ylabel('Acessos')
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig('./2_EDA/timestamp.png')
plt.close()

df_adv_copy = df_adv_copy.drop(columns=['Year', 'Shift', 'Month', 'Day'])

# DAILY TIME SPENT ON SITE

# figura: boxplot
print('Estatísticas descritivas do atributo Daily Time: ', df_adv_copy['daily_time_spent_on_site'].describe())

df_adv_copy = df_adv_copy.drop(df_adv_copy[df_adv_copy['daily_time_spent_on_site'] < 0].index)

plt.boxplot(df_adv_copy['daily_time_spent_on_site'].dropna())
plt.savefig('./2_EDA/boxplot_daily_time.png')
plt.close()

# figura: histograma
numeric_var = df_adv_copy[['daily_time_spent_on_site']]
numeric_var.hist(bins=30, color ='teal')
plt.savefig('./2_EDA/histograma_daily_time.png')
plt.close()

# figura: nível de atividade por faixa etária
# Relação entre a idade e o tempo gasto no site
#Separação do tempo gasto no site em três categorias de nível de atividade

#dropando resultados com idade menor que 18 e maior que 86
df_adv_copy = df_adv_copy.drop(df_adv_copy[(df_adv_copy['age'] < 18) | (df_adv_copy['age'] > 86)].index).reset_index(drop=True)

df_adv_copy['Activity Level'] = pd.qcut(df_adv_copy['daily_time_spent_on_site'], 3, labels=['Low', 'Medium', 'High'])
df_adv_copy['Age group'] = pd.cut(df_adv_copy['age'], bins=[17, 30, 60, 500], labels=['Young Adult', 'Adult', 'Elderly'])

grouped_age_site = df_adv_copy.groupby(['Age group', 'Activity Level'])
age_site= grouped_age_site.size().unstack()
print(age_site)

age_site.plot(kind='bar', color=['skyblue', 'teal', 'lightgreen'], title='Distribuição de níveis de atividade por faixa etária', xlabel='Faixa etária', ylabel='Usuários')
plt.legend(title='Nível de Atividade')
plt.xticks(rotation=0)
plt.savefig('./2_EDA/nivel_atividade_faixa_etaria.png')
plt.close()

# figura: nível de ativiade por gênero

#Relaçao entre gênero e tempo gasto no site
grouped_gender_site= df_adv_copy.groupby(['male', 'Activity Level'])
cont_gender_site= grouped_gender_site.size().unstack()
print(cont_gender_site)

avr_gender= df_adv_copy.groupby("male")["daily_time_spent_on_site"].mean()
cont_gender_site.plot(kind='bar', color=['skyblue', 'teal', 'lightgreen'], title='Distribuição de níveis de atividade por gênero', xlabel='Gênero', ylabel='Usuários')
plt.legend(title='Nível de Atividade')
plt.xticks(rotation=0)
plt.savefig('./2_EDA/nivel_atividade_genero.png')
plt.close()

# figura: Distribuição de cliques por nível de atividade

#Relação entre cliques em anúncios e tempo gasto no site
grouped_activity_ad= df_adv_copy.groupby(['Activity Level', 'clicked_on_ad'])
cont_activity_ad= grouped_activity_ad.size().unstack()
print(cont_activity_ad)

cont_activity_ad.plot(kind='bar', color=['lightgreen', 'teal'], title='Distribuição de cliques por nível de atividade', xlabel='Nível de atividade', ylabel='Usuários')
plt.legend(title='Clique no anúncio')
plt.xticks(rotation=0)
plt.savefig('./2_EDA/distribuicao_cloques_nivel_atividade.png')
plt.close()

df_adv_copy.drop(['Activity Level', 'Age group'], axis=1, inplace=True)

# AGE

print('Estatísticas descritivas do atributo Idade: ', df_adv_copy['age'].describe())

idades = df_adv_copy['age']

# Configurações do gráfico de barras
plt.bar(idades.value_counts().index, idades.value_counts().values, color='teal')

# Rótulos e título
plt.xlabel('Idade')
plt.ylabel('Quantidade')
plt.title('Gráfico de Barras para Idade')

# Exibição do gráfico
plt.savefig('./2_EDA/grafico_barras_idade.png')
plt.close()

plt.boxplot(df_adv_copy['age'])
plt.savefig('./2_EDA/boxplot_age.png')
plt.close()
# AREA INCOME 

# Função para calcular o iqr das variáveis
def get_limits(data_variable):
    q1=data_variable.quantile(0.25)
    q3=data_variable.quantile(0.75)
    iqr=q3-q1
    lim_sup=q3+1.5*iqr
    lim_inf=q1-1.5*iqr

    return lim_inf,lim_sup,iqr

# Variavel area_income (quantitativa discreta)
# Medidas de resumo
print("Estatísticas descritivas do atributo Renda por Área: ", df_adv_copy['area_income'].describe())

lim_inf, lim_sup, iqr = get_limits(df_adv_copy.area_income)

print(f"'lim_inf:' {lim_inf}, 'lim_sup:'{lim_sup} e 'iqr:'{iqr}")

# figura: Boxplot
plt.boxplot(df_adv_copy.area_income.dropna())
plt.savefig('./2_EDA/boxplot_area_income.png')
plt.close()

# figura: Histrograma
# Histograma de area_income
plt.hist(df_adv_copy.area_income, bins=32, edgecolor='black', alpha=0.7, color ='teal' )
plt.xlabel('Renda da Área')
plt.ylabel('Frequência')
plt.title('Histograma de Renda da Área')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('./2_EDA/histograma_area_income')
plt.close()

# COUNTRY

# Quantidade de países
print('Valores únicos do atributo País:', df_adv_copy['country'].nunique())

# Porcentagem total de países
df_adv_copy['country'].value_counts(normalize=True)

sns.countplot(df_adv_copy, y='country', order=df_adv_copy['country'].value_counts().index[:10], palette='crest')
plt.title('Top 10: Países')
plt.xlabel('Quantidade Total')
plt.ylabel('Países')
plt.savefig('./2_EDA/top_10_paises.png')
plt.close()


# figura: países por cliques
fig, axes = plt.subplots(1,2, figsize=(24,10))
fig.subplots_adjust()

df_paises_clicaram = df_adv_copy.loc[df_adv_copy['clicked_on_ad']==1]
df_paises_nao_clicaram = df_adv_copy.loc[df_adv_copy['clicked_on_ad']==0]

sns.countplot(df_paises_clicaram, y='country', order=df_paises_clicaram['country'].value_counts().index[:10], palette='crest', ax=axes[0]).set(title='Top 10: Países que clicaram')
sns.countplot(df_paises_nao_clicaram, y='country', order=df_paises_nao_clicaram['country'].value_counts().index[:10], palette='crest', ax=axes[1]).set(title='Top 10: Países que não clicaram')

axes[0].set_xlabel('Quantidade')
axes[0].set_ylabel('Países')
axes[1].set_xlabel('Quantidade')
axes[1].set_ylabel('Países')

plt.savefig('./2_EDA/top_10_paises_cliques.png')
plt.close()

# DAILY INTERNET USAGE

print('Estatísticas descritivas do Daily Internet Usage:', df_adv_copy['daily_internet_usage'].describe())

df_adv_copy['daily_internet_usage'].value_counts()

fig, axes = plt.subplots(1,2, figsize=(12,6))
fig.subplots_adjust()

sns.boxplot(df_adv_copy['daily_internet_usage'], palette='crest', ax=axes[0])
sns.boxplot(data=df_adv_copy, y="daily_internet_usage", hue="clicked_on_ad", palette='crest')

axes[0].set_xlabel('Total')
axes[0].set_ylabel('Uso diário de Internet')
axes[1].set_xlabel('Não clicaram [0] ou clicaram [1]')
axes[1].set_ylabel('Uso diário de Internet')

plt.savefig('./2_EDA/boxplot_daily_internet_usage.png')
plt.close()

df_adv_copy.loc[df_adv_copy['daily_internet_usage']==df_adv_copy['daily_internet_usage'].min()]

df_adv_copy.loc[df_adv_copy['daily_internet_usage']==df_adv_copy['daily_internet_usage'].max()]

df_adv_copy['user'] = pd.qcut(df_adv_copy['daily_internet_usage'], 3, labels=['Light', 'Medium', 'Heavy'])
df_adv_copy[['daily_internet_usage','user']]

df_adv_copy['user'].value_counts(normalize=True)

# figura: usuários
df_adv_copy['user'].value_counts().plot(kind='bar',color='teal')
plt.xticks(rotation=0)
plt.savefig('./2_EDA/usuarios.png')
plt.close()

# figura: 
sns.countplot(df_adv_copy, x='user', hue='clicked_on_ad', palette={0: 'teal', 1: 'lightgreen'})
plt.ylabel('Quantidade de pessoas')
plt.xlabel('Tipo de usuário')
plt.savefig('./2_EDA/tipo_de_usuarios_por_cliques.png')
plt.close()

# AD TOPIC LINE

print('Estatísticas descritivas do atributo Ad Topic Line:', df_adv_copy['ad_topic_line'].describe())

# figura: wordcloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap = 'crest').generate(' '.join(df_adv_copy['ad_topic_line']))

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Nuvem de Palavras - Assunto do Anúncio')
plt.savefig('./2_EDA/wordcloud_ad_topic_line.png')
plt.close()

# figura: top 10 ad topic line
sns.countplot(df_adv_copy, y='ad_topic_line', order=df_adv_copy['ad_topic_line'].value_counts().index[:10], palette = 'crest')
plt.savefig('./2_EDA/top10_ad_topic_line.png')
plt.close()

# CITY

city = df_adv_copy['city']
print('Estatísticas descritivas de cidade: ', city.describe())


# figura: wordcloud

text = ' '.join(city.dropna())

wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='crest').generate(text)

plt.figure(figsize=(10, 5))

plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Nuvem de Palavras - Cidades')
plt.axis('off')

plt.savefig('./2_EDA/wordcloud_cidades.png')
plt.close()

# figura: top10 cidades

sns.countplot(df_adv_copy, y='city', order=df_adv_copy['city'].value_counts().index[:10], palette = 'crest')

plt.title('Cidades mais frequentes na base de dados')

plt.savefig('./2_EDA/top10_cidades.png')
plt.close()

# fiura: top10 cidades por cliques

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

sns.countplot(df_adv_copy[df_adv_copy['clicked_on_ad']==1], y='city',
              order=df_adv_copy[df_adv_copy['clicked_on_ad']==1]['city'].value_counts().index[:10], ax=axs[0], palette = 'crest')
axs[0].set_title('Cidades mais frequentes COM cliques no anúncio')


sns.countplot(df_adv_copy[df_adv_copy['clicked_on_ad']==0], y='city',
              order=df_adv_copy[df_adv_copy['clicked_on_ad']==0]['city'].value_counts().index[:10], ax=axs[1], palette = 'crest')
axs[1].set_title('Cidades mais frequentes SEM cliques no anúncio')

plt.tight_layout()
plt.savefig('./2_EDA/top10_cidades_cliques')
plt.close()

# MALE

# Variavel male (quantitativa discreta)
df_adv_copy['male'].value_counts()

# figura: distruição por sexo
df_sex = df_adv_copy['male'].value_counts()
df_sex = df_sex.rename({0: 'female', 1: 'male'})
df_sex.plot(kind='bar', rot=0, color='teal')
# titulo e legenda
plt.title('Distribuição por Sexo')
plt.xlabel('Sexo')
plt.ylabel('Contagem')

plt.savefig('./2_EDA/distribuicao_sexo.png')
plt.close()

# figura: cliques por sexo

# 1-Observação: calculando número de cliques agrupados por sexo
df_adv_copy.groupby(['male','clicked_on_ad']).size()

# DataFrame dos resultados
df_clicked_by_sex = df_adv_copy.groupby(['male', 'clicked_on_ad']).size().reset_index(name='count')
df_clicked_by_sex['male'] = df_clicked_by_sex['male'].map({0: 'female', 1: 'male'})

# Gráfico de barras
plt.figure(figsize=(8, 6))
sns.barplot(x='male', y='count', hue='clicked_on_ad', data=df_clicked_by_sex, palette={0: 'teal', 1: 'lightgreen'})

# Adicionando legenda e título
plt.xlabel('Sexo')
plt.ylabel('Contagem')
plt.title('Cliques por sexo')
plt.legend(title='Clicked')

plt.savefig('./2_EDA/cliques_por_sexo.png')
plt.close()


# figura: média do tempo diário gasto no site por sexo

# 2-Observação: calculando métricas de tempo diário gasto no site agrupadas por sexo
df_adv_copy.groupby(['male'])['daily_time_spent_on_site'].mean()

# Criando um DataFrame com os resultados
df_time_spent_on_site_by_sex = df_adv_copy.groupby(['male'])['daily_time_spent_on_site'].mean().reset_index(name='mean_time')

# Criando o gráfico de barras
plt.figure(figsize=(8, 6))
sns.barplot(x='male', y='mean_time', data=df_time_spent_on_site_by_sex, hue='male', palette='crest', dodge=False)

# Adicionando rótulos e título
plt.xlabel('Sexo')
plt.ylabel('Tempo Médio Diário Gasto no Site')
plt.title('Média do Tempo Diário Gasto no Site por Sexo')

plt.savefig('./2_EDA/media_tempo_gasto_site_por_sexo.png')
plt.close()

# figura: média de tempo gasto no site - scatterplot

# Gráfico de ponto
plt.figure(figsize=(8, 6))
sns.scatterplot(x='male', y='mean_time', data=df_time_spent_on_site_by_sex, color='black', s=100)

# Adicionando rótulos e título
plt.xlabel('Sexo')
plt.ylabel('Tempo Médio Diário Gasto no Site')
plt.title('Média do Tempo Diário Gasto no Site por Sexo')

plt.savefig('./2_EDA/media_tempo_gasto_site_por_sexo_scatterplot.png')
plt.close()


# CLICKED ON AD

# Estatítica descritiva da coluna
click = df_adv_copy['clicked_on_ad']
print('Estatísticas descritivas do Clique por Anúncio: ', click.describe())


# figura: distribuição de cliques por anúncio

click.value_counts().plot(kind='bar', color=['lightgreen','teal'])
plt.title('Distribuição de cliques no anúncio')
plt.xlabel('clicked_on_ad')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.savefig('./2_EDA/distribuicao_de_cliques.png')
plt.close()

# figura: cliques no anúncio por idade

#clicked_on_ad vs age

click_age = df_adv_copy[click == 1]['age']
sns.histplot(data=click_age, kde=True, bins=300, color='lightgreen')

not_click_age = df_adv_copy[click == 0]['age']
sns.histplot(data=not_click_age, kde=True, bins=300, color='teal')

plt.title('Distribuição de cliques no anúncio por idade')
plt.legend(title='clicked_on_ad', labels=['0', '1'])
plt.savefig('./2_EDA/cliques_no_anuncio_por_idade.png')
plt.close()

# figura: distribuição de cliques no anúncio por renda da região

# clicked_on_ad vs area_income

click_income = df_adv_copy[click == 1]['area_income']
sns.histplot(data=click_income, kde=True, bins=300, color='lightgreen')

not_click_income = df_adv_copy[click == 0]['area_income']
sns.histplot(data=not_click_income, kde=True, bins=300, color='teal')

plt.title('Distribuição de cliques no anúncio por renda da região')
plt.legend(title='clicked_on_ad', labels=['0', '1'])
plt.savefig('./2_EDA/cliques_no_anuncio_por_renda.png')
plt.close()

# figura: cliques no anúncio por uso diário de internet

# clicked_on_ad vs daily_internet_usage

click_usage = df_adv_copy[click == 1]['daily_internet_usage']
sns.histplot(data=click_usage, kde=True, bins=300, color='lightgreen')

not_click_usage = df_adv_copy[click == 0]['daily_internet_usage']
sns.histplot(data=not_click_usage, kde=True, bins=300, color='teal')

plt.title('Distribuição de cliques no anúncio por uso diário de internet')
plt.legend(title='clicked_on_ad', labels=['0', '1'])
plt.savefig('./2_EDA/cliques_no_anuncio_uso_diario.png')
plt.close()

# figura: cliques no anúncio por tempo diário gasto no site

# clicked_on_ad vs daily_time_spent_on_site

click_site = df_adv_copy[click == 1]['daily_time_spent_on_site']
sns.histplot(data=click_site, kde=True, bins=300, color='lightgreen')

not_click_site = df_adv_copy[click == 0]['daily_time_spent_on_site']
sns.histplot(data=not_click_site, kde=True, bins=300, color='teal')

plt.title('Distribuição de cliques no anúncio por tempo diário gasto no site')
plt.legend(title='clicked_on_ad', labels=['0', '1'])
plt.savefig('./2_EDA/cliques_anuncio_tempo_diario_gasto_site.png')
plt.close()