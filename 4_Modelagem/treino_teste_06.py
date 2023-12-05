from bibliotecas_model import *

df_adv_copy = pd.read_csv('./0_Dados/advertising_processed.csv', parse_dates=['timestamp'])

# Visualizar as colunas e os tipos de variáveis
print('Informações sobre a base: ')
print(df_adv_copy.info())

# Verificar variáveis categóricas
ad = df_adv_copy['ad_topic_line'].nunique()
country = df_adv_copy['country'].nunique()
city = df_adv_copy['city'].nunique()
timestamp = df_adv_copy['timestamp'].nunique()

valores_unicos = [ad, country, city, timestamp]
variaveis = ['Ad Topic Line', 'Country', 'City', 'Timestamp']

# gráfico de barras
plt.figure(figsize=(10, 6))
sns.barplot(x=variaveis, y=valores_unicos, palette='crest')

plt.xlabel('Variáveis')
plt.ylabel('Quantidade de Valores Únicos')
plt.title('Valores Únicos para Cada Variável Categórica e Timestamp')

# Mostre o gráfico
plt.savefig('./4_Modelagem/quantidade_valores_unicos_variaveis_categoricas.png')
plt.close()

# Separar a base em atributos e target
X = df_adv_copy.drop(columns=['clicked_on_ad', 'city', 'country', 'ad_topic_line', 'timestamp'])
y = df_adv_copy['clicked_on_ad']

df_adv_copy['clicked_on_ad'].value_counts(normalize=True)

# Verificar se o target é desbalanceado
df_adv_copy['clicked_on_ad'].value_counts().plot(kind='bar', rot=0, color=['teal', 'lightgreen'])
plt.savefig('./4_Modelagem/target_desbalanceado.png')
plt.close()

# Dividir a base nos conjuntos de treino e teste:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y )

# Verificar o tamanho dos conjuntos
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Stratified KFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)