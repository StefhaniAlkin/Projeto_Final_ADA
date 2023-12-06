from bibliotecas_eda import *

df_adv_copy = pd.read_csv('./0_Dados/advertising_processed.csv', parse_dates=['timestamp'])

# Tabela de correlações
tab_corr = df_adv_copy.corr(numeric_only=True)

print('Tabela de correlações: ', tab_corr)

# Heatmap
sns.heatmap(df_adv_copy.corr(numeric_only=True), annot=True, fmt='.2f', cmap = 'crest' )
plt.savefig('./2_Eda/heatmap_correlacoes.png')
plt.close()