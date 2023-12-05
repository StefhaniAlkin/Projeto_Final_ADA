from bibliotecas_preprocess import * 

df_adv_copy = pd.read_csv('./0_Dados/advertising_raw.csv')

df_adv_copy = df_adv_copy.drop(df_adv_copy[df_adv_copy['age'] <= 0].index)
# Além dos dados negativos, foram retirados valores iguais a zero, por entendermos que esses dados não agregam informação à análise

df_adv_copy = df_adv_copy.drop(df_adv_copy[df_adv_copy['daily_time_spent_on_site'] < 0].index)
df_adv_copy = df_adv_copy.drop(df_adv_copy[df_adv_copy['area_income'] < 0].index)
df_adv_copy = df_adv_copy.drop(df_adv_copy[df_adv_copy['daily_internet_usage'] < 0].index)

# Estatística descritiva dos dados após remoção de valores negativos
print("Dados descritivos : \n", df_adv_copy.describe())

print('\nInformações da base: ')
print(df_adv_copy.info())

# Porcentagem de linhas retiradas
qnt_lin, qnt_col = df_adv_copy.shape
pct_excluidos = (101450-qnt_lin)/101450
print(f'A quantidade de linhas retiradas foi {101450-qnt_lin}, que representa {pct_excluidos:.2%} do conjunto de dados.')
print('O shape da base: ', df_adv_copy.shape)

df_adv_copy.to_csv('./0_Dados/advertising_positive.csv', index=False)