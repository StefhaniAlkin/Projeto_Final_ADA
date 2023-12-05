from bibliotecas_preprocess import *

df_adv = pd.read_csv(".\0_Dados\advertising_full.csv", parse_dates=["Timestamp"])

# Cópia da base
df_adv_copy = df_adv.copy()

# Padronização das colunas para minúscula e sem espaços
df_adv_copy.columns = df_adv_copy.columns.str.lower().str.replace (' ','_')

print(f'\nShape da base de dados: ', df_adv_copy.shape)

print(f"\nInformações sobre a base: \n")
print(df_adv_copy.info())

print(f"\nVerificar se há dados duplicadas:", df_adv_copy.duplicated().sum())

print(f"\nDados descritivos:", df_adv_copy.describe())

df_adv_copy.to_csv('./0_Dados/advertising_raw.csv', index=False)