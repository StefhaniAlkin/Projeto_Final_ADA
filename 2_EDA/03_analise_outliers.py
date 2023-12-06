from bibliotecas_eda import * 

df_adv_copy = pd.read_csv('./0_Dados/advertising_positive.csv')

plt.boxplot(df_adv_copy['age'])
plt.savefig('./2_Eda/boxplot_age_com_outliers.png')
plt.close()

q1 = df_adv_copy['age'].quantile(0.25)
q3 = df_adv_copy['age'].quantile(0.75)
iqr = q3 - q1

lim_inf = q1 - 1.5 * iqr
lim_sup = q3 + 1.5 * iqr

df_adv_copy = df_adv_copy.drop(df_adv_copy[(df_adv_copy['age'] < 18) | (df_adv_copy['age'] > 86)].index).reset_index(drop=True)

qnt_lin, qnt_col = df_adv_copy.shape
pct_excluidos = (101450-qnt_lin)/101450
print(f'A quantidade de linhas retiradas foi {101450-qnt_lin}, que representa {pct_excluidos:.2%} do conjunto de dados.')

df_adv_copy.to_csv('./0_Dados/advertising_processed.csv', index=False)