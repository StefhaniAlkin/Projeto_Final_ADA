from bibliotecas_model import *
from treino_teste_06 import X_train, X_test, y_test, y_train, skf
from regressao_logistica_09 import *

param_dist = {'C': loguniform(1e-4, 1e4),'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga']}

random_search = RandomizedSearchCV(logreg, param_distributions=param_dist,
                                   n_iter=10, cv=skf, scoring='precision', random_state=42)
random_search.fit(X_train, y_train)

# Melhor parâmetro Random_search
print('Melhores parâmetros: ', random_search.best_params_)

# Treino do melhor modelo de RL com Random Search
y_pred_rs = random_search.predict(X_test)
print(classification_report(y_test, y_pred_rs))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rs)
plt.savefig("./4_Modelagem/Matrix_RegLog2.png")
plt.close()

fpr, tpr, thresholds = roc_curve(y_test, random_search.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.savefig("./4_Modelagem/Curva_ROC.png")
plt.close()