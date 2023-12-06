from bibliotecas_model import *
from treino_teste_06 import X_train, X_test, y_test, y_train, skf

# Instanciando o modelo
logreg = LogisticRegression(random_state=42)

pipeline_logreg = Pipeline([('scaler',MinMaxScaler()), ('modelo',logreg)])

# Utilizar cross_validate para verificar as médias das métricas de teste e treino: overfit/underfit?
results = cross_validate(pipeline_logreg, X_train, y_train, cv=skf, scoring='precision', return_train_score=True)

print(f'Precisão de treino: {results["train_score"].mean()}')
print(f'Precisão de teste: {results["test_score"].mean()}')

# Treino do modelo com os dados de treino
pipeline_logreg.fit(X_train, y_train)

# Predizer os valores de y baseados nos dados de teste
y_pred_test_rl = pipeline_logreg.predict(X_test)

print('\nConfusion Matrix for Logistic Regression:')
print(confusion_matrix(y_test, y_pred_test_rl))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test_rl)
plt.savefig("./4_Modelagem/Matrix_RegLog1.png")
plt.close()

# Verificar a taxa de acerto e erro para os valores preditos e o y_test
print(classification_report(y_test, y_pred_test_rl))