from bibliotecas_model import *
from treino_teste_06 import X_train, X_test, y_test, y_train, skf

# instanciar classificador KNN
knn = KNeighborsClassifier(n_neighbors=5)

# criar pipeline com KNN
pipeline_knn = Pipeline(steps=[('scaler', MinMaxScaler()), ('classifier', knn)])

# Utilizar cross_validate para verificar as médias das métricas de teste e treino: overfit/underfit?
results = cross_validate(pipeline_knn, X_train, y_train, cv=skf, scoring='precision', return_train_score=True)

print(f'Precisão de treino: {results["train_score"].mean()}')
print(f'Precisão de teste: {results["test_score"].mean()}')

# treinar random search para o KNN
pipeline_knn.fit(X_train, y_train)

# predicao para KNN
y_pred_test_knn = pipeline_knn.predict(X_test)

# confusion matrix para KNN
conf_matrix_knn = confusion_matrix(y_test, y_pred_test_knn)
print("\nConfusion Matrix for KNN:")
print(conf_matrix_knn)

ConfusionMatrixDisplay.from_estimator(pipeline_knn, X_test, y_test)
plt.savefig("./4_Modelagem/Matrix_KNN.png")
plt.close()

print("\nClassification Report for KNN:")
print(classification_report(y_test, y_pred_test_knn))