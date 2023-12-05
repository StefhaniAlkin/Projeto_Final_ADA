from bibliotecas_model import *
from treino_teste_06 import X_train, X_test, y_test, y_train, skf

# instanciar classificador
tree = DecisionTreeClassifier(max_depth=10, random_state=42)

# criar pipeline
pipeline_tree = Pipeline(steps=[('scaler', MinMaxScaler()),('classifier', tree)])

# treinar
pipeline_tree.fit(X_train, y_train)

# predicao
y_pred_test_tree = pipeline_tree.predict(X_test)
y_pred_train_tree = pipeline_tree.predict(X_train)

# Utilizar cross_validate para verificar as médias das métricas de teste e treino: overfit/underfit?
results = cross_validate(pipeline_tree, X_train, y_train, cv=skf, scoring='precision', return_train_score=True)

print(f'Precisão de treino: {results["train_score"].mean()}')
print(f'Precisão de teste:  {results["test_score"].mean()}')

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test_tree)
print('Confusion matrix for Decision Tree:')
print(conf_matrix)

ConfusionMatrixDisplay.from_predictions(y_test,y_pred_test_tree)
plt.savefig("./4_Modelagem/Matrix_Tree.png")
plt.close()

print(classification_report(y_test, y_pred_test_tree))