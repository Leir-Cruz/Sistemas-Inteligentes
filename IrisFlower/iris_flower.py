from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#setando base de dados
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
attributes = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=attributes)
data_values =  dataset.values

#classes associadas a treino e teste de forma aleatória
X_train, X_validation, Y_train, Y_validation = train_test_split(data_values[:, 0:4], data_values[:, 4], test_size=0.20, random_state=1)


#visualização dos dados de treino e de testes.
#note que X e Y são arrays, o primeiro com as flores e seus atributos e o segundo com os outputs de cada flor
print(f"X_train: \n {X_train} \n" )
print(f"X_validation: \n {X_validation} \n" )
print(f"Y_train: \n {Y_train} \n" )
print(f"Y_validation: \n {Y_validation} \n" )

#agora iremos testar o algoritmo com a chamada validação cruzada: k-cross validation
#usaremos a métrica de acurácia (positivo / total) para avaliar os modelos
#esse modelos que nos referimos são os algoritmos que iremos testar \o/
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))


#Quando testarmos nosso algoritmo, a cada vez que a gente ta pegando os dados de treino e de test
#então faremos a validacao cruzada de cada modelo, assim, cada um deles terá uma medida de acurácia mais precisa
#depois, note que no momento de impressão do resultado de cada modelo, a gente ta pegando a média de todos os dez testes de cada um
#além da média, também estamos imprimindo o desvio padrão
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# visualização por gráficos boxplot
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()