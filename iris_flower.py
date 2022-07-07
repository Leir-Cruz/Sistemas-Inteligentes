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
#x_train: dados de treino
#x_validation: dados referentes aos tvalidacao (nesse caso 20%)
#y_train: classes associadas ao dados de treino
#y_validation: classes associadas ao dados de validacao
X_train, X_validation, Y_train, Y_validation = train_test_split(data_values[:, 0:4], data_values[:, 4], test_size=0.20, random_state=1)
#data_values[:, 0:4] - Mostra todos os atributos de cada objeto do db com as características
#data_values[:, 4]) - Mostra todas as classes de cada objeto do db com as caraterísticas

#agora iremos testar o algoritmo com a chamada validação cruzada: k-cross validation
#usaremos a métrica de acurácia (positivo / total) para avaliar os modelos
#esse modelos que nos referimos são os algoritmos que iremos testar \o/
