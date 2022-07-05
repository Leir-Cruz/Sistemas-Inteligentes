#Define Problem.
#Prepare Data.
#Evaluate Algorithms.
#Improve Results.
#Present Results.

#classification of iris flowers
#Each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). 
#Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other. 




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


#banco de dados;
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"

#parametros usados para classificar
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

#fazendo leitura carregando banco de dados;
dataset = read_csv(url, names=names)


# Os processos a seguir serão para "entendermos" nosso banco de dados

#Quantos objetos tem e em quantas categorias ta dividido
print(dataset.shape)

#visualização limitada geral do banco de dados
print(dataset.head(20))

#resumo das estatisticas
print(dataset.describe())

#o método groupby é para mostrarmos a distribuição por meio de uma das categorias que definimos.
#nesse caso estaremos visualizando pela classe, mas poderiamos buscar pela largura da petola, etc
print(dataset.groupby('class').size())

# Gráficos
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

#histograma para distribuição de frequência
#nesse caso X ta sendo os atributos da planta e y a quantidade
dataset.hist()
pyplot.show()


#analisando interações entre variáveis
#parece um pouco com o histograma, porém levando em consideração duas variaveis
scatter_matrix(dataset)
pyplot.show()



