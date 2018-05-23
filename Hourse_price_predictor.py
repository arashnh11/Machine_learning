'''
Arash Nemati Hayati
05/22/2018
Iris flower problem
https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
https://archive.ics.uci.edu/ml/datasets/Iris
'''
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
labels = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width','class']
dataset = pandas.read_csv(url, names = labels)

#shape of the data
print(dataset.shape)

#head of the datasets - labels
print(dataset.head(5))

# summary of the dataset
print(dataset.describe())

# class distribution

print(dataset.groupby('class').size())

# plot the data - first, plots of each individual variable
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

dataset.hist()
#plt.show()

# muti variate plots
scatter_matrix(dataset)
#plt.show()

#create a validation dataset
# use 80% for training, 20% for test
array = dataset.values # this will filter the label attributes
X = array[:, 0:4] # use all columns for training
Y = array[:,4] # class of the data - the answer that ML wants to predict
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                test_size = validation_size, random_state = seed)

#Test with 10-fold cross validation - test on 9 parts - validation on one part - do for all combinations of parts
scoring = 'accuracy'

#include different algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

#Let's evaluate each model
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
