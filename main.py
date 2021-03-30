from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn import linear_model
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt


def k_nearest_neighbours():
    # KNN k-nearest-neighbors, better to odd numbers
    # weights - nearer the points are, the bigger weights it receive
    # read, print data
    data = pd.read_csv('car.data')
    print(data.head())

    # split data - features and labels
    X = data[[
        'buying',
        'maint',
        'safety'
    ]].values
    y = data['class']

    # converting strings into numbers
    # label encoder
    Le = LabelEncoder()
    for i in range(len(X[0])):
        X[:, i] = Le.fit_transform(X[:, i])
    y[:] = Le.fit_transform(y[:])

    # easier implementation
    # dictionary and mapping
    '''
    Label_dic = {
        'unacc': 0,
        'acc': 1,
        'good': 2,
        'vgood': 3
    }

    y = y.map(Label_dic)
    '''

    X = np.array(X)
    y = np.array(y)

    # model
    knn = neighbors.KNeighborsClassifier(n_neighbors=31, weights='distance')
    # split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # compile the model
    knn.fit(X_train, y_train)

    # test data
    predictions = knn.predict(X_test)

    # test accuracy
    accuracy = metrics.accuracy_score(predictions, y_test)
    print(predictions)
    print(accuracy)


def support_vector_machine():
    # SVM - support vector machine - useful with many dimensional space -> many features classification, regressions knn
    # - makes same spaces where points are classified, svm -> uses a line (2D) (could be linear, could be polynomial,
    # sigmoid) or plane (3D) to classify data, useful with overlapping (knn calculates the distance between nearest
    # neighbors, svm between points and vectors, classifier)
    # load dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    classes = [
        'Iris Setosa',
        'Iris Versicolour',
        'Iris Virginica'
    ]
    # model
    model = svm.SVC()

    # split -> train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # train model
    model.fit(X_train, y_train)

    # make prediction
    predictions = model.predict(X_test)

    # accuracy
    accuracy = metrics.accuracy_score(y_test, predictions)
    print(accuracy)
    print('real:', y_test)
    print('predictions:', predictions)


def linear_regresiion():
    # boston
    boston = datasets.load_boston()

    # split -> features, labels
    X = boston.data
    y = boston.target

    # test, train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # model
    l_reg = linear_model.LinearRegression()
    model = l_reg.fit(X_train, y_train)

    # make predictions
    predictions = model.predict(X_test)

    # calculate accuracy
    print(l_reg.score(X, y))

    # data visualisation
    plt.scatter(X.T[0], y)
    plt.xlabel('crime rate')
    plt.ylabel('price (1000$)')
    plt.show()

    # linear regression -> y = ax + b, minimize r^2 which determines how good is the estimation
    # logistic regression -> y1 = ax + b, then makes sigmoid y2 = 1/(1 + e^-y) (two, separable groups)


def k_means_alg():
    bc = datasets.load_breast_cancer()

    # X & y, scale to improve algorithm
    X = scale(bc.data)
    y = bc.target

    # train, test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # model
    k_means = KMeans(n_clusters=2, random_state=0)

    # in KMeans you're passsing only features, not labels
    model = k_means.fit(X_train)

    predictions = model.predict(X_test)
    print(predictions)
    print(y_test)
    print(metrics.accuracy_score(y_test, predictions))

print('****************************************** KNN ******************************************')
# k_nearest_neighbours()
print('****************************************** SVM ******************************************')
support_vector_machine()
print('*************************************** linear reg ***************************************')
linear_regresiion()
print('***************************************** k-means *****************************************')
k_means_alg()
