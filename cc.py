from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler

iris = datasets.load_iris()
#iris = MinMaxScaler.fit_transform(iris.data)
#iris1 = datasets.load_iris()

iris.data = MinMaxScaler().fit_transform(iris.data)

'''
                # Decision Tree
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1, random_state=0)

clf = tree.DecisionTreeClassifier(min_impurity_split=.1)
scores = cross_val_score(clf, iris.data, iris.target, cv=10)

print scores
print scores.mean()
#End of Decision Tree
'''

'''
                # Perceptron
clf = linear_model.Perceptron(n_iter=20)
scores = cross_val_score(clf, iris.data, iris.target, cv=10)

print scores
print scores.mean()
#End of Perceptron
# Parameters = 1 - epochs(n_iter)
#3-4 iterations - best ans at 9,13,19
'''

'''
                # Neural Net
clf = MLPClassifier(hidden_layer_sizes=(25,15),activation='identity')
scores = cross_val_score(clf, iris.data, iris.target, cv=10)

print scores
print scores.mean()
# End of Neural Net
'''

'''
                # Deep Learning Net
clf = MLPClassifier(hidden_layer_sizes=(30,20,15,10,8,5),activation='identity')
scores = cross_val_score(clf, iris.data, iris.target, cv=10)

print scores
print scores.mean()
# End of Deep Learning Net
'''

'''
                # SVM
clf = svm.LinearSVC(tol=0.001,max_iter=500,C=1000)
scores = cross_val_score(clf, iris.data, iris.target, cv=10)

print scores
print scores.mean()
# End of SVM
'''

'''
                # Naive Bayes
clf = naive_bayes.GaussianNB()
scores = cross_val_score(clf, iris.data, iris.target, cv=10)

print scores
print scores.mean()
#End of Naive Bayes
#scoring='precision_macro'
'''

'''
                # Logistic Regression
clf = linear_model.LogisticRegression(tol = 0.001, max_iter=500, C=500, solver='lbfgs')
scores = cross_val_score(clf, iris.data, iris.target, cv=10)

print scores
print scores.mean()
#End of Logistic Regression
'''

'''
                 # kNN
clf = KNeighborsClassifier(11)
scores = cross_val_score(clf, iris.data, iris.target, cv=10)

print scores
print scores.mean()
#End of kNN
'''

'''
                # Bagging CLassifier
clf = BaggingClassifier(n_estimators=50,bootstrap=False,bootstrap_features=False)
scores = cross_val_score(clf, iris.data, iris.target, cv=10)

print scores
print scores.mean()
#End of Bagging CLassifier
'''

'''
                # Random Forest
clf = RandomForestClassifier(n_estimators=20,min_impurity_split=0.01,bootstrap=False )
scores = cross_val_score(clf, iris.data, iris.target, cv=10)

print scores
print scores.mean()
#End of Random Forest
'''

'''
                # AdaBoost
clf = AdaBoostClassifier(n_estimators=50,algorithm='SAMME.R')
scores = cross_val_score(clf, iris.data, iris.target, cv=10)

print scores
print scores.mean()
#End of AdaBoost
'''

'''
                # Gradient Boosting
clf = GradientBoostingClassifier(n_estimators=50, min_impurity_split= 0.01)
scores = cross_val_score(clf, iris.data, iris.target, cv=10)

print scores
print scores.mean()
#End of Gradient Boosting
#scoring='precision_weighted'
'''


classifiers = [["Decision Tree",tree.DecisionTreeClassifier(min_impurity_split=.1)],
               ["Perceptron",linear_model.Perceptron(n_iter=8)],
               ["Neural Net",MLPClassifier(hidden_layer_sizes=(25,15),activation='identity')],
               ["Deep Learning",MLPClassifier(hidden_layer_sizes=(30,20,15,10,8,5),activation='identity')],
               ["SVM",svm.LinearSVC(tol=0.001,max_iter=500,C=1000)],
               ["Naive Bayes",naive_bayes.GaussianNB()],
               ["Logistic Regression",linear_model.LogisticRegression(tol = 0.001, max_iter=500, C=500, solver='lbfgs')],
               ["kNN",KNeighborsClassifier(7)],
               ["Bagging CLassifier",BaggingClassifier(n_estimators=50,bootstrap=False,bootstrap_features=False)],
               ["Random Forest",RandomForestClassifier(n_estimators=10,min_impurity_split=0.001,bootstrap=True)],
               ["AdaBoosting",AdaBoostClassifier(n_estimators=50,algorithm='SAMME.R')],
               ["Gradient Boosting",GradientBoostingClassifier(n_estimators=20, min_impurity_split= 0.01)]]
results = []
for name,clf in classifiers:
    scores = cross_val_score(clf, iris.data, iris.target, cv=10)
    prec_scores = cross_val_score(clf, iris.data, iris.target, cv=10, scoring='precision_weighted')
    model = [name,scores.mean(),prec_scores.mean()]
    results.append(model)
print results