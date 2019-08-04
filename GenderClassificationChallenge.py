<<<<<<< HEAD

from sklearn import tree,svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

clfDecisionTree = tree.DecisionTreeClassifier()
clfSVM = svm.LinearSVC()
clfNeural = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(9, 2), random_state=1)

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

clfDecisionTree = clfDecisionTree.fit(X, Y)
clfSVM = clfSVM.fit(X,Y)
clfNeural = clfNeural.fit(X,Y)

predictDecision = clfDecisionTree.predict([[190, 70, 43]])
predictSVM = clfSVM.predict([[190, 70, 43]])
predictNeural = clfNeural.predict([[190,70,43]])

# accDecision = accuracy_score(Y, predictDecision) * 100
# accSVM = accuracy_score(Y, predictSVM) * 100
# accNeural = accuracy_score(Y, predictNeural) * 100

print(predictDecision)
print(predictSVM)
print(predictNeural)
=======

from sklearn import tree,svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

clfDecisionTree = tree.DecisionTreeClassifier()
clfSVM = svm.LinearSVC()
clfNeural = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(9, 2), random_state=1)

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

clfDecisionTree = clfDecisionTree.fit(X, Y)
clfSVM = clfSVM.fit(X,Y)
clfNeural = clfNeural.fit(X,Y)

predictDecision = clfDecisionTree.predict([[190, 70, 43]])
predictSVM = clfSVM.predict([[190, 70, 43]])
predictNeural = clfNeural.predict([[190,70,43]])

# accDecision = accuracy_score(Y, predictDecision) * 100
# accSVM = accuracy_score(Y, predictSVM) * 100
# accNeural = accuracy_score(Y, predictNeural) * 100

print(predictDecision)
print(predictSVM)
print(predictNeural)
>>>>>>> 3d9368d587bef6676528d515b6e0305474551fc9
