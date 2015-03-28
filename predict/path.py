__author__ = 'sean'

import sys
from sklearn.externals import joblib
from sklearn.datasets import load_iris
from sklearn import svm
from classifyMutation import main
import getpass
import sklearn

print(getpass.getuser())
print(sklearn.__version__)
print(joblib.__version__)



"""
class PPtest(object):
    def __init__(self, classifier):
        self.classifier = classifier

iris = load_iris()

X = iris.data
y = iris.target

clf = svm.SVC()
clf.fit(X, y)

PPt = PPtest(clf)

print(PPt.classifier.predict(X[10]))

joblib.dump(PPt, "/opt/predict/testobj.jl")

ppt2 = joblib.load("/opt/predict/testobj.jl")

print(ppt2.classifier.predict(X[10]))
"""
