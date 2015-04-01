__author__ = 'sean'

from sklearn.externals import joblib
from sklearn.metrics import f1_score, make_scorer, classification_report, confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from makeIndexVectors import *
from outputScores import *
from SVMhelpers import *

mutList = startMutList()

mutList = addMAPPscores(mutList)

mutList = addSSandASAscores(mutList)

mutList = addPPh2scores(mutList)

mutList = addSuspectScores(mutList)

mutList = addVarModScores(mutList)

PTENmutations = MutationGroup(mutList)

h = .02  # step size in the mesh





Cats = [ # all categorizations of the vectors
    PTENmutations.cat2list,
    PTENmutations.cat2_2list,
    PTENmutations.cat3list,
    PTENmutations.cat4list,
]

titles = [
    'Null and Pathogenic',
    'Null+Autism and Pathogenic',
    'Null, Autism, and Pathogenic',
    'Null, Autism, Somatic, and PHTS'
]
plt.figure(1, (6,8))
for i, cat in enumerate(Cats):
    plt.subplot(2, 2, i + 1)

    X, y = PTENmutations.vectorArray_scaled, cat

    clf = SVC(kernel='rbf', gamma=0.001, C=200, cache_size=2000,
              class_weight='auto')

    X = PCA(n_components=2).fit_transform(X, y=y)

    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    mesh_grid = np.c_[xx.ravel(), yy.ravel()]

    meshPredict = clf.predict(mesh_grid)

    print meshPredict

    meshPredict = meshPredict.reshape(xx.shape)

    print meshPredict

    plt.contourf(xx, yy, meshPredict, cmap='hsv')

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='hsv', alpha=0.6)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()