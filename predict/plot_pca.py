#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
plot_pca.py

Shrinks vectors down to two dimensions and fits to the SVC. Displays
predictions using pyplot.

usage: plot_pca.py
"""

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

from cross_val import get_x_y_scorer_folds
from mut_group_pred_pack import get_full_mut_list, MutationGroup, \
    stratified_tt_split
from sklearn.cross_validation import StratifiedShuffleSplit


def get_contour_colors(num_cats):
    color = {
        2: ('#5abdbd', '#e05254'),
        22: ('#5abdbd', '#e05254'),
        3: ('#5abdbd', '#f69568', '#e05254'),
        33: ('#5abdbd', '#f69568', '#e05254'),
        4: ('#5abdbd', '#f69568', '#fdc865', '#e05254'),
    }[num_cats]
    return color

def get_scatter_colors_labels(num_cats):
    colors = {
        2: ('#027878', '#c22326'),
        22: ('#027878', '#c22326'),
        3: ('#027878', '#f69568', '#e05254'),
        33: ('#027878', '#f69568', '#e05254'),
        4: ('#027878', '#f69568', '#fdc865', '#e05254'),
    }[num_cats]
    labels = {
        2: ('Null', 'Pathogenic'),
        22: ('Null+Autism', 'Pathogenic'),
        3: ('Null', 'Autism', 'Pathogenic'),
        33: ('Null', 'Autism+Somatic', 'PHTS'),
        4: ('Null', 'Autism', 'Somatic', 'PHTS'),
    }[num_cats]
    return colors, labels

# #027878 blue light ##5abdbd
# #f37338 orange light #f69568
# #fdb632 yellow light #fdc865
# #c22326 red light #e05254

def plot_contour(num_cats, xx, yy, meshPredict):
    levels = [-.5, .5, 1.5, 2.5, 3.5]
    plt.contourf(xx, yy, meshPredict, levels, colors=get_contour_colors(
        num_cats))
    plt.contour(xx, yy, meshPredict, levels, colors=('k'))

def plot_scatter(num_cats, X, y):
    colors, labels = get_scatter_colors_labels(num_cats)
    for i in range(len(colors)):
        indices = np.where(y == i)
        plt.scatter(X[indices, 0], X[indices, 1], c=colors[i],
            label=labels[i])

def plot_subfigure(mut_group, num_cats, subplot, indices):
    X, y, scorer, folds = get_x_y_scorer_folds(mut_group, num_cats)

    X = PCA(n_components=2).fit_transform(X, y=y)


    param_grid = [
    {'C': [.1, 1, 10, 100, 1000 ],
     'gamma': [0.1, 0.01, 0.001, 0.0001]},
    ]

    svc = SVC(kernel='rbf', cache_size=2000, class_weight='auto')
    clf = GridSearchCV(svc, param_grid=param_grid, scoring=scorer, cv=folds)

    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    mesh_grid = np.c_[xx.ravel(), yy.ravel()]
    meshPredict = clf.fit(X, y).predict(mesh_grid)
    meshPredict = meshPredict.reshape(xx.shape)

    plt.subplot(1, 2, subplot)

    plot_contour(num_cats, xx, yy, meshPredict)

    plot_scatter(num_cats, X[indices], y[indices])

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks()
    plt.yticks()

    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.legend(loc="lower right")

def plot_pdf(filename):
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(filename)
    pp.savefig()
    pp.close()


if __name__ == "__main__":

    classes = [2, 22]

    plt.figure(figsize=(10, 5))

    pten_mutations = MutationGroup(get_full_mut_list())

    strat_split = StratifiedShuffleSplit(pten_mutations.category_22_list, 1,
                                         test_size=0.2, random_state=0)

    for train, test in strat_split:
        train = train
        test = test

    for i, num_cats in enumerate(classes):

        plot_subfigure(pten_mutations, num_cats, i, test)

    plot_pdf("catsplit{}-{}.pdf".format(classes[0], classes[1]))

    plt.show()
