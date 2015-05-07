#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
roc_crossval.py

Uses binary classifiers (category splits 2 and 22) to generate a Receiver
Operating Characteristic (ROC) curve with calculated area under the curve (AUC).

usage: roc_crossval.py
"""

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from mut_group_pred_pack import *
from cross_val import get_x_y_scorer_folds
from plot_pca import plot_pdf


def get_colors_labels(num_cats):
    colors = {
        2: '#027878',
        22: '#f69568',
        3: ('#027878', '#f69568', '#e05254'),
        33: ('#027878', '#f69568', '#e05254'),
        4: ('#027878', '#f69568', '#fdc865', '#e05254'),
    }[num_cats]
    labels = {
        2: 'Null / Pathogenic',
        22: 'Null+Autism / Pathogenic',
        3: ('Null', 'Autism', 'Pathogenic'),
        33: ('Null', 'Autism+Somatic', 'PHTS'),
        4: ('Null', 'Autism', 'Somatic', 'PHTS'),
    }[num_cats]
    return colors, labels


def plot_roc_figure(pten_mutations, num_cats):
    param_grid = [
    {'C': [.1, 1, 10, 100, 1000 ],
     'gamma': [0.1, 0.01, 0.001, 0.0001]},
    ]

    X, y, scorer, folds = get_x_y_scorer_folds(pten_mutations, num_cats)
    cv = StratifiedKFold(y, n_folds=6, shuffle=True, random_state=0)
    svc = svm.SVC(kernel='rbf', cache_size=2000, class_weight='auto',
                  probability=True)
    classifier = GridSearchCV(svc, param_grid=param_grid, scoring=scorer,
                              cv=folds, n_jobs=-1)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0,1,100)

    color, label = get_colors_labels(num_cats)

    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        # plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i,
        # roc_auc))

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color,
             label='ROC %s (%0.2f)' % (label, mean_auc), lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")


if __name__ == "__main__":

    classes = [2, 22]

    mut_list = get_full_mut_list()
    pten_mutations = MutationGroup(mut_list)

    plt.figure(figsize=(8, 6))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    for num_cat in classes:

        plot_roc_figure(pten_mutations, num_cat)

    plot_pdf("catsplitroc2{}.pdf".format(2))

    plt.show()