#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cross_val.py

Runs through grid cross-validation and gives output representing accuracy
scores for every possible class split. Modify output using "cycles" in the
main script.

usage: cross_val.py

"""


from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, make_scorer
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from numpy import mean, std
from scipy.stats import f_oneway
from sklearn.decomposition import PCA

from mut_group_pred_pack import MutationGroup, get_full_mut_list
import warnings

warnings.simplefilter("ignore")


def get_x_y_scorer_folds(mut_group, num_cats):
    vectors = mut_group.scaled_vector_array
    labels = {
        2: mut_group.category_2_list,
        22: mut_group.category_22_list,
        3: mut_group.category_3_list,
        33: mut_group.category_33_list,
        4: mut_group.category_4_list
    }[num_cats]
    scorer = {
        2: make_scorer(f1_score, pos_label=0),
        22: make_scorer(f1_score, pos_label=0),
        3: make_scorer(f1_score),
        33: make_scorer(f1_score),
        4: make_scorer(f1_score),
    }[num_cats]
    folds = {
        2: 8,
        22: 8,
        3: 8,
        33: 8,
        4: 8,
    }[num_cats]
    return vectors, labels, scorer, folds

def output(string, f):
    print(string)
    f.write("{}\n".format(string))

def end_cv_run_output(i, cycles, cv_scores, f):
    output("\nCross-validation scores for {}/{} run of {} category "
          "split are {}".format(i+1, cycles, num_cats, cv_scores), f)
    output("Mean is {}".format(mean(cv_scores)), f)
    output("Std is {}".format(std(cv_scores)), f)

def end_cv_run_set(scores, fi):
    f, p = f_oneway(*scores)
    output("\nCross-validation finished for category split {}".format(
        num_cats), fi)
    output("One-way ANOVA results:", fi)
    output("F = {}".format(f), fi)
    output("p = {}".format(p), fi)


if __name__ == "__main__":
    # Main script
    pten_mutations = MutationGroup(get_full_mut_list())

    num_cats_list = [2, 22, 3, 33, 4]

    param_grid = [
        {'C': [.1, 1, 10, 100, 1000 ],
         'gamma': [0.1, 0.01, 0.001, 0.0001]},
    ]

    cycles = 10

    with open("datafiles/CV Score Results.txt", "w") as f:
        output("----Starting CV for all categories----", f)

        for num_cats in num_cats_list:
            scores = []
            X, y, scorer, folds = get_x_y_scorer_folds(pten_mutations, num_cats)

            output("\n\n      --  CV for {}  --".format(num_cats), f)
            for i in range(cycles):
                svc = SVC(kernel='rbf', cache_size=2000, class_weight='auto')

                clf = GridSearchCV(svc, param_grid=param_grid,
                                   scoring=scorer, cv=folds, n_jobs=-1)

                cv = StratifiedKFold(y, 6, shuffle=True)

                cv_scores = [
                    clf.fit(X[train], y[train]).score(X[test], y[test])
                    for train, test in cv
                ]

                scores.append(cv_scores)

                end_cv_run_output(i, cycles, cv_scores, f)
            end_cv_run_set(scores, f)
        output("\n\nCross-validation completed for all categories", f)




'''
param_grid = [
    {'C': [.1, 1, 10, 100, 1000 ],
     'gamma': [0.1, 0.01, 0.001, 0.0001]},
]

f1pos0 = make_scorer(f1_score, pos_label=0)

svc = SVC(kernel='rbf', cache_size=2000, class_weight='auto')
cv = StratifiedKFold(y, 5, shuffle=True, random_state=0)
clf = GridSearchCV(svc, param_grid=param_grid, scoring=f1pos0)


print([clf.fit(X[train], y[train]).score(X[test], y[test]) for train, test in
       cv])
'''
