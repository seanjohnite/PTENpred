__author__ = 'sean'

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, make_scorer
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from numpy import mean, std, array
from scipy.stats import f_oneway

from mut_group_pred_pack import MutationGroup, get_full_mut_list, stratified_tt_split



pten_mutations = MutationGroup(get_full_mut_list())

X, y = pten_mutations.scaled_vector_array, pten_mutations.category_2_list


scores = []

for i in range(10):

    cv = StratifiedKFold(y, 6, shuffle=True, random_state=0)


    param_grid = [
        {'C': [.1, 1, 10, 100, 1000 ],
         'gamma': [0.1, 0.01, 0.001, 0.0001]},
    ]

    f1pos0 = make_scorer(f1_score, pos_label=0)

    svc = SVC(kernel='rbf', cache_size=2000, class_weight='auto')

    clf = GridSearchCV(svc, param_grid=param_grid, scoring=f1pos0, cv=4)



    cv_scores = cross_val_score(clf, X, y, cv=cv)

    scores.append(cv_scores)

    print(cv_scores)
    print(mean(cv_scores))
    print(std(cv_scores))

scores = array(scores)

f, p = f_oneway(*scores)

print(f)
print(p)



"""
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
"""

if __name__ == "main":
    # Main script
    pten_mutations = MutationGroup(get_full_mut_list())

    X, y = pten_mutations.scaled_vector_array, pten_mutations.category_2_list
