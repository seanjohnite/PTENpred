__author__ = 'sean'

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from mut_group_pred_pack import *
from cross_val import get_x_y_scorer_folds

mut_list = get_full_mut_list()
pten_mutations = MutationGroup(mut_list)

X, y, scorer, folds = get_x_y_scorer_folds(pten_mutations, 2)

n_samples, n_features = X.shape

# classification and ROC analysis

# run classifier with cross-validation and plot ROC curves

param_grid = [
    {'C': [.1, 1, 10, 100, 1000 ],
     'gamma': [0.1, 0.01, 0.001, 0.0001]},
]

cv = StratifiedKFold(y, n_folds=6, shuffle=True, random_state=0)
svc = svm.SVC(kernel='rbf', cache_size=2000, class_weight='auto',
              probability=True)
classifier = GridSearchCV(svc, param_grid=param_grid, scoring=scorer,
                          cv=folds, n_jobs=-1)

mean_tpr = 0.0
mean_fpr = np.linspace(0,1,100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    #plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()