__author__ = 'sean'

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from mut_group_pred_pack import *

mut_list = get_full_mut_list()
pten_mutations = MutationGroup(mut_list)
fliplist = []
for i in pten_mutations.category_2_list:
    if i == 0:
        fliplist.append(1)
    else:
        fliplist.append(0)

fliplist = np.array(fliplist)

X, y = pten_mutations.scaled_vector_array, fliplist

n_samples, n_features = X.shape

# classification and ROC analysis

# run classifier with cross-validation and plot ROC curves

cv = StratifiedKFold(y, n_folds=6, shuffle=True)
classifier = svm.SVC(kernel='rbf', C=600, gamma=0.001, probability=True, \
                                                            random_state=0)

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
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

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