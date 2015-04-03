__author__ = 'sean'

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from mut_group_pred_pack import *
from cross_val import get_x_y_scorer_folds
from plot_pca import plot_pdf
from roc_crossval import plot_roc_figure


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


def plot_polyphen_roc(num_cats):

    color = 'g'
    label = 'polyphen'

    pph2_dict = make_pph2_dict()

    X, y, scorer, folds = get_x_y_scorer_folds(pten_mutations, num_cats)

    cv = StratifiedKFold(y, n_folds=6, shuffle=True, random_state=0)

    # hacky way to make pph2 array
    mut_list = start_mut_list()
    pph2_array = []
    for mut_dict in mut_list:
        wtres = mut_dict["wtres"]
        codon = mut_dict["codon"]
        mutres = mut_dict["mutres"]
        pph2prob = get_pph2_prob(pph2_dict, wtres, codon, mutres, "pph2_prob")
        pph2_array.append(pph2prob)
    pph2_array = np.array(pph2_array)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0,1,100)

    for i, (train, test) in enumerate(cv):
        probas_ = pph2_array[test]
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_)
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

def plot_provean_roc(num_cats):

    color = 'b'
    label = 'Provean'

    provean_dict = make_provean_sift_dict()

    X, y, scorer, folds = get_x_y_scorer_folds(pten_mutations, num_cats)

    cv = StratifiedKFold(y, n_folds=6, shuffle=True, random_state=0)

    # hacky way to make pph2 array
    mut_list = start_mut_list()
    provean_array = []
    for mut_dict in mut_list:
        variant = "{}{}{}".format(mut_dict["wtres"], mut_dict["codon"],
                                  mut_dict["mutres"])
        provean_score = get_provean_sift_score(provean_dict, variant, 'PSCORE')
        provean_array.append(provean_score)
    provean_array = np.array(provean_array)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0,1,100)

    for i, (train, test) in enumerate(cv):
        probas_ = provean_array[test]
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_)
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

def plot_varmod_roc(num_cats):

    color = 'r'
    label = 'Varmod'

    varmod_dict = make_var_mod_dict()

    X, y, scorer, folds = get_x_y_scorer_folds(pten_mutations, num_cats)

    cv = StratifiedKFold(y, n_folds=6, shuffle=True, random_state=0)

    # hacky way to make pph2 array
    mut_list = start_mut_list()
    varmod_array = []
    for mut_dict in mut_list:
        wtres = mut_dict["wtres"]
        codon = mut_dict["codon"]
        mutres = mut_dict["mutres"]
        varmod_score = get_var_mod_score(varmod_dict, wtres, codon, mutres,
                                         'VarMod Probability')
        varmod_array.append(varmod_score)
    varmod_array = np.array(varmod_array)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0,1,100)

    for i, (train, test) in enumerate(cv):
        probas_ = varmod_array[test]
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_)
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

    mut_list = get_full_mut_list()
    pten_mutations = MutationGroup(mut_list)

    plt.figure(figsize=(8, 6))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    plot_roc_figure(pten_mutations, 2)
    plot_polyphen_roc(2)
    plot_provean_roc(2)
    plot_varmod_roc(2)

    plot_pdf("pphenroc4.pdf")

    plt.show()