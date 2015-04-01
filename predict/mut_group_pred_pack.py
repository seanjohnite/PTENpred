"""
Helper file for classifyMutation.py
Stores definitions for PredictionPackage class and MutationGroup class
"""

__author__ = 'sean'

import numpy as np
from output_scores import *
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn import cross_validation
from sklearn.svm import SVC

allFields = [
    'category',
    'codon',
    'wtres',
    'mutres',
    'numCancerMuts',
    'avgSampleMutCount',
    'mutMAPPscore',
    'wtMAPPscore',
    'pph2_prob',
    'pph2_FPR',
    'pph2_TPR',
    'pph2_FDR',
    'pph2_dScore',
    'SuspectScore',
    'VarModScore',
    'VMinterface',
    'VMconservation',
    'ProveanScore',
    'SIFTscore',
    'ASA',
    'helix',
    '310helix',
    'strand',
    'turn',
    'bridge',
    'coil',
    'phi',
    'psi',
]

INCLUDED_SCORES = [
    # 'numCancerMuts',
    #'avgSampleMutCount',
    #'codon',
    'mutMAPPscore',
    'wtMAPPscore',
    'pph2_prob',
    'pph2_FPR',
    'pph2_TPR',
    'pph2_FDR',
    'pph2_dScore',
    'SuspectScore',
    'VarModScore',
    'VMinterface',
    'VMconservation',
    'ProveanScore',
    'SIFTscore',
    'ASA',
    'helix',
    '310helix',
    'strand',
    'turn',
    'bridge',
    'coil',
    'phi',
    'psi',
]


def get_cat_2(category):
    return {
        "SOMATIC": 1,
        "PHTS": 1,
        "AUTISM": 1,
        "NULL": 0
    }[category]


def get_cat_22(category):
    return {
        "SOMATIC": 1,
        "PHTS": 1,
        "AUTISM": 0,
        "NULL": 0
    }[category]


def get_cat_3(category):
    return {
        "SOMATIC": 2,
        "PHTS": 2,
        "AUTISM": 1,
        "NULL": 0
    }[category]

def get_cat_33(category):
    return {
        "SOMATIC": 1,
        "PHTS": 2,
        "AUTISM": 1,
        "NULL": 0
    }[category]


def get_cat_4(category):
    return {
        "SOMATIC": 2,
        "PHTS": 3,
        "AUTISM": 1,
        "NULL": 0
    }[category]


class MutationGroup(object):
    """
    Class for storing lists of mutations. Contains different listings of
    categories for different category splits, plus a scaled vector array of
    scores.
    Instantiation requires a list of "mutation dictionaries".
    Each mutation dictionary contains a number of keys from included_scores,
    and the key points to that associated score.
    """

    def __init__(self, mut_list):
        list_mut = []
        category_4_list = []
        category_2_list = []
        category_22_list = []
        category_3_list = []
        category_33_list = []
        vector_list = []

        for mut_dict in mut_list:
            list_mut.append("{}{}{}".format(mut_dict["wtres"],
                                            mut_dict["codon"],
                                            mut_dict["mutres"]))
            category = mut_dict["category"]
            category_4_list.append(get_cat_4(category))
            category_2_list.append(get_cat_2(category))
            category_22_list.append(get_cat_22(category))
            category_3_list.append(get_cat_3(category))
            category_33_list.append(get_cat_33(category))

            vector = []
            for key in INCLUDED_SCORES:
                vector.append(mut_dict[key])
            vector_list.append(vector)

        vector_array = np.array(vector_list)

        scaler = preprocessing.StandardScaler().fit(vector_array)

        scaled_vector_array = scaler.transform(vector_array)

        self.listmut = list_mut
        self.scaler = scaler
        self.category_4_list = np.array(category_4_list)
        self.category_2_list = np.array(category_2_list)
        self.category_22_list = np.array(category_22_list)
        self.category_3_list = np.array(category_3_list)
        self.category_33_list = np.array(category_33_list)
        self.scaled_vector_array = scaled_vector_array

    def get_cat4array(self):
        return np.array(self.category_4_list)

    def get_cat3array(self):
        return np.array(self.category_3_list)

    def get_cat2array(self):
        return np.array(self.category_2_list)

    def make_mut_txt(self, filename):
        with open(filename, "w") as f:
            for mutation in self.listmut:
                f.write(mutation + "\n")


class PredictionPackage(object):
    """
    Stores everything needed to predict a new score vector.
    Instantiation requires the category split (2, 22, 3, 4) and whether or not
    to use probabilities from sklearn.svm.SVC
    """

    def __init__(self, num_cats, probs, mut_list=None, test=False, **kwargs):
        if not mut_list:
            mut_list = get_full_mut_list()

        if test:
            classifier, scaler, category_array, scaled_vector_array, X_test, \
            y_test = \
                get_prediction_info(num_cats, probs, mut_list, test=True,
                                    **kwargs)
            self.X_test = X_test
            self.y_test = y_test
        else:
            classifier, scaler, category_array, scaled_vector_array = \
                get_prediction_info(num_cats, probs, mut_list, **kwargs)

        self.classifier = classifier
        self.scaler = scaler
        self.num_cats = num_cats
        self.category_array = category_array
        self.vector_array = scaled_vector_array
        self.included_scores = INCLUDED_SCORES

    def predict(self, score_vector):
        return self.classifier.predict(score_vector)

    def predict_proba(self, score_vector):
        return self.classifier.predict_proba(score_vector)

    def get_scores(self, wt_res, codon, mut_res):
        """
        Gets scaled scores for a specified variant
        :param wt_res: WT PTEN residue
        :param codon: PTEN amino acid position
        :param mut_res: variant
        :return: scaled score vector
        """
        # make all score dictionaries
        score_vector = []
        mapp_dict = make_mapp_dict()
        sus_struc_dict = make_suspect_score_dict(
            "/opt/predict/datafiles/SuspectStructure.csv"
        )
        sus_seq_dict = make_suspect_score_dict(
            "/opt/predict/datafiles/SuspectSequence.csv"
        )
        ss_and_asa_dict = make_ss_and_asa_dict()
        sub_ss_and_asa_dict = make_sub_ss_and_asa_dict()
        pph2_dict = make_pph2_dict()
        varmod_dict = make_var_mod_dict()
        provean_dict = make_provean_sift_dict()
        variant = "{}{}{}".format(wt_res, codon, mut_res)

        for score in self.included_scores:
            score_vector.append({
                'codon': codon,
                'mutMAPPscore': get_mapp_score(mapp_dict,
                                               codon, mut_res),
                'wtMAPPscore': get_mapp_score(mapp_dict,
                                              codon, wt_res),
                'pph2_prob': get_pph2_prob(pph2_dict, wt_res,
                                           codon, mut_res, 'pph2_prob'),
                'pph2_FPR': get_pph2_prob(pph2_dict,
                                          wt_res, codon, mut_res, 'pph2_FPR'),
                'pph2_TPR': get_pph2_prob(pph2_dict,
                                          wt_res, codon, mut_res, 'pph2_TPR'),
                'pph2_FDR': get_pph2_prob(pph2_dict,
                                          wt_res, codon, mut_res, 'pph2_FDR'),
                'pph2_dScore': get_pph2_prob(pph2_dict,
                                             wt_res, codon, mut_res, 'dScore'),
                'SuspectScore': get_suspect_score(sus_struc_dict,
                                                  sus_seq_dict, codon, mut_res),
                'VarModScore': get_var_mod_score(varmod_dict,
                                                 wt_res, codon, mut_res,
                                                 'VarMod Probability'),
                'VMinterface': get_var_mod_score(varmod_dict,
                                                 wt_res, codon, mut_res,
                                                 'Interface'),
                'VMconservation': get_var_mod_score(varmod_dict,
                                                    wt_res, codon, mut_res,
                                                    'Conservation'),
                'ProveanScore': get_provean_sift_score(provean_dict, variant,
                                                       'PSCORE'),
                'SIFTscore': get_provean_sift_score(provean_dict, variant,
                                                    'SSCORE'),
                'ASA': get_asa_score(codon, ss_and_asa_dict,
                                     sub_ss_and_asa_dict),
                'helix': get_ss_score(codon, ss_and_asa_dict,
                                      sub_ss_and_asa_dict, 'H'),
                '310helix': get_ss_score(codon, ss_and_asa_dict,
                                         sub_ss_and_asa_dict, 'G'),
                'strand': get_ss_score(codon, ss_and_asa_dict,
                                       sub_ss_and_asa_dict, 'E'),
                'turn': get_ss_score(codon, ss_and_asa_dict,
                                     sub_ss_and_asa_dict, 'T'),
                'bridge': get_ss_score(codon, ss_and_asa_dict,
                                       sub_ss_and_asa_dict, 'B'),
                'coil': get_ss_score(codon, ss_and_asa_dict,
                                     sub_ss_and_asa_dict, 'C'),
                'phi': get_phi_score(codon, ss_and_asa_dict,
                                     sub_ss_and_asa_dict),
                'psi': get_psi_score(codon, ss_and_asa_dict,
                                     sub_ss_and_asa_dict),
            }[score])
        return self.scaler.transform(np.array(score_vector))


def get_prediction_info(num_cats, probs, mut_list, test=False, **kwargs):
    """
    Returns prediction information from the list of known mutations
    :param num_cats: number of category split (2, 22, 3, 4)
    :param probs: whether to fit with probability option
    :return: prediction package object
    """

    pten_mutations = MutationGroup(mut_list)

    print(pten_mutations.category_2_list)
    print(pten_mutations.category_22_list)
    print(pten_mutations.category_3_list)
    print(pten_mutations.category_33_list)
    print(pten_mutations.category_4_list)

    cat_array = {
        2: pten_mutations.category_2_list,
        22: pten_mutations.category_22_list,
        3: pten_mutations.category_3_list,
        33: pten_mutations.category_33_list,
        4: pten_mutations.category_4_list,
    }[num_cats]

    scaled_vector_array = pten_mutations.scaled_vector_array

    if test:
        clf, X_test, y_test = get_classifier(scaled_vector_array, cat_array, num_cats,
                              probs, test=True, **kwargs)
        return clf, pten_mutations.scaler, cat_array, scaled_vector_array, \
               X_test, y_test
    else:
        clf = get_classifier(scaled_vector_array,
                         cat_array, num_cats, probs, **kwargs)
        return clf, pten_mutations.scaler, cat_array, scaled_vector_array


def get_classifier(vector_array, cat_array, num_cats, probs, test=False,
                   **kwargs):
    """
    Function to get a cross-validated grid-searched classifier
        for training data
    :param vector_array: numpy array of score vectors
    :param cat_array: numpy array of classes of score vectors
    :param num_cats: category split (2, 22, 3, 4)
    :param probs: boolean, whether to use probabilities in fit
    :return: cross-validated and grid searched classifier
    """
    if test:
        X_train, y_train, X_test, y_test = stratified_tt_split(
            vector_array, cat_array, **kwargs
        )
    else:
        X_train, y_train = vector_array, cat_array

    scorer = {
        2: make_scorer(f1_score, pos_label=0),
        22: make_scorer(f1_score, pos_label=0),
        3: make_scorer(f1_score),
        33: make_scorer(f1_score),
        4: make_scorer(f1_score)
    }[num_cats]

    folds = {
        2: 10,
        22: 10,
        33: 10,
        3: 10,
        4: 10
    }[num_cats]

    param_grid = [
        {'C': [100, 200, 300, 400, 600, 800, 1000],
         'gamma': [0.001, 0.0008, 0.0006, 0.0004, 0.0002,  0.0001]},
    ]

    cv = cross_validation.StratifiedKFold(y_train, folds, shuffle=True)

    clf = GridSearchCV(SVC(cache_size=2000,
                           class_weight='auto',
                           probability=probs),
                       param_grid, scoring=scorer, n_jobs=-1, cv=cv)

    clf = clf.fit(X_train, y_train)

    if test:
        return clf, X_test, y_test
    else:
        return clf

def stratified_tt_split(vectors, labels, **kwargs):
    split1 = cross_validation.StratifiedShuffleSplit(
        labels, 1, **kwargs
    )
    for split in split1:
        testIndex = split[0]
        trainIndex = split[1]

    return vectors[trainIndex], labels[trainIndex], vectors[testIndex], labels[testIndex]