__author__ = 'sean'

import numpy as np
from outputScores import *
from sklearn import preprocessing

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

includedScores = [
    #'numCancerMuts',
    #'avgSampleMutCount',
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
    categories for different splits
    """
    def __init__(self, mutList):
        listmut = []
        category4List = []
        category2List = []
        category2_2List = []
        category3List = []
        vectorList = []
        vectorLists = {
            "NULL": [],
            "AUTISM": [],
            "SOMATIC": [],
            "PHTS": []
        }
        conditionNumbers = {
            "NULL": 0,
            "AUTISM": 0,
            "SOMATIC": 0,
            "PHTS": 0
        }

        for mutDict in mutList:
            listmut.append(mutDict["wtres"] + str(mutDict["codon"]) + mutDict["mutres"])
            category = mutDict["category"]
            category4List.append(get_cat_4(category))
            category2List.append(get_cat_2(category))
            category2_2List.append(get_cat_22(category))
            category3List.append(get_cat_3(category))

            conditionNumbers[category] += 1

            vector = []
            for key in includedScores:
                vector.append(mutDict[key])
            vectorList.append(vector)
            vectorLists[category].append(vector)


        vectorArray = np.array(vectorList)

        scaler = preprocessing.StandardScaler().fit(vectorArray)

        vectorArray_scaled = scaler.transform(vectorArray)



        self.listmut = listmut
        self.scaler = scaler
        self.cat4list = np.array(category4List)
        self.cat2list = np.array(category2List)
        self.cat2_2list = np.array(category2_2List)
        self.cat3list = np.array(category3List)
        self.vectorList = vectorArray
        self.vectorArray_scaled = vectorArray_scaled
        self.conditionNumbers = conditionNumbers
        self.vectorLists = vectorLists

    def get_cat4array(self):
        return np.array(self.cat4list)

    def get_cat3array(self):
        return np.array(self.cat3list)

    def get_cat2array(self):
        return np.array(self.cat2list)

    def get_vectorArray(self):
        return np.array(self.vectorList)

    def makeMutTXT(self, filename):
        with open(filename, "w") as f:
            for mutation in self.listmut:
                f.write(mutation + "\n")




class PredictionPackage(object):
    def __init__(self, classifier, scaler, includedScores, numCats, categoryArray, vectorArray):
        self.classifier = classifier
        self.scaler = scaler
        self.numCats = numCats
        self.categoryArray = categoryArray
        self.vectorArray = vectorArray
        self.includedScores = includedScores # the list of scores used to create this classifier

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
        :param scores_needed: list of scores required for classifier
        :param scaler: scikit-learn scaler object used for classifier
        :return: scaled score vector
        """
        # make all score dictionaries
        score_vector = []
        mapp_dict = makeMAPPdict()
        sus_struc_dict = makeSuspectScoreDict(
            "/opt/predict/datafiles/SuspectStructure.csv"
        )
        sus_seq_dict = makeSuspectScoreDict(
            "/opt/predict/datafiles/SuspectSequence.csv"
        )
        ss_and_asa_dict = makeSSandASAdict()
        sub_ss_and_asa_dict = makeSubSS_ASAdict()
        pph2_dict = makePPh2Dict()
        varmod_dict = makeVarModDict()

        for score in self.includedScores:
            score_vector.append({
                'mutMAPPscore': getMAPPscore(mapp_dict,
                                             codon, mut_res),
                'wtMAPPscore': getMAPPscore(mapp_dict,
                                            codon, wt_res),
                'pph2_prob': getPPh2_prob(pph2_dict, wt_res,
                                          codon, mut_res, 'pph2_prob'),
                'pph2_FPR': getPPh2_prob(pph2_dict,
                                         wt_res, codon, mut_res, 'pph2_FPR'),
                'pph2_TPR': getPPh2_prob(pph2_dict,
                                         wt_res, codon, mut_res, 'pph2_TPR'),
                'pph2_FDR': getPPh2_prob(pph2_dict,
                                         wt_res, codon, mut_res, 'pph2_FDR'),
                'pph2_dScore': getPPh2_prob(pph2_dict,
                                            wt_res, codon, mut_res, 'dScore'),
                'SuspectScore': getSuspectScore(sus_struc_dict,
                                                sus_seq_dict, codon, mut_res),
                'VarModScore': getVarModScore(varmod_dict,
                                              wt_res, codon, mut_res,
                                              'VarMod Probability'),
                'VMinterface': getVarModScore(varmod_dict,
                                              wt_res, codon, mut_res,
                                              'Interface'),
                'VMconservation': getVarModScore(varmod_dict,
                                                 wt_res, codon, mut_res,
                                                 'Conservation'),
                'ASA': getASAscore(codon, ss_and_asa_dict, sub_ss_and_asa_dict),
                'helix': getSSscore(codon, ss_and_asa_dict, sub_ss_and_asa_dict, 'H'),
                '310helix': getSSscore(codon, ss_and_asa_dict, sub_ss_and_asa_dict, 'G'),
                'strand': getSSscore(codon, ss_and_asa_dict, sub_ss_and_asa_dict, 'E'),
                'turn': getSSscore(codon, ss_and_asa_dict, sub_ss_and_asa_dict, 'T'),
                'bridge': getSSscore(codon, ss_and_asa_dict, sub_ss_and_asa_dict, 'B'),
                'coil': getSSscore(codon, ss_and_asa_dict, sub_ss_and_asa_dict, 'C'),
                'phi': getPhiscore(codon, ss_and_asa_dict, sub_ss_and_asa_dict),
                'psi': getPsiscore(codon, ss_and_asa_dict, sub_ss_and_asa_dict),
            }[score])
        return self.scaler.transform(np.array(score_vector))