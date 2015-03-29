"""
Score outputter. Stores scores in a list of dictionaries with these keys:
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
    'VarmodScore',
    'ASA',
    'helix',
    '310helix',
    'strand',
    'turn',
    'bridge',
    'coil',
    'phi',
    'psi'


"""
__author__ = 'sean'


import csv



def make_mapp_dict():
    mapp_list = []
    with open("/opt/predict/datafiles/MAPP_output.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapp_list.append(row)
    return mapp_list


def get_mapp_score(mapp_dict, codon, residue):
    """
    mapp_dict is a list of dictionaries (0 is the same as 1)

    :param mapp_dict:
    :param codon:
    :param residue:
    :return:
    """
    if mapp_dict[codon][residue] == "N/A":
        return 5.0
    else:
        return float(mapp_dict[codon][residue])

def start_mut_list():
    with open("/opt/predict/datafiles/CollatedMuts2.csv", "r") as f:
        reader = csv.DictReader(f)
        mut_list = []
        for row in reader:
            rowdict = {}
            rowdict["codon"] = int(row["Codon"])
            rowdict["wtres"] = row["OrigAA"].upper()
            rowdict["mutres"] = row["MutAA"].upper()
            if row["type"] == "NULL":
                rowdict["category"] = "NULL"
            elif row["type"] == "Autism or developmental delay":
                rowdict["category"] = "AUTISM"
            elif row["type"] == "PHTS":
                rowdict["category"] = "PHTS"
            else:
                rowdict["category"] = "SOMATIC"
            if row["AvgsampleMutCount"] == "":
                if row["type"] == "NULL":
                    rowdict["avgSampleMutCount"] = 4.
                    rowdict["numCancerMuts"] = 1.
                elif row["type"] == "Autism or developmental delay":
                    rowdict["avgSampleMutCount"] = 1.
                    rowdict["numCancerMuts"] = 1.
                elif row["type"] == "PHTS":
                    rowdict["avgSampleMutCount"] = 1.
                    rowdict["numCancerMuts"] = 1.
            else:
                rowdict["avgSampleMutCount"] = float(row["AvgsampleMutCount"])
                rowdict["numCancerMuts"] = float(row["Count"])
            mut_list.append(rowdict)
    return mut_list

def add_mapp_scores(mut_list):
    mapp_dict = make_mapp_dict()
    for mut_dict in mut_list:
            mut_dict["mutMAPPscore"] = get_mapp_score(mapp_dict,
                                                     int(mut_dict["codon"]),
                                                     mut_dict["mutres"])
            mut_dict["wtMAPPscore"] = get_mapp_score(mapp_dict,
                                                    int(mut_dict["codon"]),
                                                    mut_dict["wtres"])
    return mut_list

def make_pph2_dict():
    with open("/opt/predict/datafiles/pph2-full3.csv", "r") as f:
        reader = csv.DictReader(f)
        pph2_dict = {}
        for row in reader:
            variant = row["aa1"] + row["pos"] + row["aa2"]
            pph2_dict[variant] = row
    return pph2_dict

def add_pph2_scores(mut_list):
    pph2_dict = make_pph2_dict()
    for mut_dict in mut_list:
        variant = mut_dict["wtres"] + str(mut_dict["codon"]) + mut_dict["mutres"]
        mut_dict["pph2_prob"] = float(pph2_dict[variant]["pph2_prob"])
        mut_dict["pph2_FPR"] = float(pph2_dict[variant]["pph2_FPR"])
        mut_dict["pph2_TPR"] = float(pph2_dict[variant]["pph2_TPR"])
        mut_dict["pph2_FDR"] = float(pph2_dict[variant]["pph2_FDR"])
        mut_dict["pph2_dScore"] = float(pph2_dict[variant]["dScore"])
    return mut_list

def get_pph2_prob(pph2_dict, wtres, codon, mutres, scoreString):
    variant = "{}{}{}".format(wtres, codon, mutres)
    return float(pph2_dict[variant][scoreString])

def make_ss_and_asa_dict():
    with open("/opt/predict/datafiles/SSandASA.csv", "r") as f:
        reader = csv.DictReader(f)
        ss_and_asa_dict = {}
        for row in reader:
            codon = int(row["codon"])
            ss_and_asa_dict[codon] = row
    return ss_and_asa_dict

def make_sub_ss_and_asa_dict():
    with open("/opt/predict/datafiles/SpineXASASSP.csv") as f:
        reader = csv.DictReader(f)
        spine_x_dict = {}
        for row in reader:
            codon = int(row["#"])
            spine_x_dict[codon] = row
    return spine_x_dict

def get_asa_score(codon, ss_and_asa_dict, spine_x_dict):
    try:
        score = float(ss_and_asa_dict[codon]["ASA"])
    except KeyError:
        score = float(spine_x_dict[codon]["ASA"])
    return score

def get_phi_score(codon, ss_and_asa_dict, spine_x_dict):
    try:
        score = float(ss_and_asa_dict[codon]["phi"])
    except KeyError:
        score = float(spine_x_dict[codon]["Phi"])
    return score

def get_psi_score(codon, ss_and_asa_dict, spine_x_dict):
    try:
        score = float(ss_and_asa_dict[codon]["psi"])
    except KeyError:
        score = float(spine_x_dict[codon]["Psi"])
    return score

def get_ss_score(codon, ss_and_asa_dict, spine_x_dict, ss_abbrev):
    try:
        if ss_and_asa_dict[codon]["DSSP"] == ss_abbrev:
            score = 1
        else:
            score = 0
    except KeyError:
        if spine_x_dict[codon]["SS"] == ss_abbrev:
            score = 1
        else:
            score = 0
    return score

def add_ss_and_asa_scores(mut_list):
    ss_and_asa_dict = make_ss_and_asa_dict()
    spine_x_dict = make_sub_ss_and_asa_dict()
    for mut_dict in mut_list:
        codon = int(mut_dict["codon"])
        mut_dict["helix"] = get_ss_score(codon, ss_and_asa_dict,
                                         spine_x_dict, 'H')
        mut_dict["310helix"] = get_ss_score(codon, ss_and_asa_dict,
                                            spine_x_dict, 'G')
        mut_dict["coil"] = get_ss_score(codon, ss_and_asa_dict,
                                        spine_x_dict, 'C')
        mut_dict["bridge"] = get_ss_score(codon, ss_and_asa_dict,
                                          spine_x_dict, 'B') 
        mut_dict["turn"] = get_ss_score(codon, ss_and_asa_dict,
                                        spine_x_dict, 'T')
        mut_dict["strand"] = get_ss_score(codon, ss_and_asa_dict,
                                          spine_x_dict, 'E')
        mut_dict["ASA"] = get_asa_score(codon, ss_and_asa_dict, spine_x_dict)
        mut_dict["phi"] = get_phi_score(codon, ss_and_asa_dict, spine_x_dict)
        mut_dict["psi"] = get_psi_score(codon, ss_and_asa_dict, spine_x_dict)
    return mut_list

def makeSuspectScoreDict(filename):
    with open(filename) as f:
        reader = csv.DictReader(f)
        suspectDict = {}
        for row in reader:
            codon = int(row["codon"])
            suspectDict[codon] = row
    return suspectDict

def getSuspectScore(susStrDict, susSeqDict, codon, mutRes):
    try:
        score = float(susStrDict[codon][mutRes])
    except KeyError:
        #logging.info("Codon " + str(codon) + " was not found in SuspectStructure dictionary")
        score = float(susSeqDict[codon][mutRes])
    return score

def addSuspectScores(mut_list):
    susStrDict = makeSuspectScoreDict("/opt/predict/datafiles/SuspectStructure.csv")
    susSeqDict = makeSuspectScoreDict("/opt/predict/datafiles/SuspectSequence.csv")
    for mut_dict in mut_list:
        codon = int(mut_dict["codon"])
        try:
            mut_dict["SuspectScore"] = float(susStrDict[codon][mut_dict["mutres"]])
        except KeyError:
            mut_dict["SuspectScore"] = float(susSeqDict[codon][mut_dict["mutres"]])
    return mut_list

def makeVarModDict():
    with open("/opt/predict/datafiles/VarmodResults2.csv") as f:
        varModDict = {}
        reader = csv.DictReader(f)
        for row in reader:
            variant = row["Variant"]
            varModDict[variant] = row
    return varModDict

def getVarModScore(VarModDict, wtRes, codon, mutRes, scoreString):
    variant = "{}{}{}".format(wtRes, codon, mutRes)
    return float(VarModDict[variant][scoreString])


def addVarModScores(mut_list):
    varModDict = makeVarModDict()
    for mut_dict in mut_list:
        variant = mut_dict["wtres"] + str(mut_dict["codon"]) + mut_dict["mutres"]
        mut_dict["VarModScore"] = float(varModDict[variant]['VarMod Probability'])
        mut_dict["VMinterface"] = float(varModDict[variant]['Interface'])
        mut_dict["VMconservation"] = float(varModDict[variant]['Conservation'])
    return mut_list

def makeProveanSiftDict():
    with open("/opt/predict/datafiles/ProveanResults.csv") as f:
        proveanDict = {}
        reader = csv.DictReader(f)
        for row in reader:
            variant = "{}{}{}".format(row["RESIDUE_REF"], row["POSITION"], row["RESIDUE_ALT"])
            proveanDict[variant] = row
    return proveanDict

def getProveanSiftScore(proveanDict, wtRes, codon, mutRes, scoreString):
    variant = "{}{}{}".format(wtRes, codon, mutRes)
    return float(proveanDict[variant][scoreString])

def get_full_mut_list():
    mut_list = start_mut_list()
    mut_list = add_mapp_scores(mut_list)
    mut_list = add_ss_and_asa_scores(mut_list)
    mut_list = add_pph2_scores(mut_list)
    mut_list = addSuspectScores(mut_list)
    mut_list = addVarModScores(mut_list)

    return mut_list
