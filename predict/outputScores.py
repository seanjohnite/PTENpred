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
import logging



#logging.basicConfig(filename='/opt/predict/logs/SVM.log',level=logging.DEBUG)

def makeMAPPdict():
    MAPPlist = []
    with open("/opt/predict/datafiles/MAPP_output.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            MAPPlist.append(row)
    return MAPPlist


def getMAPPscore(MAPPdict, codon, residue):
    """
    MAPPdict is a list of dictionaries (0 is the same as 1)

    :param MAPPdict:
    :param codon:
    :param residue:
    :return:
    """
    if MAPPdict[codon][residue] == "N/A":
        return 5.0
    else:
        return float(MAPPdict[codon][residue])

def startMutList():
    with open("/opt/predict/datafiles/CollatedMuts2.csv", "r") as f:
        reader = csv.DictReader(f)
        mutList = []
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
            mutList.append(rowdict)
    return mutList

def addMAPPscores(mutList):
    MAPPdict = makeMAPPdict()
    for mutDict in mutList:
        if getMAPPscore(MAPPdict, int(mutDict["codon"]), mutDict["mutres"]) == "N/A":
            mutDict["mutMAPPscore"] = 5.
        else:
            mutDict["mutMAPPscore"] = float(getMAPPscore(MAPPdict, int(mutDict["codon"]), mutDict["mutres"]))
        if getMAPPscore(MAPPdict, int(mutDict["codon"]), mutDict["wtres"]) == "N/A":
            mutDict["wtMAPPscore"] = 5.
        else:
            mutDict["wtMAPPscore"] = float(getMAPPscore(MAPPdict, int(mutDict["codon"]), mutDict["wtres"]))

    return mutList

def makePPh2Dict():
    with open("/opt/predict/datafiles/pph2-full3.csv", "r") as f:
        reader = csv.DictReader(f)
        PPh2Dict = {}
        for row in reader:
            variant = row["aa1"] + row["pos"] + row["aa2"]
            PPh2Dict[variant] = row
    return PPh2Dict

def addPPh2scores(mutList):
    PPh2Dict = makePPh2Dict()
    for mutDict in mutList:
        variant = mutDict["wtres"] + str(mutDict["codon"]) + mutDict["mutres"]
        mutDict["pph2_prob"] = float(PPh2Dict[variant]["pph2_prob"])
        mutDict["pph2_FPR"] = float(PPh2Dict[variant]["pph2_FPR"])
        mutDict["pph2_TPR"] = float(PPh2Dict[variant]["pph2_TPR"])
        mutDict["pph2_FDR"] = float(PPh2Dict[variant]["pph2_FDR"])
        mutDict["pph2_dScore"] = float(PPh2Dict[variant]["dScore"])
    return mutList

def getPPh2_prob(PPh2dict, wtres, codon, mutres, scoreString):
    variant = "{}{}{}".format(wtres, codon, mutres)
    return float(PPh2dict[variant][scoreString])

def makeSSandASAdict():
    with open("/opt/predict/datafiles/SSandASA.csv", "r") as f:
        reader = csv.DictReader(f)
        SSandASAdict = {}
        for row in reader:
            codon = int(row["codon"])
            SSandASAdict[codon] = row
    return SSandASAdict

def makeSubSS_ASAdict():
    with open("/opt/predict/datafiles/SpineXASASSP.csv") as f:
        reader = csv.DictReader(f)
        SpineXdict = {}
        for row in reader:
            codon = int(row["#"])
            SpineXdict[codon] = row
    return SpineXdict

def getASAscore(codon, SSandASAdict, SpineXdict):
    try:
        score = float(SSandASAdict[codon]["ASA"])
    except KeyError:
        #logging.info("Codon " + str(codon) + " was not found in SSandASA dictionary\n Using Spine...")
        score = float(SpineXdict[codon]["ASA"])
    return score

def getPhiscore(codon, SSandASAdict, SpineXdict):
    try:
        score = float(SSandASAdict[codon]["phi"])
    except KeyError:
        #logging.info("Codon " + str(codon) + " was not found in SSandASA dictionary\n Using Spine...")
        score = float(SpineXdict[codon]["Phi"])
    return score

def getPsiscore(codon, SSandASAdict, SpineXdict):
    try:
        score = float(SSandASAdict[codon]["psi"])
    except KeyError:
        #logging.info("Codon " + str(codon) + " was not found in SSandASA dictionary\n Using Spine...")
        score = float(SpineXdict[codon]["Psi"])
    return score

def getSSscore(codon, SSandASAdict, SpineXdict, SSabbrev):
    try:
        if SSandASAdict[codon]["DSSP"] == SSabbrev:
            score = 1
        else:
            score = 0
    except KeyError:
        #logging.info("Codon " + str(codon) + " was not found in SSandASA dictionary\n Using Spine...")
        if SpineXdict[codon]["SS"] == SSabbrev:
            score = 1
        else:
            score = 0
    return score

def addSSandASAscores(mutList):
    SSandASAdict = makeSSandASAdict()
    SpineXdict = makeSubSS_ASAdict()
    for mutDict in mutList:
        codon = int(mutDict["codon"])
        mutDict["helix"] = 0.
        mutDict["310helix"] = 0.
        mutDict["coil"] = 0.
        mutDict["bridge"] = 0.
        mutDict["turn"] = 0.
        mutDict["strand"] = 0.
        try:
            if SSandASAdict[codon]["DSSP"] == "H":
                mutDict["helix"] = 1.
            elif SSandASAdict[codon]["DSSP"] == "G":
                mutDict["310helix"] = 1.
            elif SSandASAdict[codon]["DSSP"] == "B":
                mutDict["bridge"] = 1.
            elif SSandASAdict[codon]["DSSP"] == "T":
                mutDict["turn"] = 1.
            elif SSandASAdict[codon]["DSSP"] == "E":
                mutDict["strand"] = 1.
            else:
                mutDict["coil"] = 1.
            mutDict["ASA"] = float(SSandASAdict[codon]["ASA"])
            mutDict["phi"] = float(SSandASAdict[codon]["phi"])
            mutDict["psi"] = float(SSandASAdict[codon]["psi"])
        except KeyError:
            #logging.info("Codon " + str(codon) + " was not found in SSandASA dictionary")
            if SpineXdict[codon]["SS"] == "H":
                mutDict["helix"] = 1.
            elif SpineXdict[codon]["SS"] == "G":
                mutDict["310helix"] = 1.
            elif SpineXdict[codon]["SS"] == "B":
                mutDict["bridge"] = 1.
            elif SpineXdict[codon]["SS"] == "T":
                mutDict["turn"] = 1.
            elif SpineXdict[codon]["SS"] == "E":
                mutDict["strand"] = 1.
            else:
                mutDict["coil"] = 1.
            mutDict["ASA"] = float(SpineXdict[codon]["ASA"])
            mutDict["phi"] = float(SpineXdict[codon]["Phi"])
            mutDict["psi"] = float(SpineXdict[codon]["Psi"])
    return mutList

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

def addSuspectScores(mutList):
    susStrDict = makeSuspectScoreDict("/opt/predict/datafiles/SuspectStructure.csv")
    susSeqDict = makeSuspectScoreDict("/opt/predict/datafiles/SuspectSequence.csv")
    for mutDict in mutList:
        codon = int(mutDict["codon"])
        try:
            mutDict["SuspectScore"] = float(susStrDict[codon][mutDict["mutres"]])
        except KeyError:
            mutDict["SuspectScore"] = float(susSeqDict[codon][mutDict["mutres"]])
    return mutList

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


def addVarModScores(mutList):
    varModDict = makeVarModDict()
    for mutDict in mutList:
        variant = mutDict["wtres"] + str(mutDict["codon"]) + mutDict["mutres"]
        mutDict["VarModScore"] = float(varModDict[variant]['VarMod Probability'])
        mutDict["VMinterface"] = float(varModDict[variant]['Interface'])
        mutDict["VMconservation"] = float(varModDict[variant]['Conservation'])
    return mutList

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
    mutList = startMutList()
    mutList = addMAPPscores(mutList)
    mutList = addSSandASAscores(mutList)
    mutList = addPPh2scores(mutList)
    mutList = addSuspectScores(mutList)
    mutList = addVarModScores(mutList)

    return mutList
