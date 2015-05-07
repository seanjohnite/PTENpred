#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
classifyMutation.py

Takes in a mutation, classifies it using a stored sklearn classifier (or
optionally creates a new one), and outputs the category the classifier
believes it belongs to.

usage: classifyMutation.py [-h] [-c {2,22,3,4}] [-pp PREDPACK]
                           [-mk {True,False}] [-pr {True,False}]
                           variant

positional arguments:
  variant               the variant you want to check

optional arguments:
  -h, --help            show this help message and exit
  -c {2,22,3,4}, --cats {2,22,3,4}
                        number of categories to fit to
  -mk {True,False}, --mkpack {True,False}
                        remake prediction pack?
  -pr {True,False}, --probs {True,False}
                        use probabilities?
"""

import argparse
from sklearn.externals import joblib
from mut_group_pred_pack import PredictionPackage
import getpass


def load_pred_pack(filename):
    """
    :param filename: string of location of a PredictionPackage object,
    serialized with joblib
    :return: the indicated PredictionPackage object
    """
    return joblib.load(filename)


def get_variant_info(variant):
    """
    :param variant:
    :return: WT residue, codon, Variant residue
    """
    codon = int(filter(str.isdigit, variant))
    wtres = variant[0]
    mutres = variant[-1]

    return wtres, codon, mutres


def get_category_name(class_, num_cats):
    """
    Returns the name of the category
    :param class_: classified number (0, 1, 2, 3)
    :param num_cats: number of category split (2, 22, 3, 4)
    :return:
    """
    return {
        2: ["null", "pathogenic"],
        22: ["null or autism", "pathogenic"],
        3: ["null", "autism", "pathogenic"],
        4: ["null", "autism", "somatic", "PHTS"]
    }[num_cats][class_]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', help='the variant you want to check')
    parser.add_argument('-c', '--cats',
                        choices=[2, 22, 3, 4],
                        default=2,
                        help='number of categories to fit to',
                        type=int,
    )
    parser.add_argument('-mk', '--mkpack',
                        choices=[True, False],
                        default=False,
                        help='remake prediction pack?',
                        type=bool
    )
    parser.add_argument('-pr', '--probs',
                        choices=[True, False],
                        default=False,
                        help='use probabilities?',
                        type=bool
    )

    return parser.parse_args()


def main():
    # parse arguments from command line
    args = parse_arguments()

    num_cats = args.cats  # which category split to use (2, 22, 3, 4)
    variant = args.variant  # variant to test/predict
    make_new_pack = args.mkpack  # whether to recollect and refit the data
    probs = args.probs

    # -------------------Which Prediction Package to Use?-------------------- #
    # checks for probability feature (scores are not very accurate)
    prob = {
        True: "pr",
        False: ""
    }
    # www-data user has slightly different stored objects
    if getpass.getuser() == "www-data":
        wwwString = "/www"
    else:
        wwwString = ""

    if make_new_pack:  # new pack is requested
        pred_pack = PredictionPackage(num_cats, probs)
        joblib.dump(
            pred_pack,
            "datafiles/joblib{}/defaultPack{}{}.jl".format(
                wwwString, num_cats, prob[probs]
            )
        )
    else:  # using old prediction package
        pred_pack = load_pred_pack(
            "datafiles/joblib{}/defaultPack{}{}.jl".format(
                wwwString, num_cats, prob[probs]
            )
        )

    # ------------- Prediction package is stored in pred_pack --------------- #

    # ------------- Get scores and scale scores for new variant ------------- #
    # getting variant information
    wtRes, codon, mutRes = get_variant_info(variant)

    # get score vector and scale score vector with scaler stored in pred_pack
    scaledScoreVector = pred_pack.get_scores(wtRes, codon, mutRes)

    class_ = pred_pack.predict(scaledScoreVector)

    print("------Prediction information for variant {}-------".format(variant))

    print("Possible classes are {}".format({
        2: "null and pathogenic.",
        22: "null+autism and pathogenic.",
        3: "null, autism, and pathogenic.",
        4: "null, autism, somatic, and PHTS."
    }[num_cats]))

    print("Predicted class is {}.".format(get_category_name(class_, num_cats)))

    if num_cats == 22:  # catch weird categorization class type
        num = 2
    else:
        num = num_cats

    if probs:
        probabilities = pred_pack.predict_proba(scaledScoreVector)
        for i in range(num):
            print("Probability of class {} is {:.2f}.".format(
                get_category_name(i, num_cats), probabilities[0][i]))


if __name__ == "__main__":
    main()