__author__ = 'sean'

from mut_group_pred_pack import PredictionPackage, get_full_mut_list
import pprint
from sklearn.metrics import classification_report

mut_list = get_full_mut_list()

new_list = []




for mut_dict in mut_list:
    if mut_dict['category'] != 'SOMATIC':
        new_list.append(mut_dict)

def make_score_dict(ppack):
    grid = ppack.classifier.param_grid[0]
    score_dict = {}

    for param in grid.keys():
        score_dict[param] = {}
        for num in grid[param]:
            score_dict[param][num] = []
            score_dict[param][num].append(0)
            score_dict[param][num].append([])
    return score_dict







def average(ls):
    sum = 0
    count = 0
    for num in ls:
        sum += num
        count += 1
    if count == 0:
        return 0
    else:
        return sum / count

# test_size is actually training set size ratio
p4scores = PredictionPackage(33, False, test=True, test_size=0.6)
score_dict = make_score_dict(p4scores)

for i in range(10):
    new_pack = PredictionPackage(33, False, test=True, test_size=0.6)
    params = new_pack.classifier.best_params_
    score = new_pack.classifier.best_score_
    for param in params.keys():
        score_dict[param][params[param]][0] += 1
        score_dict[param][params[param]][1].append(score)

    print("Best params are {}, \n best score is {}".format(
        new_pack.classifier.best_params_, new_pack.classifier.best_score_))
    print("Detailed classification report:")
    print
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print
    y_true, y_pred = \
        new_pack.y_test, new_pack.classifier.predict(new_pack.X_test)
    print(classification_report(y_true, y_pred))
    print
for param in score_dict.keys():
    for num in score_dict[param].keys():
        if len(score_dict[param][num][1]) == 0:
            pass
        else:
            score_dict[param][num][1] = average(score_dict[param][num][1])


pprint.pprint(score_dict)


"""
print score_dict
for i in range(len(score_dict['C'])):
    print("C={} gamma={} score={} C/gamma={}".format(score_dict['C'][i],
                                          score_dict['gamma'][i],
                                          score_dict['score'][i],
                                          score_dict['C'][i]/score_dict[
                                              'gamma'][i]))





    print("Grid scores on development set:")
    print
    for params, mean_score, scores in new_pack.classifier.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))

"""