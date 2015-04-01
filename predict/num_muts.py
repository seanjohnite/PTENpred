__author__ = 'sean'

from mut_group_pred_pack import *
from output_scores import *

mut_list = get_full_mut_list()

pten_mutations = MutationGroup(mut_list)

counter = {
    0: 0,
    1: 0,
    2: 0,
    3: 0
}


for class_ in pten_mutations.category_4_list:
    counter[class_] += 1

print(counter)