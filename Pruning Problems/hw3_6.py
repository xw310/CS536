#!/usr/bin/env python3
#-*-coding:utf-8-*-
'''
pruning decision tree for CS536 HW3 question2
'''
from hw3 import *
import matplotlib.pyplot as plt

sample_size_threshold = 5
list_for_loops = list(range(10,10010,200))
list_for_count = []

for loop in list_for_loops:
    list_for_average = []
    for i in range(30):
        decision_tree = DecisionTree(loop)
        original = decision_tree.GenerateDataPoint()

        # use the sample size threshold we got in hw3_3b
        decision_tree.GenerateTree_SampleSize(original,sample_size_threshold)

        count_irr = decision_tree.GetNumberOfIrrelevant()

        list_for_average.append(count_irr)

    average_count = sum(list_for_average)/30
    list_for_count.append(average_count)

plt.plot(list_for_loops, list_for_count, color='blue', linewidth=2.0, linestyle='-')
plt.title('Question6')
plt.xlabel('number of data for training')
plt.ylabel('number of irrevelants in the tree')
plt.savefig("Q6")
plt.show()

print('finish')
