#!/usr/bin/env python3
#-*-coding:utf-8-*-
'''
pruning decision tree for CS536 HW3 question3a
'''
from hw3 import *
import matplotlib.pyplot as plt

list_for_loops = list(range(0,21))    #list of depth
list_for_train_error = []
list_for_test_error = []

for loop in list_for_loops:
    list_for_train_average = []
    list_for_test_average = []
    for i in range(50):
        decision_tree = DecisionTree(8000)
        original = decision_tree.GenerateDataPoint()

        decision_tree.GenerateTree_Depth(original,loop)

        error_train = decision_tree.GetError(original)
        list_for_train_average.append(error_train)

        #generate new data points to get test error
        decision_tree.m = 2000
        new = decision_tree.GenerateDataPoint()

        error_test = decision_tree.GetError(new)
        list_for_test_average.append(error_test)

    train_average = sum(list_for_train_average)/50
    list_for_train_error.append(train_average)
    test_average = sum(list_for_test_average)/50
    list_for_test_error.append(test_average)

plt.plot(list_for_loops, list_for_train_error, color='red', linewidth=2.0, linestyle='-')
plt.plot(list_for_loops, list_for_test_error, color='blue', linewidth=2.0, linestyle='-')
plt.title('Question3a')
plt.xlabel('depth of decision tree')
plt.ylabel('error')
plt.savefig("Q3a")
plt.show()

print('finish')
