#!/usr/bin/env python3
#-*-coding:utf-8-*-
'''
pruning decision tree for CS536 HW3 question1
'''
from hw3 import *
import matplotlib.pyplot as plt

list_for_loops = list(range(10,10010,100))
list_for_diff = []

for loop in list_for_loops:
    list_for_average = []
    for i in range(50):
        decision_tree = DecisionTree(loop)
        original = decision_tree.GenerateDataPoint()

        decision_tree.GenerateTree(original)

        #generate new data points to get test error
        decision_tree.m = 5000
        new = decision_tree.GenerateDataPoint()

        error_test = decision_tree.GetError(new)

        list_for_average.append(error_test)

    average = sum(list_for_average)/50
    list_for_diff.append(average)

plt.plot(list_for_loops, list_for_diff, color='blue', linewidth=2.0, linestyle='-')
plt.title('Question5')
plt.xlabel('number of data for training')
plt.ylabel('error(f)')
plt.savefig("Q1")
plt.show()
#print(list_for_diff)
print('finish')
