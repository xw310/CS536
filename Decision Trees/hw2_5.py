#!/usr/bin/env python3
#-*-coding:utf-8-*-
'''
decision tree for CS536 HW2 (for questions 5)
'''
import random
import numpy as np
import math
import copy
import sys
from hw2 import *
import matplotlib.pyplot as plt

list_for_loops = list(range(100,5001,100))
list_for_diff = []

for loop in list_for_loops:
    list_for_average = []
    for i in range(50):
        decision_tree = DescisonTree(loop,10)
        original = decision_tree.GenerateDataPoint()

        decision_tree.GenerateTree(original)

        error_train = decision_tree.GetError(original)

        #generate new data points to get test error
        decision_tree.m = 5000
        new = decision_tree.GenerateDataPoint()

        error_test = decision_tree.GetError(new)

        list_for_average.append(abs(error_train-error_test))

    average = sum(list_for_average)/50
    list_for_diff.append(average)

plt.plot(list_for_loops, list_for_diff, color='blue', linewidth=2.0, linestyle='-')
plt.title('Question5')
plt.xlabel('number of data for training')
plt.ylabel('abs(error_train-error_test)')
plt.savefig("difference for Q5")
plt.show()
print(list_for_diff)
print('finish')
