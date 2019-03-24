#!/usr/bin/env python3
#-*-coding:utf-8-*-
'''
Perceptron for CS536 HW4 question4
'''
from hw4 import *
import matplotlib.pyplot as plt

list_of_k = list(range(2,41,1))
list_of_step = []

for k in list_of_k:
    list_for_average = []
    for i in range(50):
        perceptron = Perceptron(1000,k)
        matrix = perceptron.GenerateDataPoint(1)
        step = perceptron.Train(matrix)
        list_for_average.append(step)
    average = sum(list_for_average)/50
    list_of_step.append(average)

plt.plot(list_of_k, list_of_step, color='blue', linewidth=2.0, linestyle='-')
plt.title('Question4_1 m=1000')
plt.xlabel('varying of k')
plt.ylabel('changing of average step')
plt.savefig("Q4_2")
plt.show()
