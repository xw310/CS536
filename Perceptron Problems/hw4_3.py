#!/usr/bin/env python3
#-*-coding:utf-8-*-
'''
Perceptron for CS536 HW4 question2
'''
from hw4 import *
import matplotlib.pyplot as plt

list_of_epsilon = list(np.arange(0,1,0.05))
list_of_step = []

for epsilon in list_of_epsilon:
    list_for_average = []
    for i in range(50):
        perceptron = Perceptron(100,20)
        matrix = perceptron.GenerateDataPoint(epsilon)
        step = perceptron.Train(matrix)
        list_for_average.append(step)
    average = sum(list_for_average)/50
    list_of_step.append(average)

plt.plot(list_of_epsilon, list_of_step, color='blue', linewidth=2.0, linestyle='-')
plt.title('Question3')
plt.xlabel('varying of epsilon')
plt.ylabel('changing of average step')
plt.savefig("Q3")
plt.show()
