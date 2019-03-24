#!/usr/bin/env python3
#-*-coding:utf-8-*-
'''
Perceptron for CS536 HW4 question2
'''
from hw4 import *
import matplotlib.pyplot as plt

perceptron = Perceptron(100,20)
matrix = perceptron.GenerateDataPoint(1)
step = perceptron.Train(matrix)
#print(perceptron.weight)

plt.plot(range(21), perceptron.weight, color='blue', linewidth=2.0, linestyle='-')
plt.title('Question2')
plt.xlabel('x')
plt.ylabel('weights corresponding to x')
plt.savefig("Q2")
plt.show()
