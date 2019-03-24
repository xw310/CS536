#!/usr/bin/env python3
#-*-coding:utf-8-*-
'''
Perceptron for CS536 HW4 question4
'''
from hw4 import *
import matplotlib.pyplot as plt

#generating data
m = 100
k = 2
matrix = np.zeros((m, k+2))
weights = [0]*(k+1)
for i in range(m):
    for j in range(1,k+1):
        matrix[i][j] = np.random.normal(0,1)
    matrix[i][0] = 1
    matrix[i][-1] = 1 if matrix[i][1]**2+matrix[i][2]**2>k else -1

list = list(range(500))
list_of_b = []
list_of_w1 = []
list_of_w2 = []

loop = 0    #counting the loop
mistakes = 1    #count the mistakes in each loop
step = 0    #count the numbers of changing of weights
while mistakes:
    print(loop)
    if loop > 500 and mistakes>m/20:
        print('data not separable')
        break
    mistake = 0
    loop += 1
    for i in range(m):
        prediction = 1 if np.dot(matrix[i][:k+1],weights)>0 else -1
        if prediction == matrix[i][-1]:
            pass
        else:
            mistakes += 1
            step += 1
            weights += matrix[i][-1]*matrix[i][:k+1]
            list_of_b.append(weights[0])
            list_of_w1.append(weights[1])
            list_of_w2.append(weights[2])

plt.figure(1)
plt.plot(range(len(list_of_b))[:500], list_of_b[:500], color='blue', linewidth=2.0, linestyle='-')
plt.title('Question5 changing of bias')
plt.xlabel('varying of bias')
plt.ylabel('step')
plt.savefig("Q5_bias")

plt.figure(2)
plt.plot(range(len(list_of_w1))[:500], list_of_w1[:500], color='red', linewidth=2.0, linestyle='-')
plt.title('Question5 changing of w1')
plt.xlabel('varying of w1')
plt.ylabel('step')
plt.savefig("Q5_w1")

plt.figure(3)
plt.plot(range(len(list_of_w2))[:500], list_of_w2[:500], color='red', linewidth=2.0, linestyle='-')
plt.title('Question5 changing of w2')
plt.xlabel('varying of w2')
plt.ylabel('step')
plt.savefig("Q5_w2")

plt.show()
