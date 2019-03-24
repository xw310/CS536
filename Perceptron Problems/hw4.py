#!/usr/bin/env python3
#-*-coding:utf-8-*-
'''
Perceptron for CS536 HW4
'''
import numpy as np
import random
import copy
import sys
from sys import argv

script,log = argv
f = open(log,'w')
__con__ = sys.stderr
#sys.stderr = f

class Perceptron():
    '''Perceptron Algo '''

    def __init__(self,m,k):
        self.m = m
        self.k = k
        self.weight = [0]*(self.k+1)
        self.label = [None]*(self.k+1)

        #print('initialization successful',file=sys.stderr)

    def GenerateDataPoint(self,epsilon):
        '''generate data according to the given pattern '''
        #compute data point
        matrix = np.zeros((self.m,self.k+2))
        for i in range(self.m):
            for j in range(1,self.k):
                matrix[i][j] = np.random.normal(0,1)
            matrix[i][0] = 1
            D = np.random.exponential(1)
            matrix[i][self.k] = epsilon + D if random.random()>0.5 else -(epsilon+D)
            matrix[i][-1] = 1 if matrix[i][self.k]>0 else -1
        #print('generating data successfully',file=sys.stderr)
        return matrix

    def Train(self,matrix):
        '''Training process '''
        count = 0
        step = 0
        while count != self.m:
            count = 0
            for i in range(self.m):
                prediction = 1 if np.dot(matrix[i][:self.k+1],self.weight)>0 else -1
                if prediction == matrix[i][-1]:
                    count += 1
                else:
                    step += 1
                    self.weight += matrix[i][-1]*matrix[i][:self.k+1]

        return step

if __name__ == '__main__':
    perceptron = Perceptron(20,3)
    matrix = perceptron.GenerateDataPoint(1)
    print(matrix)
