#!/usr/bin/env python3
#-*-coding:utf-8-*-
'''
decision tree for CS536 HW2 (for question 6)
'''
import random
import numpy as np
import math
import copy
import sys
import matplotlib.pyplot as plt

script,log = sys.argv
f = open(log,'w')
__con__ = sys.stderr

class DescisonTree():
    '''  111  '''
    def __init__(self,m,k):
        self.m = m
        self.k = k
        #self.loop = 0
        self.tree = [None]*(2**self.k-1)
        self.label = [None]*(2**(self.k+1)-1)

    def GenerateDataPoint(self):
        '''generate data according to the given pattern '''
        #compute weight list
        w = []
        denominator = 0
        for i in range(2,self.k+1):
            denominator += 0.9**i
        for i in range(1,self.k+1):
            w_i = 0.9**i/denominator
            w.append(w_i)

        #compute data point
        matrix = np.zeros((self.m,self.k+1),'int')
        for i in range(self.m):
            value = 0
            matrix[i][0] = 1 if random.random()<0.5 else 0
            for j in range(1,self.k):
                matrix[i][j] = matrix[i][j-1] if random.random()<0.75 else 1-matrix[i][j-1]
                value += matrix[i][j]*w[j]
            matrix[i][self.k] = matrix[i][0] if value>=1/2 else 1-matrix[i][0]

        return matrix
        print(f'generate {self.m} data points successful ')

    def Partition(self,matrix,x):
        '''partition the data matrix based on x, return two matrixs according to value of x '''
        list_x_0 = []
        list_x_1 = []
        for i in range(matrix.shape[0]):
            if matrix[i][x] == 0:
                list_x_0.append(matrix[i])
            else:
                list_x_1.append(matrix[i])
        mat0 = np.array(list_x_0).reshape(-1,self.k+1)
        mat1 = np.array(list_x_1).reshape(-1,self.k+1)
        return mat0,mat1

    def Gini(self,matrix):
        '''compute the Gini impurity and return min Gini and its corresponding x '''

        list_gini = []
        for j in range(self.k):
            count_x_1 = 0    #compute sum of Xi=1
            count_x_1_y_1 = 0    #compute sum of yi=1 when Xi=1
            count_x_0_y_1 = 0    #compute sum of yi=1 when Xi=0
            for i in range(matrix.shape[0]):
                if matrix[i][j] == 1:
                    count_x_1 += 1
                    if matrix[i][self.k] == 1:
                        count_x_1_y_1 += 1
                else:
                    if matrix[i][self.k] == 1:
                        count_x_0_y_1 += 1

            p1 = count_x_1/matrix.shape[0]
            p0 = 1-p1
            p11 = count_x_1_y_1/count_x_1 if count_x_1 != 0 else 0
            p10 = 1-p11
            p01 = count_x_0_y_1/(matrix.shape[0]-count_x_1) if (matrix.shape[0]-count_x_1) != 0 else 0
            p00 = 1-p01
            #print(p1,p0,p11,p10,p01,p00)

            if p0==0 or p1==0:    #if last time use x1 to part, this time p0=0 or p1 =0 when computing based on x1
                continue
            gini = p0*(1-p00**2-p01**2)+p1*(1-p10**2-p11**2)

            list_gini.append([j,gini])
        #print(list_gini)
        #input()
        return min(list_gini, key=lambda x: x[1])

    def GenerateTree(self,matrix,id = 0):
        '''recursively building tree and label'''
        #print('id',id)
        count = 0
        for i in range(matrix.shape[0]):
            if matrix[i][-1]==0:
                count += 1

        #print('bizhi',count/matrix.shape[0])
        #input()
        if count/matrix.shape[0]>0.95 :
            self.label[id] = 0
            return
        if count/matrix.shape[0]<0.05 :
            self.label[id] = 1
            return

        pos, gini = self.Gini(matrix)
        #print('pos&gini',pos,gini)
        #input()
        self.tree[id] = pos+1
        mat0, mat1 = self.Partition(matrix, pos)
        self.GenerateTree(mat0,2*id+1)
        #print('now id', id)
        self.GenerateTree(mat1,2*id+2)

    def GetError(self, matrix):
        '''get error of a built tree  '''
        if matrix.shape[1]-1 != self.k:
            print('depth of matrix does not comply with depth of the built tree')
            return 0

        right = 0
        for i in range(matrix.shape[0]):
            pos = 0
            id = self.tree[pos]
            while id:
                if matrix[i][id-1] == 0:
                    pos = 2*pos+1
                else:
                    pos = 2*pos+2
                if pos > len(self.tree):
                    id = None
                    break
                id = self.tree[pos]
            predict = self.label[pos]
            if predict == matrix[i][-1]:
                right += 1

        return 1-right/matrix.shape[0]

if __name__ == '__main__':
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

            diff = list_for_average.append(abs(error_train-error_test))

        average = sum(list_for_average)/50
        list_for_diff.append(average)

    plt.plot(list_for_loops, list_for_diff, color='blue', linewidth=2.0, linestyle='-')
    plt.title('Question6')
    plt.xlabel('number of data for training')
    plt.ylabel('abs(error_train-error_test)')
    plt.savefig("difference for Q6")
    plt.show()
    print(list_for_diff)
    print('finish')
