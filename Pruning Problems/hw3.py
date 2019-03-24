#!/usr/bin/env python3
#-*-coding:utf-8-*-
'''
pruning decision tree for CS536 HW3
'''
import numpy as np
import random
import math
import sys

script,log = sys.argv
f = open(log,'w')
__con__ = sys.stderr

class DecisionTree():
    '''  111  '''
    def __init__(self,m):
        self.m = m
        self.k = 21
        #self.loop = 0
        self.tree = [None]*(2**self.k-1)
        self.label = [None]*(2**(self.k+1)-1)

    def GenerateDataPoint(self):
        '''generate data according to the given pattern '''
        #compute data point
        matrix = np.zeros((self.m,self.k+1),'int')
        for i in range(self.m):
            count1_x_1 = 0
            count2_x_1 = 0
            matrix[i][0] = 1 if random.random()<0.5 else 0
            for j in range(1,8):
                matrix[i][j] = matrix[i][j-1] if random.random()<0.75 else 1-matrix[i][j-1]
                count1_x_1 += 1 if matrix[i][j]==1 else 0
            for j in range(8,15):
                matrix[i][j] = matrix[i][j-1] if random.random()<0.75 else 1-matrix[i][j-1]
                count2_x_1 += 1 if matrix[i][j]==1 else 0
            if matrix[i][0] == 0:
                matrix[i][self.k] = 1 if count1_x_1 > 3 else 0
            else:
                matrix[i][self.k] = 1 if count2_x_1 > 3 else 0

            for j in range(15,21):
                matrix[i][j] = 1 if random.random()<0.5 else 0

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

    def InformationGain(self,matrix):
        '''compute the IGs and return max IG and its corresponding x '''
        #compute H(Y)
        count_y = 0
        for i in range(matrix.shape[0]):
            count_y = count_y+1 if matrix[i][self.k] == 1 else count_y
        p_y = count_y/matrix.shape[0]
        H_y = -p_y*math.log(p_y,2)-(1-p_y)*math.log((1-p_y),2)

        #compute each H(Y/Xi)
        list_IG = []
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

            H_yx = p1*(-p11*math.log((p11+0.0000001),2)-p10*math.log((p10+0.0000001),2)) + p0*(-p01*math.log((p01+0.0000001),2)-p00*math.log((p00+0.0000001),2))
            IG = H_y-H_yx
            list_IG.append([j,IG])
        #print(list_IG)
        return max(list_IG, key=lambda x: x[1])

    def GenerateTree(self,matrix,id = 0):
        '''recursively building tree and label'''
        count = 0
        for i in range(matrix.shape[0]):
            if matrix[i][-1]==0:
                count += 1

        if count/matrix.shape[0]>0.95 :
            self.label[id] = 0
            return
        if count/matrix.shape[0]<0.05 :
            self.label[id] = 1
            return

        pos, IG = self.InformationGain(matrix)
        #print('pos&IG',pos,IG)
        #input()
        self.tree[id] = pos    # in HW3, X begins at 0
        mat0, mat1 = self.Partition(matrix, pos)
        self.GenerateTree(mat0,2*id+1)
        #print('now id', id)
        self.GenerateTree(mat1,2*id+2)

    def GenerateTree_Depth(self,matrix,d,id=0):
        '''recursively building tree and label and ceasing at depth of d'''
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

        if id > 2**d-2:

            self.label[id] = 0 if count>=matrix.shape[0]-count else 1
            return

        pos, IG = self.InformationGain(matrix)
        #print('pos&IG',pos,IG)
        #input()
        self.tree[id] = pos    # in HW3, X begins at 0
        mat0, mat1 = self.Partition(matrix, pos)

        self.GenerateTree_Depth(mat0,d,2*id+1)
        #print('now id', id)
        self.GenerateTree_Depth(mat1,d,2*id+2)

    def GenerateTree_SampleSize(self,matrix,size,id=0):
        '''recursively building tree and label and ceasing at the node of controled sample size'''
        count = 0
        for i in range(matrix.shape[0]):
            if matrix[i][-1]==0:
                count += 1
        #print('bizhi',count/matrix.shape[0])
        #input()
        if count/matrix.shape[0]>0.99 :
            self.label[id] = 0
            return
        if count/matrix.shape[0]<0.01 :
            self.label[id] = 1
            return

        #print(matrix.shape)
        #print(size)
        #input()
        if matrix.shape[0] <= size:
            #print('reach the smaplesize control at ',id)
            self.label[id] = 0 if count>=matrix.shape[0]-count else 1
            return

        pos, IG = self.InformationGain(matrix)
        #print('pos&IG',pos,IG)
        #input()
        self.tree[id] = pos    # in HW3, X begins at 0
        mat0, mat1 = self.Partition(matrix, pos)
        #print(mat0.shape,mat1.shape)
        self.GenerateTree_SampleSize(mat0,size,2*id+1)
        #print('now id', id)
        self.GenerateTree_SampleSize(mat1,size,2*id+2)

    def GenerateTree_Independence(self,matrix,im,id=0):
        '''recursively building tree and label and ceasing at the node where the data are independent'''
        count = 0
        for i in range(matrix.shape[0]):
            if matrix[i][-1]==0:
                count += 1
        #print('bizhi',count/matrix.shape[0])
        #input()
        if count/matrix.shape[0]>0.99 :
            self.label[id] = 0
            return
        if count/matrix.shape[0]<0.01 :
            self.label[id] = 1
            return

        pos, IG = self.InformationGain(matrix)
        #print('pos&IG',pos,IG)
        #input()

        # compute indenpence
        x0_y0 = 0
        x0_y1 = 0
        x1_y0 = 0
        x1_y1 = 0
        n = matrix.shape[0]
        for i in range(n):
            if matrix[i][pos]==0 and matrix[i][-1]==0:
                x0_y0 += 1
            if matrix[i][pos]==0 and matrix[i][-1]==1:
                x0_y1 += 1
            if matrix[i][pos]==1 and matrix[i][-1]==0:
                x1_y0 += 1
            if matrix[i][pos]==1 and matrix[i][-1]==1:
                x1_y1 += 1

        p_x0 = (x0_y0+x0_y1)/n
        p_x1 = (x1_y0+x1_y1)/n
        p_y0 = (x0_y0+x1_y0)/n
        p_y1 = (x0_y1+x1_y1)/n

        T = (p_x0*p_y0*n-x0_y0)**2/p_x0*p_y0*n+ \
            (p_x0*p_y1*n-x0_y1)**2/p_x0*p_y1*n+ \
            (p_x1*p_y0*n-x1_y0)**2/p_x1*p_y0*n+ \
            (p_x1*p_y1*n-x1_y1)**2/p_x1*p_y1*n
        #print(p_x0,p_x1,p_y0,p_y1)
        #print(T)
        if T < im:
            return

        self.tree[id] = pos    # in HW3, X begins at 0
        mat0, mat1 = self.Partition(matrix, pos)
        self.GenerateTree_Independence(mat0,im,2*id+1)
        #print('now id', id)
        self.GenerateTree_Independence(mat1,im,2*id+2)

    def GetError(self, matrix):
        '''get error of a built tree  '''
        if matrix.shape[1]-1 != self.k:
            print('depth of matrix does not comply with depth of the built tree')
            return 0

        right = 0
        for i in range(matrix.shape[0]):
            pos = 0
            id = self.tree[pos]
            while id!=None:    #### attention here id could equal 0 !!!!!!!!!!!!!!!
                if matrix[i][id] == 0:
                    pos = 2*pos+1
                else:
                    pos = 2*pos+2
                if pos > len(self.tree):
                    id = None
                    break
                id = self.tree[pos]
                #print('pos',pos,'id',id)
                #input()
            predict = self.label[pos]
            if predict == matrix[i][-1]:
                right += 1

        return 1-right/matrix.shape[0]

    def GetNumberOfIrrelevant(self):
        '''get the number of irrevelant variables that included in decision tree '''
        count_irr = 0
        for i in range(len(self.tree)):
            if self.tree[i] != None:
                count_irr += 1 if self.tree[i] > 14 else 0
        return count_irr

if __name__ == '__main__':
    print('DecisionTree module for 536 HW3',file = sys.stderr)
    decision = DecisionTree(10)
    matrix = decision.GenerateDataPoint()
    print(matrix)
    decision.GenerateTree_Inpendence(matrix,0,1)
    print(decision.tree[:10])
    print(decision.label[:10])
