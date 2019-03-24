#!/usr/bin/env python3
#-*-coding:utf-8-*-
'''
Perceptron for CS536 HW6
'''

import numpy as np
import random
import matplotlib.pyplot as plt

class regression():
    ''' '''
    def __init__(self,m=200,w=1,b=5,err=0.1**0.5):
        self.m = m
        self.w = w
        self.b = b
        self.err = err
        self.data = []

    def generate_data(self):
        for i in range(self.m):
            x = random.uniform(100,102)
            err = self.err*np.random.randn()
            y = self.w*x + self.b + err
            self.data.append([x,y])

    def recenter(self):
        for i in range(self.m):
            self.data[i][0] -= 101

    def compute(self):
        x_sum = 0
        x_squre_sum = 0
        y_sum = 0
        xy_sum = 0
        for i in range(self.m):
            x_sum += self.data[i][0]
            x_squre_sum += self.data[i][0]**2
            y_sum += self.data[i][1]
            xy_sum += self.data[i][0]*self.data[i][1]

        w_r = (self.m*xy_sum - x_sum*y_sum)/(self.m*x_squre_sum - x_sum*x_sum)
        b_r = (y_sum - w_r*x_sum)/self.m
        return w_r, b_r

if __name__ == '__main__':

    loop = 1000
    list_of_w = []
    list_of_b = []
    list_of_w1 = []
    list_of_b1 = []
    for i in range(loop):
        reg = regression()
        reg.generate_data()
        w,b = reg.compute()
        list_of_w.append(w)
        list_of_b.append(b)

        reg.recenter()
        w1,b1 = reg.compute()
        list_of_w1.append(w1)
        list_of_b1.append(b1)

    plt.plot(range(loop)[:100],list_of_w[:100],color='blue', linewidth=1.0, linestyle='-',label="w before recenter")
    plt.plot(range(loop)[:100],list_of_w1[:100],color='red', linewidth=2.0, linestyle=':',label="w after recenter")
    plt.legend(loc='upper right')
    plt.title('w before recentering and after')
    plt.xlabel('loop')
    plt.ylabel('computed w')
    plt.savefig("hw6_w")
    plt.show()

    plt.plot(range(loop)[:100],list_of_b[:100],color='blue', linewidth=1.0, linestyle='-',label="b before recenter")
    plt.legend(loc='upper right')
    plt.title('b before recentering')
    plt.xlabel('loop')
    plt.ylabel('computed b')
    plt.savefig("hw6_b")
    plt.show()

    plt.plot(range(loop)[:100],list_of_b1[:100],color='red', linewidth=1.0, linestyle='-',label="b after recenter")
    plt.legend(loc='upper right')
    plt.title('b after recentering')
    plt.xlabel('loop')
    plt.ylabel('computed b')
    plt.savefig("hw6_b1")
    plt.show()
    # compute the expectation and var
    w_e = sum(list_of_w)/loop
    w1_e = sum(list_of_w1)/loop
    b_e = sum(list_of_b)/loop
    b1_e = sum(list_of_b1)/loop

    sum_w = 0
    sum_w1 = 0
    sum_b = 0
    sum_b1 = 0
    for i in range(loop):
        sum_w += (list_of_w[i]-w_e)**2
        sum_w1 += (list_of_w1[i]-w1_e)**2
        sum_b += (list_of_b[i]-b_e)**2
        sum_b1 += (list_of_b1[i]-b1_e)**2

    var_w = sum_w/loop
    var_w1 = sum_w1/loop
    var_b = sum_b/loop
    var_b1 = sum_b1/loop

    print(f'before recentering\nE[w] = {w_e}, Var(w) = {var_w}\nE[b] = {b_e}, Var(b) = {var_b}\n\n')
    print(f'after recentering\nE[w] = {w1_e}, Var(w) = {var_w1}\nE[b] = {b1_e}, Var(b) = {var_b1}')
