import random
import sys
from sys import argv

#script,log = argv
#f = open(log,'a')
#__con__ = sys.stderr
#sys.stderr = f

n=100
L=10
theo_MSE_MOM = L**2/(3*n)
theo_MSE_MLE = 2*L**2/((n+1)*(n+2))
print(f'theoretical MSE of MOM is {theo_MSE_MOM}\n')
print(f'theoretical MSE of MLE is {theo_MSE_MLE}\n')

list_for_MOM = []
list_for_MLE = []
for i in range(1000):
    list_for_sample = []
    for j in range(100):
        sample = random.random()
        sample *= L
        list_for_sample.append(sample)
    L_MOM = sum(list_for_sample)/n*2
    L_MLE = max(list_for_sample)
    list_for_MOM.append(L_MOM)
    list_for_MLE.append(L_MLE)

# compute expectation of MSE_MOM
MSE_for_MOM = 0
for i in range(1000):
    MSE_for_MOM += (list_for_MOM[i]-L)**2
MSE_for_MOM /= 1000
print(f'estimated MSE of MOM is {MSE_for_MOM}\n')

# compute expectation of MSE_MLE
MSE_for_MLE = 0
for i in range(1000):
    MSE_for_MLE += (list_for_MLE[i]-L)**2
MSE_for_MLE /= 1000
print(f'estimated MSE of MOM is {MSE_for_MLE}\n')
