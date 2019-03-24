#!/usr/bin/env python3
#-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
X = np.c_[(1, 1), (-1, -1), (-1, 1), (1, -1)].T
y = [-1, -1, 1, 1]

poly_svc = svm.SVC(kernel='poly', degree=2, coef0=1).fit(X, y)
Gaussian_svc = svm.SVC(kernel='rbf').fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

titles = ['SVC with polynomial (degree 2) kernel','SVC with Gaussian kernel']

for i, clf in enumerate([poly_svc,Gaussian_svc]):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])


    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

    plt.show()
