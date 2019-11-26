# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:54:53 2019

@author: Vrutik
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 21:33:00 2019

@author: Vrutik
"""
import numpy as np
import pandas as pd
from rmse import RMSE,R2_SCORE
from dataplot import DataPlot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class L1PolyGradientDescent:

    def __init__(self,degree):
        self.dataset = pd.read_csv('./data/normalized_dataset.csv')
        X = self.dataset.iloc[:, 2:4].values
        #X = self.dataset.iloc[:, 3].values
        Y = self.dataset.iloc[:, 4].values
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.3, random_state=9,shuffle=True)
        self.X0_train=list(self.X_train[:,0])
        self.X1_train=list(self.X_train[:,1])
        self.X0_test=list(self.X_test[:,0])
        self.X1_test=list(self.X_test[:,1])
        self.Y_test=list(self.Y_test)
        self.Y_train=list(self.Y_train)
        self.w=[]
        self.terms=[]
        self.degree=degree
        self.alph=[0,0.00000425,0.000003,0.000003,0.000003,0.000002,0.000002]
        self.alpha=self.alph[self.degree]
        self.la=0.001


    def sumOfError(self):
        retw=[]
        for i in range(len(self.w)):
            retw.append(0)

        for i in range(len(self.X0_train)):
            ss=-self.Y_train[i]
            j=0
            for k in self.terms:
               ss+=self.w[j]*(pow(1,k[0])*pow(self.X0_train[i],k[1])*pow(self.X1_train[i],k[2]))
               j+=1

            for h in range(len(retw)):
                k=self.terms[h]
                retw[h]+=ss*(pow(1,k[0])*pow(self.X0_train[i],k[1])*pow(self.X1_train[i],k[2]))
        for h in range(len(retw)):
            retw[h]+=(self.la*np.sign(self.w[h]))

        return retw


    def trainModel(self):
        for j in range(10):
            retw=self.sumOfError()
            print(j)
            for i in range(len(self.w)):
                self.w[i]=self.w[i]-(self.alpha*retw[i])


    def getPredictedValues(self):
        Y_pred=[]
        for i in range(len(self.X0_test)):
            ans=0
            for j in range(len(self.w)):
                k=self.terms[j]
                ans+=(self.w[j])*(pow(1,k[0])*pow(self.X0_test[i],k[1])*pow(self.X1_test[i],k[2]))
            Y_pred.append(ans)
        return Y_pred


    def poly_features(self):
        n=self.degree
        cnt = 0
        for i in range(n+1):
            for j in range(n+1):
                k = n - i - j
                if k >= 0:
                    self.terms.append([k, j, i])
                    cnt += 1
                    self.w.append(0)


if __name__ == '__main__':
    lam=0.001
    x=[]
    y=[]
    while lam<=0.01:
        gd=L1PolyGradientDescent(6)
        gd.la=lam
        gd.poly_features()
        print(gd.la,len(gd.w),len(gd.terms))
        x.append(gd.la)
        gd.trainModel()
        Y_test=list(gd.Y_test)
        Y_pred=gd.getPredictedValues()
        print("Parameters found by Gradient Descent are: \n", gd.w)
        print("\nRMSE Error: ", RMSE().rmse(Y_pred, Y_test))
        y.append(RMSE().R2_SCORE().rmse(Y_pred, Y_test))
        lam+=0.001
    plt.plot(x,y)
    plt.xlabel("Lambda")
    plt.ylabel("RMSE")
    plt.title("L1 Linear Gradient Descent")
    plt.show()
    #print("R-square Score: ", r2_score(Y_pred, Y_test))
