# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 17:40:24 2019

@author: Sheth_Smit
"""

class RMSE:

    def rmse(self, a, b):
        sum_of_squares = 0

        for (h, y) in zip(a, b):
            sum_of_squares += (h-y)**2

        rmse = (sum_of_squares / (len(a)))**0.5
        return rmse

class R2_SCORE:
    # R^2 Score Calculation
    # Determines how close the data is to the fitted regression model.
    #Implementing r2 = (nE(xy)-(E(x)*E(y)))/((nE(x**2)-E(x)**2)*(nE(y**2)-E(y)**2))
    def r2_score(self,X,Y):

        val = 0
        for i in range(len(X)):
            val+=X[i]*Y[i]
        val*= len(X)
        s1=0
        s2=0
        for i in range(len(X)):
            s1+=X[i]
            s2+=Y[i]
        denm1 = 0
        denm2 = 0
        for i in range(len(X)):
            denm1+=X[i]**2
            denm2+=Y[i]**2
        denm1*=len(X)
        denm2*=len(Y)
        denm1 -= (s1**2)
        denm2 -= (s2**2)
        val -= (s1*s2)
        ans = val**2
        ans /= denm1
        ans /= denm2
        return ans
    
