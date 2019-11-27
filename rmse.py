# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 17:40:24 2019

@author: Sheth_Smit
"""
import numpy as np
class RMSE:
    '''
    Calculate RMSE of the given two lists i.e predicted and original
    n = length of dataset
    RMSE = sqrt(1/n * Σ[(Y_pred_i - Y_actual_i)^2] )
    '''
    def rmse(self, a, b):
        sum_of_squares = 0

        for (h, y) in zip(a, b):
            sum_of_squares += (h-y)**2

        rmse = (sum_of_squares / (len(a)))**0.5
        return rmse

class R2_SCORE:
    '''
    R^2 Score Calculation
    Determines how close the data is to the fitted regression model.
    Implementing r2 = (TSS - RSS) / (TSS)
    TSS = Σ(y_test - mean_y)^2
    RSS = Σ(y_test - y_pred)^2
    '''
    def r2_score(self,X,Y):

        TSS = 0
        RSS = 0
        mean = np.mean(X)
        
        for i in range(len(Y)):
            TSS += (X[i] - mean) ** 2
        
        for i in range(len(Y)):
            RSS += (X[i] - Y[i]) ** 2
            
        R2 = (TSS - RSS) / (TSS)
        
        return R2
