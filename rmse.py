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
        