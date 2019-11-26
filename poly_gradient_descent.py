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
from sklearn.model_selection import train_test_split


class PolyGradientDescent:

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
        '''
        Initialising initial value of coefficients of Linear Regression Model
        Formula of Polynomial Regression Model
        '''
        self.w=[]
        self.terms=[]
        '''
        Saving the degree of Regression Model
        '''
        self.degree=degree
        '''
        Saved values of learning rate of Regression Model selected
        by trial and error method and checking via plotting
        alpha vs RMSE graph for every polynomial degree Model until degree 6
        '''
        self.alph=[0,0.00000425,0.000003,0.000003,0.000003,0.000002,0.000002]
        '''
        Initialising learning rate alpha according to degree of Polynomial Regression Model
        '''
        self.alpha=self.alph[self.degree]


    def sumOfError(self):
        '''
        S.S.E = (1/2) Σ[(Y_pred_i - Y_actual_i)^2]

        Also returns the value Calculated from the differential equation of Sum of Squared Error.
        𝛛(SSE)/𝛛(w_i) = Σ [(Y_pred_i - Y_actual_i) * X0_i^(term_value_0) * X1_i^(term_value_1)]

        w_i = w_i - ɑ*(𝛛(SSE)/𝛛(w_i))
        '''
        retw=[]
        for i in range(len(self.w)):
            retw.append(0)

        for i in range(len(self.X0_train)):
            ss=-self.Y_train[i]
            j=0
            for k in self.terms:
               ss+=self.w[j]*((self.X0_train[i] ** k[1])*(self.X1_train[i] ** k[2]))
               j+=1
            for h in range(len(retw)):
                k=self.terms[h]
                retw[h]+=ss*(pow(1,k[0])*pow(self.X0_train[i],k[1])*pow(self.X1_train[i],k[2]))

        return retw

    def totalError(self):
        '''
        Calculates Squared Sum of Error for train dataset for
        given values of w0 , w1 and w2.

        S.S.E = (1/2) Σ[(Y_pred_i - Y_actual_i)^2]
        '''
        error=0
        for i in range(len(self.X0_train)):
            ss=-self.Y_train[i]
            j=0
            for k in self.terms:
               ss+=self.w[j]*(pow(1,k[0])*pow(self.X0_train[i],k[1])*pow(self.X1_train[i],k[2]))
               j+=1
            error+=ss**2
        return error/2


    def trainModel(self):
        '''
        Training the Gradient Descent Model for given dataset
        for 1000 epoch and stop training the model.
        '''
        for j in range(50):
            retw=self.sumOfError()
            for i in range(len(self.w)):
                self.w[i]=self.w[i]-(self.alpha*retw[i])
            print(self.alpha,j,self.totalError())


    def getPredictedValues(self):
        '''
        Function that returns predicted values of target
        variable for all given test data points
        '''
        Y_pred=[]
        for i in range(len(self.X0_test)):
            ans=0
            for j in range(len(self.w)):
                k=self.terms[j]
                ans+=(self.w[j])*(pow(1,k[0])*pow(self.X0_test[i],k[1])*pow(self.X1_test[i],k[2]))
            Y_pred.append(ans)
        return Y_pred


    def poly_features(self):
        '''
        Calculate polynomial features and terms for polynomial regression model according
        to given degree if equation.

        Storing all terms in terms list and Initialising coefficients to 0

        Terms are tuples of form (i,j,k) where it is pow(1,i) , pow(X0,j) and pow(X1,k)
        '''
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
    gd=PolyGradientDescent(6)
    gd.poly_features()
    gd.trainModel()
    print()
    Y_test=list(gd.Y_test)
    Y_pred=gd.getPredictedValues()

    """Printing the co-efficients , RMSE and R-square score of the model"""

    print("Parameters found by Gradient Descent are: \n", gd.w)
    print("\nRMSE Error: ", RMSE().rmse(Y_pred, Y_test))
    print("R-square Score: ", R2_SCORE().r2_score(Y_pred, Y_test))
