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
from rmse import RMSE
from dataplot import DataPlot
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression 

class PolyGradientDescent:

    def __init__(self,degree):
        self.dataset = pd.read_csv('./data/normalized_dataset.csv')
        X = self.dataset.iloc[:, 2:4].values
        #X = self.dataset.iloc[:, 3].values
        Y = self.dataset.iloc[:, 4].values
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.3, random_state=9)
        
        self.X0_train=list(self.X_train[:,0])
        self.X1_train=list(self.X_train[:,1])
        self.X0_test=list(self.X_test[:,0])
        self.X1_test=list(self.X_test[:,1])
        self.Y_test=list(self.Y_test)
        self.Y_train=list(self.Y_train)
        self.w=[]
        self.terms=[]
        self.degree=degree
        self.alpha=0.000003


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
        
        return retw
    
    def totalError(self):
        error=0
        for i in range(len(self.X0_train)):
            ss=-self.Y_train[i]
            j=0
            for k in self.terms:
               ss+=self.w[j]*(pow(1,k[0])*pow(self.X0_train[i],k[1])*pow(self.X1_train[i],k[2]))
               j+=1
            error+=ss**2
        return error


    def trainModel(self):
        for j in range(0):
            retw=self.sumOfError()
            for i in range(len(self.w)):
                self.w[i]=self.w[i]-(self.alpha*retw[i])
            print("Error = ",self.totalError())
            
            
    def getPredictedValues(self):
        Y_pred=[]
        for i in range(len(self.X0_test)):
            ans=0
            for j in range(len(self.w)):
                k=self.terms[j]
                ans+=(self.w[j])*(pow(1,k[0])*pow(self.X0_test[i],k[1])*pow(self.X1_test[i],k[2]))
            Y_pred.append(ans)
        return Y_pred
    
    
    def poly_features(self,d):
        n=d
        cnt = 0
        for i in range(n+1):
            for j in range(n+1):
                k = n - i - j
                if k >= 0:
                    self.terms.append([i, j, k])
                    cnt += 1
                    self.w.append(0)
        return (self.terms)


def plot(x,y,Title):
    plt.xlabel('alpha')
    plt.ylabel('rmse') 
    plt.title(Title) 
    plt.show() 

def sum_of_error(w,terms,x0,x1,y):
    error=0
    for i in range(len(x0)):
        ans=0
        for j in range(len(w)):
            k=terms[j]
            ans+=w[j]*(pow(1,k[0]))*(pow(x0[i],k[1]))*(pow(x1[i],k[2]))
        ans-=y[i]
        error+=ans**2
    return (error/2)


if __name__ == '__main__':
    q=[0.201,0.1424,0.2295,0.00000001,0.00000001,0]
    gd=PolyGradientDescent(1)
    Y_test=list(gd.Y_test)
    for h in range(1,7):
        poly = PolynomialFeatures(degree = h) 
        X_poly = poly.fit_transform(gd.X_train) 
        poly.fit(X_poly,gd.Y_train) 
        lin2 = LinearRegression() 
        lin2.fit(X_poly, gd.Y_train)
        Y_pred1=lin2.predict(poly.fit_transform(gd.X_test))
        print("Degree : ",h)
        m={}
        feat=poly.get_feature_names()
        #print("Terms : ",feat)
        j=0
        for i in feat:
            m[i]=j
            j+=1
        
        gd.terms=[]
        org=gd.poly_features(h)
        my_terms={}
        for i in org:
            if i[1]==0 and i[2]==0:
                my_terms[m['1']]=[1,0,0]
            else:
                s=''
                f=0
                ff=0
                fff=0
                if i[1]>0 and i[1]<2:
                    s+='x0'
                    ff=1
                    f+=1
                elif i[1]>=2:
                    s+='x0^'+str(i[1])
                    ff=1
                    f+=1
                gg=''
                if i[2]>0 and i[2]<2:
                    gg+='x1'
                    fff=1
                    f+=1
                elif i[2]>=2:
                    gg+='x1^'+str(i[2])
                    fff=1
                    f+=1
                if f==2:
                    s=s+' '+gg
                elif f==1:
                    if ff==1:
                        s=s
                    elif fff==1:
                        s=gg
                my_terms[m[s]]=i
        terms=[]
        for i in range(len(my_terms)):
            terms.append(my_terms[i])
        #print("Features : ",terms)
        w=lin2.coef_
        #print(w)
        w[0]=q[h-1]
        print("Features: ",feat)
        print("Co-efficients: ",w)
        print("\nRMSE Error: ", RMSE().rmse(Y_pred1, Y_test))
        print("R2 error : ",r2_score(Y_test, Y_pred1))
        print("Training Error : ",sum_of_error(w,terms,gd.X0_train,gd.X1_train,gd.Y_train))
        print("Validation Error : ",sum_of_error(w,terms,gd.X0_test,gd.X1_test,gd.Y_test))
        print()
        
    #print("R-square Score: ", r2_score(Y_pred, Y_test))
# -*- coding: utf-8 -*-

