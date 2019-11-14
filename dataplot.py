# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 17:40:24 2019

@author: Sheth_Smit
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


class DataPlot:

    def plotModel(self, c, X_matrix, z_pred, W):

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x_data = X_matrix.transpose()[0]
        y_data = X_matrix.transpose()[1]

        X = np.arange(-0.5, 1.5, 0.05)
        Y = np.arange(-0.5, 1.5, 0.05)
        X, Y = np.meshgrid(X, Y)

        #W = self.findParameters()
        Z = []
        for (x_row, y_row) in zip(X, Y):
            row = []
            for(x_cord, y_cord) in zip(x_row, y_row):
                row.append(W[0] + W[1]*x_cord + W[2] * y_cord)
            Z.append(row)

        Z = np.array(Z)

        ax.scatter(x_data, y_data, z_pred, c=c, marker='o')
        ax.plot_surface(X, Y, Z, color='red')

        ax.set_xlabel('X-Axis')
        ax.set_ylabel('Y-Axis')
        ax.set_zlabel('Z-Axis')
        plt.show()
