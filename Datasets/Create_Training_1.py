import math
import csv
from math import cos, sin, acos
import os
import string
import numpy as np
import random as random
from matplotlib import pyplot as plt

# Creating (x,y) from Estey approach:
    # x = c_x cos(phi_c + phi_d) + b_x
    # y = c_y cos(phi_c - phi_d) + b_y
    # Note: here Estey has c_x, c_y as Ax, Ay but I think this is the same thing

numPoints = 30 # points on each ellipse plot - number of shots measuring excitation fraction
numEllipses = 100000 # number of ellipses 
X = np.empty((numPoints, numEllipses)) # x-coordinates 
Y = np.empty((numPoints, numEllipses)) # y-coordinates 
Phi_c = np.empty((numPoints, numEllipses))
Phi_d = np.empty(numEllipses) # each ellipse has a phi_d
labels = np.empty((numEllipses, 6)) # 6 parameters for each ellipse

# --- Set constants --- # 
CONTRAST = 0.65
c_x = CONTRAST/2
c_y = CONTRAST/2
# center at (0.5, 0.5)
b_x = 1/2 
b_y = 1/2

# make int(numEllipses) plots with different phi_d:
for j in range(numEllipses):
    # 0 < phi_d < pi/2 (gets full range of cos when phi_c = pi/2)
    phi_d = random.uniform(0, math.pi/2)
    Phi_d[j] = phi_d
    for i in range(numPoints):
        phi_c = random.uniform(0, 2*math.pi)
        Phi_c[i,j] = phi_c
        x_i = c_x * cos(phi_c + phi_d) + b_x
        X[i,j] = x_i
        y_i = c_y * cos(phi_c - phi_d) + b_y
        Y[i,j] = y_i
    # create known_labels using Estey equations 
    A = 1 / ((c_x)**2)
    B = -(2*cos(2*phi_d)) / (c_x*c_y)
    C = 1/(c_y)**2
    D = (2*b_y*cos(2*phi_d)/(c_x*c_y)) - (2*b_x)/(c_x)**2
    E = (2*b_x*cos(2*phi_d)/(c_x*c_y)) - (2*b_y)/(c_y)**2
    F = ((b_x)**2/((c_x)**2) + (b_y)**2/((c_y)**2) - 
    (2*b_x*b_y*cos(2*phi_d))/(c_x*c_y) - 
    4*(cos(phi_d))**2*(sin(phi_d))**2)

    assert B**2 - 4*A*C < 0

    coefficients = np.array([A,B,C,D,E,F])
    labels[j] = coefficients

# now we have X[30,500] and Y[30,500] data, labels[500,6], Phi_c[30,500] Phi_d[500,]
datasets_path = r"C:\Users\Nicor\OneDrive\Documents\KolkowitzLab\Ellipse fitting\Datasets\Updated Contrast Datasets"
training_set_path = os.path.join(datasets_path, "Training Set ("+str(numEllipses)+" ellipses)")
if not os.path.isdir(training_set_path): os.mkdir(training_set_path)

# writing X data to csv file: 
pathname = os.path.join(training_set_path, "training1X.csv")
with open(pathname, 'w', newline='') as f:
    writer = csv.writer(f)
    #now write the data in X to the csv file
    for j in range(numPoints):
        for k in range(numEllipses):
            #create row to add
            rowj = X[j,:]
        writer.writerow(rowj)
f.close() # close the file

# writing Y data to csv file: 
pathname = os.path.join(training_set_path, "training1Y.csv")
with open(pathname, 'w', newline='') as f:
    writer = csv.writer(f)
    #now write the data in Y to the csv file
    for j in range(numPoints):
        for k in range(numEllipses):
            #create row to add
            rowj = Y[j,:]
        writer.writerow(rowj)
f.close() # close the file

# writing labels to csv file: 
labels = labels.T #turns shape to (6,500)
pathname = os.path.join(training_set_path, "training1Labels.csv")
with open(pathname, 'w', newline='') as f:
    writer = csv.writer(f)
    #now write the data in labels to the csv file
    for j in range(6): #6 conic parameters for each ellipse 
        for k in range(numEllipses):
            #create row to add
            rowj = labels[j,:]
        writer.writerow(rowj)
f.close() # close the files

# writing Phi_d to csv file:
pathname = os.path.join(training_set_path, "training1Phi_d.csv")
with open(pathname, 'w', newline='') as f:
    writer = csv.writer(f)
    #now write the data in Phi_d to the csv file
    writer.writerow(Phi_d)
f.close() # close the files

# at this point, 4 csv files have been created holding the data in X, Y, labels, and phi_d values
print('\nTraining set with '+str(numEllipses)+' ellipses with '+str(numPoints)+' points each has been saved to ')
print(str(training_set_path))