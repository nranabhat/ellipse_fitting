""" 
Creates a dataset of ellipses and saves the dataset to csv files in a given path.
The datasets consist of the following csv files:

    Contrasts.csv:          contrasts for atom population 1 (x-axis) and atom population 2 (y-axis)
    L.csv:                  a1...a6 conic parameters 
    Phi_d.csv:              phi values 
    X.csv:                  x-coordinates 
    Y.csb:                  y-coordinates

Specify the number of ellipses you want to create with the variable 'numEllipses'
Specify if this data is training or testing data using the first two variables in the sript.
Custimize the range of phi, the path location, and if you want randomly chosen contrast or constant contrast

@author: nranabhat
 """

import math
import csv
import os
from math import cos, sin, acos
import string
import numpy as np
import random as random
from matplotlib import pyplot as plt

CREATING_TRAINING_DATA = True
CREATING_TESTING_DATA = False

NUMBER_ATOMS = 1000
numEllipses = 500 # number of ellipses 
MAX_SHOTS = 500
MIN_SHOTS = 5

# determines file path ~
FULL_PHI_RANGE = True
LAB_COMP = False
VARIABLE_CONTRAST = False # if True, contrast will be differnt for each ellipse. If False, contrast = 0.65

# create folder for new data. Folder name indicates:
# -  variable                    vs. constant contrast
# -  full 0 to pi/2 range of phi vs. small [0,0.15] & [0.98-0.15, 0.15]
# -  lab compuer path            vs. Nico's local machine path
if VARIABLE_CONTRAST: var_cons = "Variable"
else: var_cons = 'Constant'
if FULL_PHI_RANGE: all_phi = " all phi"
else: all_phi = " small phi interval"
if LAB_COMP:
    datasets_path = r"D:\Nico Ranabhat\Ellipse Fitting\ellipse_fitting\Datasets"
else:
    datasets_path = r"C:\Users\Nicor\OneDrive\Documents\KolkowitzLab\ellipse_fitting_git_tracking\Datasets"
if not os.path.isdir(datasets_path): os.mkdir(datasets_path)
DATASET_FOLDER = os.path.join(datasets_path, var_cons+" contrast"+all_phi)
if not os.path.isdir(DATASET_FOLDER): os.mkdir(DATASET_FOLDER)

if CREATING_TESTING_DATA:
    dataset_path = os.path.join(DATASET_FOLDER, "Testing Set ("+str(numEllipses)+" ellipses)")
elif CREATING_TRAINING_DATA:
    dataset_path = os.path.join(DATASET_FOLDER, "Training Set ("+str(numEllipses)+" ellipses)")
if not os.path.isdir(dataset_path): os.mkdir(dataset_path)

# Creating (x,y) from Estey approach:
    # x = c_x cos(phi_c + phi_d) + b_x
    # y = c_y cos(phi_c - phi_d) + b_y
    # Note: here Estey has c_x, c_y as Ax, Ay but I think this is the same thing

numPoints = np.zeros(numEllipses) # points on each ellipse plot - number of shots measuring excitation fraction
for k in range(numEllipses):
    numPoints[k] = int(np.random.randint(MIN_SHOTS, MAX_SHOTS+1)) # number of shots is picked uniformly from [5,500]

X = np.zeros((MAX_SHOTS, numEllipses)) # x-coordinates 
Y = np.zeros((MAX_SHOTS, numEllipses)) # y-coordinates 
Phi_c = np.zeros((MAX_SHOTS, numEllipses))
Phi_d = np.zeros((numEllipses, 1)) # each ellipse has a phi_d
labels = np.zeros((numEllipses, 6)) # 6 parameters for each ellipse
Contrasts = np.zeros((numEllipses, 2)) # each ellipse has 2 contrast values: c_x and c_y 


# center at (0.5, 0.5)
b_x = 1/2 
b_y = 1/2

# make numEllipses (100) plots with different phi_d:
# Define the intervals for FULL_PHI_RANGE = False.  They should be disjoint.
intervals=[[0, 0.15], [math.pi/2-0.15, math.pi/2]]  

for j in range(numEllipses):

    # --- Set Contrast --- # 
    if VARIABLE_CONTRAST:
        contrast = random.uniform(0.1, 0.98)
    else: contrast = 0.65
    c_x = contrast/2
    c_y = contrast/2
    Contrasts[j,0] = c_x
    Contrasts[j,1] = c_y

    # ---- create phi_d value -----# 
    # 0 < phi_d < pi/2 (gets full range of cos when phi_c = pi/2)
    # Choose one number uniformly inside the set
    if not FULL_PHI_RANGE:
        phi_d = random.uniform(*random.choices(intervals,
            weights=[r[1]-r[0] for r in intervals])[0])
    else:
        phi_d = random.uniform(0, math.pi/2)    # lower-case phi_d = numerical values of angle

    Phi_d[j,0] = phi_d                      # upper-case Phi_d = array holding all the phi_d angles
    for i in range(int(numPoints[j])):
        phi_c = random.uniform(0, 2*math.pi)
        Phi_c[i,j] = phi_c
        # x coordinate
        x_i = c_x * cos(phi_c + phi_d) + b_x 
        # adding QPN:
        x_i = np.random.binomial(NUMBER_ATOMS, x_i)/NUMBER_ATOMS
        X[i,j] = x_i
        # y coordinate
        y_i = c_y * cos(phi_c - phi_d) + b_y
        # adding QPN: 
        y_i = np.random.binomial(NUMBER_ATOMS, y_i)/NUMBER_ATOMS
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

# now we have X[#,100] and Y[#,100] data, labels[100,6], Phi_c[#,100] Phi_d[1,100], Contrasts[100,2]

# writing X data to csv file: 
if CREATING_TESTING_DATA:
    Phi_d_path = os.path.join(dataset_path, "testingX.csv")
elif CREATING_TRAINING_DATA:
    Phi_d_path = os.path.join(dataset_path, "trainingX.csv")
with  open(Phi_d_path, "w+", newline="") as f:
    writer = csv.writer(f)
    #now write the data in X to the csv file
    for j in range(MAX_SHOTS):
        for k in range(numEllipses):
            #create row to add
            rowj = X[j,:]
        writer.writerow(rowj)
f.close() # close the file

# writing Y data to csv file:
if CREATING_TESTING_DATA:
    Phi_d_path = os.path.join(dataset_path, "testingY.csv")
elif CREATING_TRAINING_DATA:
    Phi_d_path = os.path.join(dataset_path, "trainingY.csv")
with open(Phi_d_path, 'w+', newline='') as f:
    writer = csv.writer(f)
    #now write the data in Y to the csv file
    for j in range(MAX_SHOTS):
        for k in range(numEllipses):
            #create row to add
            rowj = Y[j,:]
        writer.writerow(rowj)
f.close() # close the file

# writing labels to csv file: 
if CREATING_TESTING_DATA:
    Phi_d_path = os.path.join(dataset_path, "testingL.csv")
elif CREATING_TRAINING_DATA:
    Phi_d_path = os.path.join(dataset_path, "trainingL.csv")
labels = labels.T #turns shape to (6,100)
with open(Phi_d_path, 'w+', newline='') as f:
    writer = csv.writer(f)
    #now write the data in labels to the csv file
    for j in range(6): #6 conic parameters for each ellipse 
        for k in range(numEllipses):
            #create row to add
            rowj = labels[j,:]
        writer.writerow(rowj)
f.close() # close the files

# at this point, 3 csv files have been created holding the data in X, Y, and labels 
# make files containing phi_d and phi_c just in case: 

# writing Phi_d to csv file:
if CREATING_TESTING_DATA:
    Phi_d_path = os.path.join(dataset_path, "testingPhi_d.csv")
elif CREATING_TRAINING_DATA:
    Phi_d_path = os.path.join(dataset_path, "trainingPhi_d.csv")
with open(Phi_d_path, 'w+', newline='') as f:
    writer = csv.writer(f)
    #now write the data in Phi_d to the csv file
    for m in range(numEllipses):
        writer.writerow(Phi_d[m])
f.close() # close the files


# writing Contrasts to csv file: 
if CREATING_TESTING_DATA:
    Contrasts_path = os.path.join(dataset_path, "testingContrasts.csv")
elif CREATING_TRAINING_DATA:
    Contrasts_path = os.path.join(dataset_path, "trainingContrasts.csv")
with open(Contrasts_path, 'w+', newline='') as f:
    writer = csv.writer(f)
    #now write the data in Contrasts to the csv file
    for m in range(numEllipses):
        writer.writerow(Contrasts[m])
f.close