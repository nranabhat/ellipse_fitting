# *** this file contains a getter function to get the X and y data from multiple CSV files.
# *** The CSV file contains 500 ellipses made out of 30 points each. 
# *** also gets testing data.
import csv
import os
import numpy as np

def load_data(filePath, shape):
    data = []
    with open(filePath) as f:
        reader = csv.reader(f)
        for row in reader:
            data = np.append(data, row)
    # loading ABCDEF labels...
    if len(shape) == 2 and shape[1] == 6:
        data = np.reshape(data, shape, order = 'F')
    # loading Phi_d targets 
    elif len(shape) ==1:
        data = np.reshape(data,shape)
    # loading target CONTRASTS
    elif len(shape) == 2 and shape[1] == 2:
        data = np.reshape(data, shape) # debug this to see if it's correct! 
    # loading X or Y COORDINATES
    else:  
        data = np.reshape(data, shape)
    return data

class loadCSVdata:
    def __init__(self, NUM_TRAINING_ELLIPSES, MAX_SHOTS):

        self._NUM_TRAINING_ELLIPSES = NUM_TRAINING_ELLIPSES
        self._numPoints = MAX_SHOTS
        numTestingEllipses = 100
        numParameters = 6
        numContrasts = 2
        self._shape_training_data = [MAX_SHOTS,NUM_TRAINING_ELLIPSES]
        self._shape_testing_data = [MAX_SHOTS,numTestingEllipses]
        self._shape_training_labels = [NUM_TRAINING_ELLIPSES,numParameters]
        self._shape_testing_labels = [numTestingEllipses,numParameters]
        self._shape_training_phi_d = [NUM_TRAINING_ELLIPSES]
        self._shape_testing_phi_d = [numTestingEllipses]
        self._shape_training_contrast = [NUM_TRAINING_ELLIPSES, numContrasts]
        self._shape_testing_contrast = [numTestingEllipses, numContrasts]

        #self._datasets_path = r"C:\Users\Nicor\OneDrive\Documents\KolkowitzLab\ellipse_fitting_git_tracking\Datasets\Variable contrast"
        self._datasets_path = r"D:\Nico Ranabhat\Ellipse Fitting\ellipse_fitting\Datasets\Variable contrast"
        self._training_set_path = os.path.join(self._datasets_path, "Training Set ("+str(NUM_TRAINING_ELLIPSES)+" ellipses)")
        self._testing_set_path = os.path.join(self._datasets_path, "Testing Set ("+str(numTestingEllipses)+" ellipses)")

    def get_train_labels(self):
            filepath = os.path.join(self._training_set_path, 'trainingL.csv')
            return load_data(filepath, self._shape_training_labels)

    def get_train_data(self):
        
        def get_train_contrasts():
            filepath = os.path.join(self._training_set_path, 'trainingContrasts.csv')
            return load_data(filepath, self._shape_training_contrast)
        def get_train_phi_d():
            filepath = os.path.join(self._training_set_path, 'trainingPhi_d.csv')
            return load_data(filepath, self._shape_training_phi_d)
        def get_x_train_coords():
            filepath = os.path.join(self._training_set_path, 'trainingX.csv')
            return load_data(filepath, self._shape_training_data)
        def get_y_train_coords():
            filepath = os.path.join(self._training_set_path, 'trainingY.csv')
            return load_data(filepath, self._shape_training_data)

        X = get_x_train_coords().T # X [500,30]
        X = np.append(X, get_y_train_coords().T, axis = 1)
        phi = get_train_phi_d()
        y = np.array([phi, get_train_contrasts()[:,0], get_train_contrasts()[:,1]])
        y_T = y.transpose()
        return X,y_T

    
    def get_test_labels(self):
        filepath = os.path.join(self._testing_set_path, 'testingL.csv')
        return load_data(filepath, self._shape_testing_labels)

    def get_test_data(self):
        def get_test_contrasts():
            filepath = os.path.join(self._testing_set_path, 'testingContrasts.csv')
            return load_data(filepath, self._shape_testing_contrast)

        def get_test_phi_d():
            filepath = os.path.join(self._testing_set_path, 'testingPhi_d.csv')
            return load_data(filepath, self._shape_testing_phi_d)

        def get_x_test_coords():
            filepath = os.path.join(self._testing_set_path, 'testingX.csv')
            return load_data(filepath, self._shape_testing_data)

        def get_y_test_coords():
            filepath = os.path.join(self._testing_set_path, 'testingY.csv')
            return load_data(filepath, self._shape_testing_data)

        X = get_x_test_coords().T # X [100,30]
        X = np.append(X, get_y_test_coords().T, axis = 1)
        phi = get_test_phi_d()
        y = np.array([phi, get_test_contrasts()[:,0], get_test_contrasts()[:,1]])
        y_T = y.transpose()
        return X,y_T