# Test Single model

import csv
import math
import os
import numpy as np
import torch
from torch import device, nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import wandb
import loadCSVdata_phi_d
from plot_nine_phi_d import plot_nine
from matplotlib.patches import Ellipse
from ellipse import LsqEllipse
from Sweep_phi_d import build_dataset, build_network

NUM_TRAINING_ELLIPSES = 100
NUM_POINTS = 30

def NN_output(num_training_ellipses):

    api = wandb.Api()
    artifact = api.artifact('nicoranabhat/ellipse_fitting/best-run4-phase-kpsfqfpb-500000-trainingEllipses.pt:v20')
    artifact_dir = artifact.download()

    #now need to get config
    config = artifact.metadata

    testloader = build_dataset(int(config['batch_size']), int(num_training_ellipses), train=False)
    # previous network build: 
    network = build_network(int(config['second_layer_size']), clamp_output=True)
    # new network built:
    weights_path = os.path.join(artifact_dir, 'weights_tensor.pt')
    network.load_state_dict(torch.load(weights_path)['model_state_dict'])
    # don't need to load scheduler and optimizer because we only run through one ~non-training~ epoch for validation

    # test once manually: 
    #for data in enumerate(testloader, 0): # should just be one big batch of all the data (for testing)

    # Get and prepare inputs
    inputs, targets = testloader.dataset[:]
    inputs, targets = inputs.float(), targets.float()
    #inputs, targets = inputs.to(device), targets.to(device)
    targets = targets.reshape((targets.shape[0], 1))
    
    # Perform forward pass w/o training gradients 
    with torch.no_grad():
        outputs = network(inputs)

    LS_output = get_LS_output(inputs)

    return inputs, outputs, LS_output, targets
    

def get_LS_output(inputs):
    inputs = inputs.detach().numpy()

    def make_test_ellipse(X,i):
        """Generate Elliptical

        Returns
        -------
        data:  list:list:float
            list of two lists containing the x and y data of the ellipse.
            of the form [[x1, x2, ..., xi],[y1, y2, ..., yi]]
        """
        x_y_arrays = np.split(X, 2, axis=1)
        ellipse_x = x_y_arrays[0]
        ellipse_y = x_y_arrays[1]
        x_coords = np.asarray(ellipse_x[i,:], dtype=float)
        y_coords = np.asarray(ellipse_y[i,:], dtype=float)
        return [x_coords, y_coords]
    
    Phi_LS = np.empty(shape=NUM_TRAINING_ELLIPSES)
    for i in range(NUM_TRAINING_ELLIPSES):
        X1, X2 = make_test_ellipse(inputs,i)
        X_single_ellipse = np.array(list(zip(X1, X2)))
        fitter = LsqEllipse()
        reg = fitter.fit(X_single_ellipse)
        center, width, height, phi = reg.as_parameters()
        LS_output = fitter.coefficients

        # finding target (a1-a6):
        loader = loadCSVdata_phi_d.loadCSVdata(NUM_TRAINING_ELLIPSES, NUM_POINTS)
        X_target, y_target = loader.get_test_data()

        # finding LS (a1-a6) loss:
        a = LS_output[0];       A = float(y_target[i,0])
        b = LS_output[1];       B = float(y_target[i,1])
        c = LS_output[2];       C = float(y_target[i,2])
        # d = LS_output[3];       D = float(targets[i,3])
        # e = LS_output[4];       E = float(targets[i,4])
        # f = LS_output[5];       F = float(targets[i,5])

        acos_arg_targets = -B/(2*math.sqrt(A*C))
        acos_arg_model = -b/(2*math.sqrt(a*c))
        if np.sign(acos_arg_targets) != np.sign(acos_arg_model):
            acos_arg_model = -acos_arg_model
        #target_phi_d = 0.5*acos(acos_arg_targets)
        model_phi_d = 0.5*math.acos(acos_arg_model)
        #Phi_targets[i] = target_phi_d
        Phi_LS[i] = model_phi_d

    return Phi_LS


inputs, NN_output, LS_output, targets = NN_output(NUM_TRAINING_ELLIPSES)

NN_output = NN_output.detach().numpy()
LS_output = np.reshape(LS_output, [100,1])

Phi_d_path = r'C:\Users\Nicor\OneDrive\Documents\KolkowitzLab\ellipse_fitting_git_tracking\Learners\Learner 2 phase loss\Phi_d values (Matt)'
Phinn_d_path = os.path.join(Phi_d_path, "phi_NN.csv")
with open(Phinn_d_path, 'w+', newline='') as f:
    writer = csv.writer(f)
    #now write the data in Phi_d to the csv file
    for m in range(100):
        writer.writerow(NN_output[m])
f.close() # close the files

Phils_d_path = os.path.join(Phi_d_path, "phi_LS.csv")
with open(Phils_d_path, 'w+', newline='') as f:
    writer = csv.writer(f)
    #now write the data in Phi_d to the csv file
    for m in range(100):
        writer.writerow(LS_output[m])
f.close() # close the files


# writing to csv: 
print('hi')