""" Script to load a saved model from wandb and plot jackknife analysis 
    Created October 26, 2022
    @author: nranabhat 
"""

import math
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, numel
import math
from math import cos, sin
import torch
from PIL import Image
import wandb
import os
import random
import numpy as np
import loadCSVdata_var_input
from ellipse import LsqEllipse
from Sweep_var_input import build_network, get_LS_test_loss

### Wandb Artifact to load
# - constant contrasts cx=cy=0.65/2
# - input is 1000 neurons, output is phase estimate
RUN_ID = '8trhbjq2'
VERSION_NUM = 'latest'
NUM_TRAINING_ELLIPSES = '500000'
NAME_OF_ARTIFACT_TO_USE = 'nicoranabhat/ellipse_fitting/best-run8-phase-'+RUN_ID+'-'+NUM_TRAINING_ELLIPSES+'-trainingEllipses.pt:'+str(VERSION_NUM)

NUM_NEW_EPOCHS = 1
MAX_SHOTS = 500

### Login to wandb and defining helper functions 
wandb.login()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(torch.utils.data.Dataset):
  def __init__(self, X, y, scale_data=False):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      # Apply scaling if necessary
      if scale_data:
          X = StandardScaler().fit_transform(X)
      self.X = torch.from_numpy(X)
      self.y = torch.from_numpy(y)

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]
    
def create_dataset():
    # make X [500, 495] with X[:,0] being the ellipse with 5 points, X[:,2] 6 points, ... X[:,495] 500 points
    NUMBER_ATOMS = 1000
    numEllipses = 495 # number of ellipses (5,6,...,500 points)
    MAX_SHOTS = 500
    MIN_SHOTS = 5
    numPoints = MAX_SHOTS

    X = np.zeros((numEllipses, numPoints)) # x-coordinates [495,500]
    Y = np.zeros((numEllipses, numPoints)) # y-coordinates 
    labels = np.zeros((1, 6)) # same 6 parameters for each ellipse

    # --- Set constants --- # 
    CONTRAST = 0.65
    c_x = CONTRAST/2
    c_y = CONTRAST/2
    # center at (0.5, 0.5)
    b_x = 1/2 
    b_y = 1/2

    # make an ellipse with a phi_d value:
    # Define the intervals.  They should be disjoint.
    intervals=[[0, 0.15], [math.pi/2-0.15, math.pi/2]]

    # 0 < phi_d < pi/2 (gets full range of cos when phi_c = pi/2)
    # Choose one number uniformly inside the set
    phi_d = random.uniform(*random.choices(intervals,
        weights=[r[1]-r[0] for r in intervals])[0])
    Phi_d = np.empty(495)

    # create each ellipse
    for i in range(int(numEllipses)): #495
        # add phi_d value to Phi_d
        Phi_d[i] = phi_d
        # for the first ellipse, add 5 non-zero points, then 6, and so on
        for j in range(i+5): #j is from 0 to 495
            phi_c = random.uniform(0, 2*math.pi)
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
        # zeropad after adding non-zero points
        # for h in range(499-j):
        #     X[i,h+j+1] = 0
        #     Y[i,h+j+1] = 0

    X = np.append(X, Y, axis = 1)

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

    return X, Phi_d, coefficients

def build_dataset(X, phi):
    dataset = Dataset(X, phi)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=X.shape[0], shuffle=False, num_workers=1)
    return testloader

def get_test_output(network, X, phi):

    testloader = build_dataset(X, phi)

    # test once manually: 
    loss_function = nn.MSELoss()
    loss = np.zeros(495)
    for i, data in enumerate(testloader, 0):

        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.reshape((targets.shape[0], 1))
        
        # Perform forward pass w/o training gradients 
        with torch.no_grad():
            outputs = network(inputs)
        
    # Compute loss ARRAY
    loss = np.zeros(495)
    for k in range(targets.shape[0]):
        rmse_loss = torch.sqrt(loss_function(outputs[k,0], targets[k,0]))
        loss[k] = rmse_loss
        
    return inputs, outputs, targets, loss

def plot_errors(targets, outputs, Phi_LS):
    # plotting errors from the NN and the LS algorithm: 

    # convert input coords to arrays
    if (type(targets) == torch.Tensor):
        targets = targets.detach().numpy()
    if (type(outputs) == torch.Tensor):
        outputs = outputs.detach().numpy()
    if (type(Phi_LS) == torch.Tensor):
        Phi_LS = Phi_LS.detach().numpy()

    true_phis = targets[:,0]
    phi_nn = outputs[:,0]
    phi_ls = np.reshape(Phi_LS, 100,)

    nn_errs = phi_nn-true_phis
    ls_errs = phi_ls-true_phis

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(true_phis, ls_errs, s=10, marker="s", label='LS')
    ax1.scatter(true_phis, nn_errs, s=10, marker="o", label='NN')
    ax1.set_xlabel('True Phase')
    ax1.set_ylabel('Error')
    plt.legend(loc='upper center')
    img = fig2img(fig)

    #plt.show()
    return img

def get_LS_test_loss(inputs, targets, coefficients):
    inputs = inputs.detach().numpy()
    targets = targets.detach().numpy()

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
        for i in range(MAX_SHOTS):
            if x_coords[i] == 0: # contrast will never be exactly 1, so points will never be exactly on [0,0].
                end_index = i
                break
        x_coords = np.delete(x_coords, slice(end_index,MAX_SHOTS))
        y_coords = np.delete(y_coords, slice(end_index,MAX_SHOTS))
        return [x_coords, y_coords]
    
    Phi_LS = np.empty(shape=len(targets))
    for i in range(len(targets)):
        X1, X2 = make_test_ellipse(inputs,i)
        X_single_ellipse = np.array(list(zip(X1, X2)))
        fitter = LsqEllipse()
        reg = fitter.fit(X_single_ellipse)
        center, width, height, phi = reg.as_parameters()
        LS_output = fitter.coefficients

        # finding target (a1-a6):
        #loader = loadCSVdata_var_input.loadCSVdata(NUM_TRAINING_ELLIPSES, MAX_SHOTS)
        #X_target, y_target = loader.get_test_data()

        # finding LS (a1-a6) loss:
        a = LS_output[0];       A = float(coefficients[0])
        b = LS_output[1];       B = float(coefficients[1])
        c = LS_output[2];       C = float(coefficients[2])
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

    Phi_LS = np.reshape(Phi_LS, (Phi_LS.shape[0], 1))
    LS_rmse_loss = np.zeros(495)
    for k in range(Phi_LS.shape[0]):
        LS_test_loss = np.sqrt(np.linalg.norm(targets[k]-Phi_LS[k])**2)
        LS_rmse_loss[k] = LS_test_loss
    return LS_rmse_loss, Phi_LS

def plot_jackknife(rmse, ls_loss, phi):

    x_values = np.arange(5,500)
    y_values_nn = rmse
    y_values_la = ls_loss

    phi = phi[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x_values, y_values_nn,  s=10, facecolors='none', edgecolors='b', label='NN')
    ax1.scatter(x_values, ls_loss,  s=10, facecolors='none', edgecolors='r', label='LS')
    ax1.set_xlabel('Number of Points')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Error vs. number of points on ellipse (phi= '+str(phi)[0:5]+')')
    plt.legend(loc='upper center')

    return fig

### Load the artifact, run on a dataset, plot analysis
if __name__ == '__main__': 
    with wandb.init(project='ellipse_fitting') as run:
  
        print("Loading artifact at: "+NAME_OF_ARTIFACT_TO_USE)
        artifact = run.use_artifact(NAME_OF_ARTIFACT_TO_USE, type='model')
        artifact_dir = artifact.download()
        state_dicts_path = os.path.join(artifact_dir, 'weights_tensor.pt')
        config = artifact.metadata

        X, phi, coefficients = create_dataset()
        network = build_network(int(config['second_layer_size']),clamp_output=True)
        network.load_state_dict(torch.load(state_dicts_path)['model_state_dict'])

        inputs, nn_outputs, targets, rmse = get_test_output(network=network, X=X, phi=phi)
        ls_loss, ls_outputs = get_LS_test_loss(inputs, targets, coefficients)

        #img = plot_errors(targets, nn_outputs, ls_outputs)
        fig = plot_jackknife(rmse, ls_loss, phi)
        fig.show()
