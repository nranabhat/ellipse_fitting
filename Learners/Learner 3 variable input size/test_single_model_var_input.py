""" Script to load a saved model from wandb and plot analysis 
    Created October 13, 2022
    @author: nranabhat 
"""

import math
from matplotlib import pyplot as plt
import torch
from torch import nn
import stat
import time
import torch
from PIL import Image
import wandb
import os
import numpy as np
import loadCSVdata_var_input
from matplotlib.patches import Ellipse
from ellipse import LsqEllipse
from Sweep_var_input import build_dataset, build_network, get_LS_test_loss

### Wandb Artifact to load
# - constant contrasts cx=cy=0.65/2
# - input is 1000 neurons, output is phase estimate
RUN_ID = '8trhbjq2'
VERSION_NUM = 'latest'
NUM_TRAINING_ELLIPSES = '500000'
NAME_OF_ARTIFACT_TO_USE = 'nicoranabhat/ellipse_fitting/best-run8-phase-'+RUN_ID+'-'+NUM_TRAINING_ELLIPSES+'-trainingEllipses.pt:'+str(VERSION_NUM)
#NAME_OF_ARTIFACT_TO_USE = 'nicoranabhat/ellipse_fitting/best-mlp-sweep-phase-'+RUN_ID+'.pt:'+str(VERSION_NUM)
#LOG_NEW_ARTIFACT_TO = f'test-run-phase-'+str(RUN_ID)+'-'+NUM_TRAINING_ELLIPSES+'-trainingEllipses.pt'

#wandbpath = r"C:\Users\Nicor\OneDrive\Documents\KolkowitzLab\ellipse_fitting\Learners\wandb"   
#wandbpath = r"D:\Nico Ranabhat\Ellipse Fitting\ellipse_fitting\Learners\wandb"
#pathname = os.path.join(wandbpath, 'best-'+NUM_TRAINING_ELLIPSES+'-trainingellipses-run-for-sweep-'+RUN_ID)
#MODEL_PATH = os.path.join(pathname, 'weights_tensor.pt')

NUM_NEW_EPOCHS = 1
MAX_SHOTS = 500

### Login to wandb and defining helper functions 
wandb.login()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_test_output(batch_size, network):
    # **first two parameters of build_dataset don't rly matter if train=False.**
    testloader = build_dataset(int(batch_size), int(NUM_TRAINING_ELLIPSES), train=False)

    # test once manually: 
    loss_function = nn.MSELoss()
    for i, data in enumerate(testloader, 0): # should just be one big batch of all the data (for testing)
    
        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.reshape((targets.shape[0], 1))
        
        # Perform forward pass w/o training gradients 
        with torch.no_grad():
            outputs = network(inputs)
        
        # Compute loss
        total_loss = loss_function(outputs[:,0], targets[:,0])
        avg_loss = total_loss

    return inputs, outputs, targets, avg_loss

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

def get_LS_test_loss(inputs, targets):
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
        loader = loadCSVdata_var_input.loadCSVdata(NUM_TRAINING_ELLIPSES, MAX_SHOTS)
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

    Phi_LS = np.reshape(Phi_LS, (Phi_LS.shape[0], 1))
    LS_test_loss = np.linalg.norm(targets-Phi_LS)**2 / len(targets)
    return LS_test_loss, Phi_LS

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

### Load the artifact, run on a dataset, plot analysis
if __name__ == '__main__': 
    with wandb.init(project='ellipse_fitting') as run:
  
        print("Loading artifact at: "+NAME_OF_ARTIFACT_TO_USE)
        artifact = run.use_artifact(NAME_OF_ARTIFACT_TO_USE, type='model')
        artifact_dir = artifact.download()
        state_dicts_path = os.path.join(artifact_dir, 'weights_tensor.pt')
        config = artifact.metadata

        network = build_network(int(config['second_layer_size']),clamp_output=True)
        network.load_state_dict(torch.load(state_dicts_path)['model_state_dict'])

        inputs, nn_outputs, targets, nn_loss = get_test_output(batch_size=100, network=network)
        ls_loss, ls_outputs = get_LS_test_loss(inputs, targets)

        img = plot_errors(targets, nn_outputs, ls_outputs)
        img.show()
