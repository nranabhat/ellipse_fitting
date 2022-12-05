# PLOTTING fit for LAST 9 TESTING SAMPLES

# imports 
from math import acos, cos, sin
import math
import numpy as np
import torch
from torch import nn
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

# imports from Matt's algo 
from numpy.core.shape_base import block
#import utils.tool_belt as tool_belt
#from utils import common
#from utils import kplotlib as kpl
#from utils.kplotlib import KplColors

CENTER = 0.5
b_y = CENTER
b_x = CENTER

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

# in place to see if nn produces outputs in non-physical rangeS
def print_output_range_warning(phi,CLAMP_EPSILON):
    ep = CLAMP_EPSILON
    ep = 0.0001 # differnt from CLAMP_EPSILON because python truncates value of a through e here.
    if (phi<-ep) or (phi>(math.pi/2+ep)):
        print('warning, output phase ('+str(phi)+') does not fit in range ['\
            +str(-ep)+', '+str(math.pi/2+ep)+']')

def plot_nine(PLOT_MLE, input_coords, targets, 
              outputs, Phi_LS,
              test_loss, test_phase_loss, train_loss, LS_test_loss, MLE_test_Loss,
              CLAMP_EPSILON):

    m,n = 3,3 # 3x3 subplot (9 total ellipse fits) 
    figure, axis = plt.subplots(m,n, sharex='all', sharey = 'all')

    # convert input coords to arrays
    if (type(input_coords) == torch.Tensor):
        input_coords = input_coords.detach().numpy()
    if (type(targets) == torch.Tensor):
        targets = targets.detach().numpy()
    if (type(outputs) == torch.Tensor):
        outputs = outputs.detach().numpy()
    if (type(Phi_LS) == torch.Tensor):
        Phi_LS = Phi_LS.detach().numpy()

    # range 3, range 3
    for k in range(m):
        for h in range(n):

            # Contour plot of known and fit
            # min = CENTER + CONTRAST/2
            # max = CENTER - CONTRAST/2
            # x = np.linspace(min, max, 400)
            # y = np.linspace(min, max, 400)
            # x, y = np.meshgrid(x, y)

            phi_d_target = targets[(k+1)*(h+1),0]
            phi_d_output = outputs[(k+1)*(h+1),0]
            phi_d_LS = Phi_LS[(k+1)*(h+1)]
            c_x_target = targets[(k+1)*(h+1),1]
            c_y_target = targets[(k+1)*(h+1),2]
            c_x_output = outputs[(k+1)*(h+1),1]
            c_y_output = outputs[(k+1)*(h+1),2]

            print('\nNeural Net output phase: '+str(phi_d_output))
            print('Least Squares output phase: '+str(phi_d_LS))
            print('Target phase:        '+str(phi_d_target))

            phase_loss_nn = np.linalg.norm(phi_d_target - phi_d_output)**2
            phase_loss_LS = np.linalg.norm(phi_d_target - phi_d_LS)**2
            print('\nPhase loss between nn and truth (one sample): '+str(phase_loss_nn))
            print('Phase loss between LS and truth (one sample): '+str(phase_loss_LS)+'\n')
            print_output_range_warning(phi_d_output, CLAMP_EPSILON)
            
            # Scatter plot of actual noisy ellipse points 
            max_shots = int(input_coords.shape[1] / 2)
            x_points = input_coords[(k+1)*(h+1), 0:max_shots]
            y_points = input_coords[(k+1)*(h+1), max_shots:2*max_shots]
            for i in range(max_shots):
                if x_points[i] == 0: # contrast will never be exactly 1, so points will never be exactly on [0,0].
                    end_index = i
                    break
            x_points = np.delete(x_points, slice(end_index,max_shots))
            y_points = np.delete(y_points, slice(end_index,max_shots))
            data = axis[k,h].scatter(x_points, y_points, s=1, label='data')

            # Scatter plot of known and fits based off of phase and known contrast 
            Phi_c = np.linspace(0, 2*math.pi, 600)
            X_fit = np.empty(600)
            Y_fit = np.empty(600)
            X_known = np.empty(600)
            Y_known = np.empty(600)
            X_ls = np.empty(600)
            Y_ls = np.empty(600)

            for i in range(600): 
                X_fit[i] = c_x_output*cos(Phi_c[i] + phi_d_output) + b_x
                Y_fit[i] = c_y_output*cos(Phi_c[i] - phi_d_output) + b_y
                X_known[i] = c_x_target*cos(Phi_c[i] + phi_d_target) + b_x
                Y_known[i] = c_y_target*cos(Phi_c[i] - phi_d_target) + b_y
                X_ls[i] = c_x_target*cos(Phi_c[i] + phi_d_LS) + b_x
                Y_ls[i] = c_y_target*cos(Phi_c[i] - phi_d_LS) + b_y
            known = axis[k,h].plot(X_known, Y_known, label='known', color='k', markersize=0.1)
            ls = axis[k,h].plot(X_ls, Y_ls, label='LS', color='y', marker='o', markersize=0.1)
            fit = axis[k,h].plot(X_fit, Y_fit, label='fit', color='r', marker='o', markersize=0.1)
            

    # Make super plot title/label axes
    test_loss = test_loss.detach().numpy()
    test_loss = float(test_loss)
    test_loss_str = '{:.2e}'.format(test_loss)
    test_phase_loss = float(test_phase_loss.detach().numpy())
    test_phase_loss_str = '{:.2e}'.format(test_phase_loss)
    LS_test_loss_str = '{:.2e}'.format(LS_test_loss)
    if PLOT_MLE: MLE_test_Loss_str = '{:.2e}'.format(MLE_test_Loss)
    else: MLE_test_Loss_str = '-'

    results_txt = \
      "NN  total Test Loss [phi_d,cx,cy]: "+test_loss_str+\
    '\nNN  Phase Loss: '+str(test_phase_loss_str)+\
    '\nLS  Phase Loss: '+LS_test_loss_str+\
    '\nMLE Phase Loss: '+MLE_test_Loss_str

    #plt.figtext(0.5, -0.5, results_txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.subplots_adjust(bottom=0.2)
    plt.figtext(0.05,0.00, results_txt, fontsize=7, va="bottom", ha="left")

    plot_title = 'NN(red) vs. LS(yellow) vs. Truth(black)'
    plt.suptitle(plot_title, fontsize=11)
    plt.sca(axis[0,2])
    plt.xticks([0, 0.5, 1])
    plt.yticks([0, 0.5, 1])
    
    figure.text(0.5, 0.11, 'P1', ha='center')
    figure.text(0.02, 0.5, 'P2', va='center', rotation='vertical')

    # Make legend
    #plt.figlegend([h1[0], h2[0]], ['Known Ellipse', 'Neural Net. Fit'],bbox_to_anchor=(1.0,1), loc="upper left")
    #handles, labels = axis.get_legend_handles_labels()
    #figure.legend(handles, labels, loc='upper center')

    img = fig2img(figure)

    #plt.show()
    return img

def plot_errors(targets, outputs, Phi_LS, Phi_MLE, PLOT_MLE):
    # plotting errors from the NN, LS, and MLE algos

    # convert input coords to arrays
    if (type(targets) == torch.Tensor):
        targets = targets.detach().numpy()
    if (type(outputs) == torch.Tensor):
        outputs = outputs.detach().numpy()
    if (type(Phi_LS) == torch.Tensor):
        Phi_LS = Phi_LS.detach().numpy()
    if PLOT_MLE:
        if (type(Phi_MLE) == torch.Tensor):
            Phi_LS = Phi_MLE.detach().numpy()

    true_phis = targets[:,0]
    phi_nn = outputs[:,0]
    phi_ls = np.reshape(Phi_LS, 100,)
    if PLOT_MLE:
        phi_mle = np.reshape(Phi_MLE, 100,)
        mle_errs = phi_mle-true_phis

    nn_errs = phi_nn-true_phis
    ls_errs = phi_ls-true_phis

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(true_phis, nn_errs, s=10, facecolors='none', edgecolors='deepskyblue', marker="o", label='NN')
    ax1.scatter(true_phis, ls_errs, s=10, facecolors='none', edgecolors='orange', marker="o", label='LS')
    if PLOT_MLE:
         ax1.scatter(true_phis, mle_errs, s=10, facecolors='none', edgecolors='seagreen', marker="o", label='MLE')
    ax1.set_xlabel('True Phase')
    ax1.set_ylabel('Error')
    plt.legend(loc='upper center')
    #plt.show()

    img = fig2img(fig)
    return img