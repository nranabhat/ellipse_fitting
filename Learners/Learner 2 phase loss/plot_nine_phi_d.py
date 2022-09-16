# PLOTTING fit for LAST 9 TESTING SAMPLES
from math import acos, cos, sin
import math
import numpy as np
import torch
from torch import nn
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

CONTRAST = 0.65
CENTER = 0.5
b_y = CENTER
b_x = CENTER
c_x = CONTRAST/2
c_y = CONTRAST/2

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

def plot_nine(input_coords, target_phase, output_phi_d, test_loss, train_loss, LS_test_loss, CLAMP_EPSILON):

    m,n = 3,3 # 3x3 subplot (9 total ellipse fits) 
    figure, axis = plt.subplots(m,n, sharex='all', sharey = 'all')

    # convert input coords to arrays
    if (type(input_coords) == torch.Tensor):
        input_coords = input_coords.detach().numpy()
    if (type(target_phase) == torch.Tensor):
        target_phase = target_phase.detach().numpy()
    if (type(output_phi_d) == torch.Tensor):
        output_phi_d = output_phi_d.detach().numpy()


    for k in range(m):
        for h in range(n):

            # Contour plot of known and fit
            # min = CENTER + CONTRAST/2
            # max = CENTER - CONTRAST/2
            # x = np.linspace(min, max, 400)
            # y = np.linspace(min, max, 400)
            # x, y = np.meshgrid(x, y)

            phi_d_target = target_phase[(k+1)*(h+1)]
            phi_d_output = output_phi_d[(k+1)*(h+1)]

            print('\nNeural Net output phase: '+str(phi_d_output))
            print('Target phase:        '+str(phi_d_target))

            phase_loss = np.linalg.norm(phi_d_target - phi_d_output)
            print('Total phase loss: '+str(phase_loss)+'\n')
            print_output_range_warning(phi_d_output, CLAMP_EPSILON)
            
            # Scatter plot of ellipse points 
            num_points = int(input_coords.shape[1] / 2)
            x_points = input_coords[(k+1)*(h+1), 0:num_points]
            y_points = input_coords[(k+1)*(h+1), num_points:2*num_points]
            data = axis[k,h].scatter(x_points, y_points, s=10, label='data')

            # Scatter plot of known and fit based off of phase and known contrast 
            Phi_c = np.linspace(0, 2*math.pi, 600)
            X_fit = np.empty(600)
            Y_fit = np.empty(600)
            X_known = np.empty(600)
            Y_known = np.empty(600)
            for i in range(600): 
                X_fit[i] = c_x*cos(Phi_c[i] + phi_d_output) + b_x
                Y_fit[i] = c_y*cos(Phi_c[i] - phi_d_output) + b_y
                X_known[i] = c_x*cos(Phi_c[i] + phi_d_target) + b_x
                Y_known[i] = c_y*cos(Phi_c[i] - phi_d_target) + b_y
            known = axis[k,h].plot(X_known, Y_known, label='known', color='k')
            fit = axis[k,h].plot(X_fit, Y_fit, label='fit', color='b')

    # Make super plot title/label axes
    test_loss = test_loss.detach().numpy()
    test_loss_str = str(test_loss*10**4)[0:5]+'e-4'
    #train_loss = train_loss.detach().numpy()
    train_loss_str = str(train_loss*10**4)[0:5]+'e-4'
    LS_test_loss_str = str(LS_test_loss*10**4)[0:5]+'e-4'
    plot_title = 'Fit (blue) vs. Truth (black). Test Loss: '+test_loss_str+'\nLS Test Loss: '+LS_test_loss_str
    plt.suptitle(plot_title, fontsize=14)
    plt.sca(axis[0,2])
    plt.xticks([0.25, 0.5, 0.75])
    plt.yticks([0.25, 0.5, 0.75])
    
    figure.text(0.5, 0.03, 'P1', ha='center')
    figure.text(0.02, 0.5, 'P2', va='center', rotation='vertical')

    # Make legend
    #plt.figlegend([h1[0], h2[0]], ['Known Ellipse', 'Neural Net. Fit'],bbox_to_anchor=(1.0,1), loc="upper left")
    #handles, labels = axis.get_legend_handles_labels()
    #figure.legend(handles, labels, loc='upper center')

    img = fig2img(figure)

    #plt.show()
    return img