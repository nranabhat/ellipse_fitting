# PLOTTING fit for LAST 9 TESTING SAMPLES
from cmath import acos, cos, sin
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
def print_output_range_warning(b,CLAMP_EPSILON):
    ep = CLAMP_EPSILON
    ep = 0.0001 # differnt from CLAMP_EPSILON because python truncates value of a through e here.
    theNum = (CONTRAST/2)**2
    if ((b.real < -2/(theNum) - ep) or b.real > (2/(theNum) + ep)):
        print('warning, output parameter \'b\' ('+str(b)+') does not fit in range ['\
            +str(-2/(theNum) - ep)+', '+str((2/(theNum) + ep))+']')

def plot_nine(input_coords, target_params, output_phi_d, test_loss, train_loss, CLAMP_EPSILON):

    m,n = 3,3 # 3x3 subplot (9 total ellipse fits) 
    figure, axis = plt.subplots(m,n, sharex='all', sharey = 'all')

    # convert input parameters to arrays
    if (type(input_coords) == torch.Tensor):
        input_coords = input_coords.detach().numpy()
    if (type(target_params) == torch.Tensor):
        target_params = target_params.detach().numpy()
    if (type(output_phi_d) == torch.Tensor):
        output_phi_d = output_phi_d.detach().numpy()


    for k in range(m):
        for h in range(n):

            # Contour plot of known and fit
            #x = np.linspace(-0.20, 1.2, 400)
            #y = np.linspace(-0.20, 1.2, 400)
            min = CENTER + CONTRAST/2
            max = CENTER - CONTRAST/2
            x = np.linspace(min, max, 400)
            y = np.linspace(min, max, 400)
            x, y = np.meshgrid(x, y)

            A,B,C,D,E,F = target_params[(k+1)*(h+1),:]
            b_output = output_phi_d[(k+1)*(h+1)]
            # make sure phase angle is the same sign
            if np.sign(b_output) != np.sign(B):
                b_output = -b_output

            a = 1 / ((c_x)**2)
            c = 1 / ((c_y)**2)
            phi_d = 0.5*acos(-b_output/(2*((a*c)**(1/2))))

            b = -(2*cos(2*phi_d)) / (c_x*c_y)
            d = (2*b_y*cos(2*phi_d)/(c_x*c_y)) - (2*b_x)/(c_x)**2
            e = (2*b_x*cos(2*phi_d)/(c_x*c_y)) - (2*b_y)/(c_y)**2
            f = ((b_x)**2/((c_x)**2) + (b_y)**2/((c_y)**2) - 
            (2*b_x*b_y*cos(2*phi_d))/(c_x*c_y) - 
            4*(cos(phi_d))**2*(sin(phi_d))**2)
            
            target_phase = 0.5*acos(-B/(2*((A*C)**(1/2))))
            print('\nNeural Net output param b: '+str(b))
            print('Target params b:        '+str(B))
            print('Neural Net output phase: '+str(phi_d))
            print('Target phase:        '+str(target_phase))
            loss_funct = nn.MSELoss()
            phase_loss = loss_funct(target_phase, phi_d)
            print('Total phase loss: '+str(phase_loss)+'\n')
            print_output_range_warning(b,CLAMP_EPSILON)
            
            # assert b**2 - 4*a*c < 0
            # Scatter plot of ellipse points 
            num_points = int(input_coords.shape[1] / 2)
            x_points = input_coords[(k+1)*(h+1), 0:num_points]
            y_points = input_coords[(k+1)*(h+1), num_points:2*num_points]

            data = axis[k,h].scatter(x_points, y_points, s=10, label='scatter')
            known = axis[k,h].contour(x, y,(A*x**2 + B*x*y + C*y**2 + D*x + E*y + F), [0], colors='k')
            fit = axis[k,h].contour(x, y,(a*x**2 + b*x*y + c*y**2 + d*x + e*y + f), [0], colors='b')
            h1,_ = known.legend_elements()
            h2,_ = fit.legend_elements()

    # Make super plot title/label axes
    test_loss_str = str(test_loss.detach().numpy())
    plot_title = 'Fit (blue) vs. Truth (black). Test Loss: '+test_loss_str[0:5]+' Train Loss: '+train_loss[0:5]
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