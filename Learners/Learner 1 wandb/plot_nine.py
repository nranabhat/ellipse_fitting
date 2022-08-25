# PLOTTING fit for LAST 9 TESTING SAMPLES
import numpy as np
import torch
from torch import nn
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

CONTRAST = 0.65
CENTER = 0.5

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

# in place to see if nn produces outputs in non-physical rangeS
def print_output_range_warning(a,b,c,d,e,f,CLAMP_EPSILON):
    ep = CLAMP_EPSILON
    ep = 0.0001 # differnt from CLAMP_EPSILON because python truncates value of a through e here.
    theNum = (CONTRAST/2)**2
    if ((a < 1/(theNum) - ep) or (a > 1/(theNum) + ep)):
        print('warning, output parameter \'a\' ('+str(a)+') does not equal '+str(1/(theNum)))
    if ((c < 1/(theNum) - ep) or (c > 1/(theNum) + ep)):
        print('warning, output parameter \'c\' ('+str(c)+') does not equal '+str(1/(theNum)))
    if ((b < -2/(theNum) - ep) or b > (2/(theNum) + ep)):
        print('warning, output parameter \'b\' ('+str(b)+') does not fit in range ['\
            +str(-2/(theNum) - ep)+', '+str((2/(theNum) + ep))+']')
    if ((d < -2/(theNum) - ep) or d > (0 + ep)):
        print('warning, output parameter \'d\' ('+str(d)+') does not fit in range ['\
            +str(-2/(theNum) - ep)+', '+str(ep)+']')
    if ((e < -2/(theNum) - ep) or e > (0 + ep)):
        print('warning, output parameter \'e\' ('+str(e)+') does not fit in range ['\
            +str(-2/(theNum) - ep)+', '+str(ep)+']')
    if ((f < -ep) or f > (2/(theNum) + ep)):
        print('warning, output parameter \'f\' ('+str(f)+') does not fit in range ['\
            +str(2/(theNum) + ep)+', '+str(-ep)+']')

def plot_nine(input_coords, target_params, output_params, test_loss, train_loss, CLAMP_EPSILON):

    m,n = 3,3 # 3x3 subplot (9 total ellipse fits) 
    figure, axis = plt.subplots(m,n, sharex='all', sharey = 'all')

    # convert input parameters to arrays
    if (type(input_coords) == torch.Tensor):
        input_coords = input_coords.detach().numpy()
    if (type(target_params) == torch.Tensor):
        target_params = target_params.detach().numpy()
    if (type(output_params) == torch.Tensor):
        output_params = output_params.detach().numpy()


    for k in range(m):
        for h in range(n):

            # Contour plot of known and fit
            #x = np.linspace(-0.20, 1.2, 400)
            #y = np.linspace(-0.20, 1.2, 400)
            min = CENTER + CONTRAST/2
            max = CENTER - CONTRAST/2
            x = np.linspace(min-0.2, max+0.2, 400)
            y = np.linspace(min-0.2, max+0.2, 400)
            x, y = np.meshgrid(x, y)

            A,B,C,D,E,F = target_params[(k+1)*(h+1),:]
            a,b,c,d,e,f = output_params[(k+1)*(h+1),:]
            print('\nNeural Net output params: '+str([a,b,c,d,e,f]))
            print('Target params:        '+str([A,B,C,D,E,F])+'\n')
            print_output_range_warning(a,b,c,d,e,f,CLAMP_EPSILON)
            
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
    #plt.xticks([])
    #plt.yticks([0.175, 0.5, 0.825])
    figure.text(0.5, 0.03, 'P1', ha='center')
    figure.text(0.02, 0.5, 'P2', va='center', rotation='vertical')

    # Make legend
    #plt.figlegend([h1[0], h2[0]], ['Known Ellipse', 'Neural Net. Fit'],bbox_to_anchor=(1.0,1), loc="upper left")
    #handles, labels = axis.get_legend_handles_labels()
    #figure.legend(handles, labels, loc='upper center')

    img = fig2img(figure)

    #plt.show()
    return img