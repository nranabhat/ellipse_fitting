# PLOTTING fit for LAST 9 TESTING SAMPLES
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def plot_nine(input_coords, target_params, output_params):

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

            # Contour plot of knwon and fit
            x = np.linspace(-0.20, 1.2, 400)
            y = np.linspace(-0.20, 1.2, 400)
            x, y = np.meshgrid(x, y)

            A,B,C,D,E,F = target_params[(k+1)*(h+1),:]
            a,b,c,d,e,f = output_params[(k+1)*(h+1),:]
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

    # Make super plot title/lavel axes
    plt.suptitle('Known Ellipse (black) vs. Fit (blue)', fontsize=14)
    figure.text(0.5, 0.04, 'P1', ha='center')
    figure.text(0.04, 0.5, 'P2', va='center', rotation='vertical')

    # Make legend
    #plt.figlegend([h1[0], h2[0]], ['Known Ellipse', 'Neural Net. Fit'],bbox_to_anchor=(1.0,1), loc="upper left")
    #handles, labels = axis.get_legend_handles_labels()
    #figure.legend(handles, labels, loc='upper center')

    img = fig2img(figure)

    #plt.show()
    return img