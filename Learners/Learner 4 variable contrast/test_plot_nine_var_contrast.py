""" 
    Tester script for plot_nine_var_contrast.py.
    Mostly used to test plot formatting 
    Created October 2022
    @author: nranabhat
 """

from matplotlib.container import Container
import loadCSVdata_var_contrast
import numpy as np
import matplotlib.pyplot as plt
import torch
from plot_nine_var_contrast import plot_nine
from plot_nine_var_contrast import plot_errors

# define constants, dataloader
NUM_ELLIPSES = 100
NUM_POINTS = 500
loader = loadCSVdata_var_contrast.loadCSVdata(NUM_ELLIPSES, NUM_POINTS)

### loading ellipse coordinates and truth parameters.
# input coords shape: [num_ellipses, num_points*2]
# target_params shape : [num_ellipses, 3]
    # target_params[i,0] => phi_d
    # target_params[i,1] => c_x
    # target_params[i,1] => c_y
input_coords, target_params = loader.get_test_data()
input_coords, target_params = np.asfarray(input_coords), np.asfarray(target_params) 

# make NN outputs and LS outputs be some offset from the truth
nn_errs = np.empty((NUM_ELLIPSES,3))
ls_errs = np.empty(NUM_ELLIPSES)
for i in range(NUM_ELLIPSES):
    nn_errs[i,:] = np.random.uniform(-0.1,0.1)
    ls_errs[i] = np.random.uniform(-0.05,0.05)

outputs = target_params - nn_errs
Phi_LS = target_params[:,0] - ls_errs

# plotting with plot_nine
m,n = 3,3 # 3x3 subplot (9 total ellipse fits) 
figure, axis = plt.subplots(m,n, sharex='all', sharey = 'all')
test_loss=torch.tensor(0.12345)
train_loss=0.12345
LS_test_loss=0.000377

# input parameters of plot_nine() function: 
#       1. Coordinates of points on each ellipse
#       2. True [phi_d, cx, cy] values
#       3. NN   [phi_d, cx, cy] values
#       4. LS   [phi_d]         values
#nine_plot = plot_nine(input_coords, target_params, outputs, Phi_LS, test_loss, 
#test_phase_loss=0.0001, train_loss=0.0002, LS_test_loss=0.0003, CLAMP_EPSILON=0)
#nine_plot.show()

errors_plot = plot_errors(target_params, outputs, Phi_LS)
errors_plot.show()

#plot = plot_nine(input_coords, target_params, output_params)
#plot.show()