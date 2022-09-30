import loadCSVdata_var_input
import numpy as np
import matplotlib.pyplot as plt
import torch
from plot_nine_var_input import plot_nine

num_training_ellipses = 100
num_points = 30
loader = loadCSVdata_var_input.loadCSVdata(num_training_ellipses, num_points)

input_coords, target_params = loader.get_test_data()
Phi_d_target = loader.get_test_phi_d()
input_coords, target_params = np.asfarray(input_coords), np.asfarray(target_params)
Phi_d_target = np.asfarray(Phi_d_target)
Phi_d_target = Phi_d_target.astype(float)
#target_params = target_params.reshape((target_params.shape[0], 6))
Phi_d_target = Phi_d_target.reshape((Phi_d_target.shape[0], 1))
Phi_d_output = Phi_d_target - 0.01
Phi_d_LS = Phi_d_target - 0.05

m,n = 3,3 # 3x3 subplot (9 total ellipse fits) 
figure, axis = plt.subplots(m,n, sharex='all', sharey = 'all')
test_loss=torch.tensor(0.12345)
train_loss=0.12345
LS_test_loss=0.000377
nine_plot = plot_nine(input_coords, Phi_d_target, Phi_d_output, test_loss, train_loss, LS_test_loss, Phi_d_LS, CLAMP_EPSILON=0)
nine_plot.show()

#plot = plot_nine(input_coords, target_params, output_params)
#plot.show()