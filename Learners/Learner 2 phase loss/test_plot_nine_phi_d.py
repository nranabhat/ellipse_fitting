from plot_nine_phi_d import plot_nine
import loadCSVdata_phi_d
import numpy as np
import matplotlib.pyplot as plt
import torch

num_training_ellipses = 100
num_points = 30
loader = loadCSVdata_phi_d.loadCSVdata(num_training_ellipses, num_points)

input_coords, target_params = loader.get_test_data()
input_coords, target_params = np.asfarray(input_coords), np.asfarray(target_params)
target_params = target_params.reshape((target_params.shape[0], 6))
output_b = target_params[:,1] + 0.5

m,n = 3,3 # 3x3 subplot (9 total ellipse fits) 
figure, axis = plt.subplots(m,n, sharex='all', sharey = 'all')
nine_plot = plot_nine(input_coords, target_params, output_b, test_loss=torch.tensor(0.12345), train_loss='0.12345', CLAMP_EPSILON=0)

nine_plot.show()

#plot = plot_nine(input_coords, target_params, output_params)
#plot.show()