from plot_nine import plot_nine
import loadCSVdata
import numpy as np
import matplotlib.pyplot as plt

num_training_ellipses = 500
num_points = 30
loader = loadCSVdata.loadCSVdata(num_training_ellipses, num_points)

input_coords, target_params = loader.get_test_data()
input_coords, target_params = np.asfarray(input_coords), np.asfarray(target_params)
target_params = target_params.reshape((target_params.shape[0], 6))
output_params = target_params + 0.0025

m,n = 3,3 # 3x3 subplot (9 total ellipse fits) 
figure, axis = plt.subplots(m,n, sharex='all', sharey = 'all')
nine_plot = plot_nine(input_coords, target_params, output_params)

nine_plot.show()

#plot = plot_nine(input_coords, target_params, output_params)
#plot.show()