import ellipse_fitting_api
import numpy as np
import loadCSVdata_var_contrast

num_ellipses = 100
MAX_SHOTS = 500
FULL_PHI_RANGE = True
LAB_COMP = False
VARIABLE_CONTRAST = False

# need to pass these params to ellipse_fitting_api.main()
"""     Parameters
    ----------
    points : list
        List of coordinates of experimental data points
    contrast : float
        Ellipse contrast, between 0 and 1
    num_atoms : int
        Number of atoms in the experiments

    Returns
    -------
    float
        Maximum likelihood estimator for phi
 """

loader = loadCSVdata_var_contrast.loadCSVdata(num_ellipses, MAX_SHOTS, FULL_PHI_RANGE, LAB_COMP, VARIABLE_CONTRAST)
X,y = loader.get_test_data()
X = X.astype(float)
y = y.astype(float)
print('data obtained')
# X is (100, 1000), y is (100, 3). 
# first 500 X are x-coordinates. Next 500 are y-coordinates
# y[:,0] are phi values
# y[:,1] is cx, 2 is cy

# now we obtain phi values for the test set using MLE:

contrast = 0.65
num_atoms = 1000
MLE_phi = np.zeros(num_ellipses)
num_shots = 0
# loop through 100 ellipses in testing sest
for i in range(num_ellipses):
    # loop through 500 x-points in X
    for j in range(MAX_SHOTS):
        # find the number of non-zero entries (how many points the ellipse has)
        if X[i,j] == 0: 
            num_shots = j+1
            break
    points_x = X[i, 0:num_shots-1]
    points_y = X[i, MAX_SHOTS:MAX_SHOTS+num_shots-1]
    points = [points_x, points_y] # need the transpose of this
    points = [list(k) for k in zip(*points)]
    
    phi_estimate = ellipse_fitting_api.main(points, contrast, num_atoms)
    MLE_phi[i] = phi_estimate

print(MLE_phi)