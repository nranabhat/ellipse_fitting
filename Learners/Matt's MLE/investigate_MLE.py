import ellipse_fitting_api
import ellipse_fitting2_api
import numpy as np
import loadCSVdata_var_contrast

num_ellipses = 100
MAX_SHOTS = 500
FULL_PHI_RANGE = True
LAB_COMP = False
VARIABLE_CONTRAST = True

# load 100 test ellipses 
loader = loadCSVdata_var_contrast.loadCSVdata(num_ellipses, MAX_SHOTS, FULL_PHI_RANGE, LAB_COMP, VARIABLE_CONTRAST)
X,y = loader.get_test_data()

targets = y
X = X.astype(float)
targets = targets.astype(float)
#contrast = 0.65
num_atoms = 1000
Phi_MLE = np.zeros(targets.shape[0])
Contrast_MLE = np.zeros(targets.shape[0])
num_shots = 0
# loop through 100 ellipses in testing set
for i in range(targets.shape[0]):
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
    
    contrast = 2 * targets[i,1]
    phi_estimate, contrast_estimate = ellipse_fitting2_api.main(points, num_atoms)
    Phi_MLE[i] = phi_estimate
    Contrast_MLE[i] = contrast_estimate

Phi_MLE = np.reshape(Phi_MLE, (Phi_MLE.shape[0],))
Contrast_MLE = np.reshape(Contrast_MLE, (Contrast_MLE.shape[0],))

phase_MSE = np.linalg.norm(targets[:,0]-Phi_MLE)**2 / len(targets[:,0])
contrast_MSE = np.linalg.norm(2*targets[:,1]-Contrast_MLE)**2 / len(targets[:,1])

print('done')