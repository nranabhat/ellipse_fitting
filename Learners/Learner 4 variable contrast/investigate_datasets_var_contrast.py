import loadCSVdata_var_contrast
import numpy as np
import matplotlib.pyplot as plt
import pylab

loader = loadCSVdata_var_contrast.loadCSVdata(NUM_TRAINING_ELLIPSES=500, MAX_SHOTS=500, FULL_PHI_RANGE=True)

# # gets 100 phi_d values from testing set
X,y = loader.get_test_data()
X = X.astype(float)
y = y.astype(float)
print('largest phi value in testing set: ' + str(np.amax(y)))
print('smallest phi value in testing set: ' + str(np.amin(y)))

plt.hist(y)
plt.show()

print('hi')

