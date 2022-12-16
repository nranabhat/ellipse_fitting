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
