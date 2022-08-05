import math
from math import cos,sin
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

contrast = 0.65
c_x = contrast
c_y = contrast
b_x = 0.5
b_y = 0.5
phi_d = np.linspace(0, math.pi/2, 400)
F = np.empty(400)
for i in range(400):
    F[i] = ((b_x)**2/((c_x)**2) + (b_y)**2/((c_y)**2) - (2*b_x*b_y*cos(2*phi_d[i]))/(c_x*c_y) - 4*(cos(phi_d[i]))**2*(sin(phi_d[i]))**2)

plt.scatter(phi_d, F, c ="blue")
plt.show()

min = np.amin(F)
print(min)
max = np.amax(F)
print(max)