import math
from math import cos,sin
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

contrast = 0.65
c_x = contrast/2
c_y = contrast/2
b_x = 0.5
b_y = 0.5
phi_d = np.linspace(0, math.pi/2, 400)
A = np.empty(400)
B = np.empty(400)
C = np.empty(400)
D = np.empty(400)
E = np.empty(400)
F = np.empty(400)
for i in range(400):
    #B[i] = -(2*cos(2*phi_d[i])) / (c_x*c_y)
    #D[i] = (2*b_y*cos(2*phi_d[i])/(c_x*c_y)) - (2*b_x)/(c_x)**2
    #E[i] = (2*b_x*cos(2*phi_d[i])/(c_x*c_y)) - (2*b_y)/(c_y)**2
    F[i] = ((b_x)**2/((c_x)**2) + (b_y)**2/((c_y)**2) - (2*b_x*b_y*cos(2*phi_d[i]))/(c_x*c_y) - 4*(cos(phi_d[i]))**2*(sin(phi_d[i]))**2)

PARAM = F
plt.scatter(phi_d, PARAM, c ="blue")
plt.show()

min = np.amin(PARAM)
print(min)
max = np.amax(PARAM)
print(max)
