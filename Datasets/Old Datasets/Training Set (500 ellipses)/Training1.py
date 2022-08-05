# This script tests the approach by Estey in calculating the differential phase
# between two clock ensembles. It shows the approch is valid. 

import math
from math import cos, sin, acos
import numpy as np
import random as random
from matplotlib import pyplot as plt

# create 10 elliptical y-coordinates 
    # Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    #    and B^2 - 4AC < 0 => ellipse
    # y = ? (no closed form? Can't find anywhere)

# NEW approach: Just make (x,y) from Xin's paper:
    # x = (1 - C cos(theta))/2
    # y = (1 - C cos(theta + phi))/2

# NEW NEW approch: Estey's paper
# --- Set constants --- # 
CONTRAST = 0.65
c_x = CONTRAST
c_y = CONTRAST
# center at (0.5, 0.5)
b_x = 1/2 
b_y = 1/2

numPoints = 30
X = np.empty(numPoints)
Y = np.empty(numPoints)
Phi_c = np.empty(numPoints)
phi_d = math.pi/3

for i in range(numPoints):
    phi_c = random.uniform(0, 2*math.pi)
    Phi_c[i] = phi_c
    x_i = c_x * cos(phi_c + phi_d) + b_x
    X[i] = x_i
    y_i = c_y * cos(phi_c - phi_d) + b_y
    Y[i] = y_i
plt.scatter(X,Y)
#plt.show()

# Getting [A,B,C,D,E,F] from C (contrast), theta, and phi
# From Etsey:

A = 1 / ((c_x)**2)
B = -(2*cos(2*phi_d)) / (c_x*c_y)
C = 1/(c_y)**2
D = (2*b_y*cos(2*phi_d)/(c_x*c_y)) - (2*b_x)/(c_x)**2
E = (2*b_x*cos(2*phi_d)/(c_x*c_y)) - (2*b_y)/(c_y)**2
F = ((b_x)**2/((c_x)**2) + (b_y)**2/((c_y)**2) - 
(2*b_x*b_y*cos(2*phi_d))/(c_x*c_y) - 
4*(cos(phi_d))**2*(sin(phi_d))**2)

# print all 6 parameters
Params = np.array([A,B,C,D,E,F])
print('[A,B,C,D,E,F] = \n' + str(Params))

# calcualte ellipse equation to see if we found correct [ABCDEF] vector 
# pick first point to check with
#x = X[1]
#y = Y[1]
x = c_x * cos(Phi_c[0] + phi_d) + 1/2
y = c_y * cos(Phi_c[0] - phi_d) + 1/2
error = A*(x**2) + B*x*y + C*(y**2) + D*x + E*y + F
e = 10**(-5) # error threshold due to non-exact floating point arithmetic 
print(e)
if abs(error) > e:
    print("error = " + str(error) + " is not ZERO. Params vector is wrong.")
else:
    print('error is small :)')

#also calculate B^2 - 4AC
discriminant = B**2 - 4*A*C
if discriminant >= 0:
    print('discriminant = ' + str(discriminant) + ' is greater than 0 --nonellipse')
else:
    print('B^2 - 4AC is negative! :)')

# Getting phi from Etsey, once we have [ABCDEF] vector:
#    phi_d = 1/2 arccos(-C/sqrt(AB))
#    0 <= phi <= pi so 0 <= phi_d <= pi/2:
#    also, 0 <= arccos(inside_acos) <= pi
#    and -1 <= (inside_acos) <= +1
if (A*B < 0):
    print('WARNING: sqrt A*B is complex')

inside_acos = -C / (2*(A*B)**(1/2))
print(type(inside_acos))
if inside_acos < -1.0 or inside_acos > 1.0:
    print('[ABC] parameters invalid for dif. phase calculation (complex ans)')
else:
    print('phi_d calculation results in real value! :)')

phi_d_calculated = 1/2 * acos(inside_acos)
print('given phi_d: '+str(phi_d)+'\ncalculated phi_d: '+str(phi_d_calculated))
difference = phi_d - phi_d_calculated
print('Difference: ' + str(difference))


## ADD Noise 
# -> 2400 atoms per ensemble 