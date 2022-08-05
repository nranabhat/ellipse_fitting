# reasonable ranges for conic parameters
from cmath import cos, sin
import math

contrast = 0.65
c_x = contrast/2
c_y = contrast/2
b_x = 0.5
b_y = 0.5
#phi_d = math.pi/2
phi_d = 0

# given these ranges, finding reasonable ranges for the conic parameters 
A = 1 / ((c_x**2))
B = -(2*cos(2*phi_d)) / (c_x*c_y)
C = 1/(c_y)**2
D = (2*b_y*cos(2*phi_d)/(c_x*c_y)) - (2*b_x)/(c_x)**2
E = (2*b_x*cos(2*phi_d)/(c_x*c_y)) - (2*b_y)/(c_y)**2
F = ((b_x)**2/((c_x)**2) + (b_y)**2/((c_y)**2) - 
(2*b_x*b_y*cos(2*phi_d))/(c_x*c_y) - 
4*(cos(phi_d))**2*(sin(phi_d))**2)

print('A:  '+str(A))
print('B:  '+str(B))
print('C:  '+str(C))
print('D:  '+str(D))
print('E:  '+str(E))
print('F:  '+str(F))