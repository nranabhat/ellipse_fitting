# outline of this code was taken from https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/301_regression.py
# the outline was modified to fit to ellipse data
# output is a linear-like fit going close to parallel to the major-axis.
# May 2022
# Author: Nico Ranabhat
"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.4
matplotlib
"""
from tokenize import Double
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
import numpy as np
import random
import math
from math import cos, sin, acos
from torch import nn

# Creating (x,y) from Estey approach:
    # x = c_x cos(phi_c + phi_d) + b_x
    # y = c_y cos(phi_c - phi_d) + b_y
    # Note: here Estey has c_x, c_y as Ax, Ay but I think this is the same thing

numPoints = 100 # points on each ellipse plot - number of shots measuring excitation fraction
numEllipses = 1 # number of ellipses 
X = torch.empty((numPoints, numEllipses)) # x-coordinates 
Y = torch.empty((numPoints, numEllipses)) # y-coordinates 
Phi_c = torch.empty((numPoints, numEllipses))
Phi_d = torch.empty(numEllipses) # each ellipse has a phi_d
labels = torch.empty((numEllipses, 6)) # 6 parameters for each ellipse

# --- Set constants --- # 
CONTRAST = 0.65
c_x = CONTRAST
c_y = CONTRAST
# center at (0.5, 0.5)
b_x = 1/2 
b_y = 1/2

# make numEllipses (1) plots with different phi_d:
for j in range(numEllipses):
    # 0 < phi_d < pi/2 (gets full range of cos when phi_c = pi/2)
    phi_d = math.pi/2*torch.rand(1,1)
    Phi_d[j] = phi_d
    for i in range(numPoints):
        phi_c = random.uniform(0, 2*math.pi)
        Phi_c[i,j] = phi_c
        x_i = c_x * cos(phi_c + phi_d) + b_x
        X[i,j] = x_i
        y_i = c_y * cos(phi_c - phi_d) + b_y
        Y[i,j] = y_i
    # create known_labels using Estey equations 
    A = 1 / ((c_x)**2)
    B = -(2*cos(2*phi_d)) / (c_x*c_y)
    C = 1/(c_y)**2
    D = (2*b_y*cos(2*phi_d)/(c_x*c_y)) - (2*b_x)/(c_x)**2
    E = (2*b_x*cos(2*phi_d)/(c_x*c_y)) - (2*b_y)/(c_y)**2
    F = ((b_x)**2/((c_x)**2) + (b_y)**2/((c_y)**2) - 
    (2*b_x*b_y*cos(2*phi_d))/(c_x*c_y) - 
    4*(cos(phi_d))**2*(sin(phi_d))**2)

    assert B**2 - 4*A*C < 0

    coefficients = torch.tensor([A,B,C,D,E,F])
    labels[j] = coefficients

# now we have X[100,1] and Y[100,1] data, labels[1,6], Phi_c[100,1] Phi_d[1,]
plt.scatter(X, Y)
plt.show()
x = X
y = Y

# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
#         self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

#     def forward(self, x):
#         x = F.ReLU(self.hidden(x))      # activation function for hidden layer
#         x = self.predict(x)             # linear output
#         return x
class Net(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(1, 60),
      nn.ReLU(),
      nn.Linear(60, 30),
      nn.ReLU(),
      nn.Linear(30, 1)
    )

  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)

net = Net()     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()   # something about plotting

for t in range(200):
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=1)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()