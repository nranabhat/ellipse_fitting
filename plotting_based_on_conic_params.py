# This program is made to practice plotting ellipse fitting data.
# I start by loading in 500 ellipses made up of 30 points each (30 points that lie directly on the ellipse,
# without nosie). I then plot a scatter plot of all the ellipses, and a contour with ellipse parameters obtained 
# from leaner1.py (I suspect the learner1.py produces ellipse parameters that result in an ellipe that averages
# over all the data).

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv

# load data set 1 so I can plot all ellipses at once
train_X = []
train_Y = []
train_Labels = []

# put X-coordinates into numpy array train_X
with open('training1X.csv') as training1X_csv:
  reader = csv.reader(training1X_csv)#, delimiter=' ', quotechar='|')
  for row in reader:
    train_X = np.append(train_X, row)
train_X = np.reshape(train_X, (30,500)) # will need to change shape if using other data

# put Y-coordinates into numpy array train_Y
with open('training1Y.csv') as training1Y_csv:
  reader = csv.reader(training1Y_csv)#, delimiter=' ', quotechar='|')
  for row in reader:
    train_Y = np.append(train_Y, row)
train_Y = np.reshape(train_Y, (30,500)) # will need to change shape if using other data

# put labels into numpy array train_labels
with open('training1Labels.csv') as training1Labels_csv:
  reader = csv.reader(training1Labels_csv)#, delimiter=' ', quotechar='|')
  for row in reader:
    train_Labels = np.append(train_Labels, row)
train_Labels = np.reshape(train_Labels, (500,6), order = 'F') # will need to change shape if using other data

X = train_X.T
X = X.flatten() # X [15000,1]
Y = train_Y.T 
Y = Y.flatten() # Y [15000,1]
y_labels = train_Labels

# PLOT 
mpl.rcParams['lines.color'] = 'k'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k'])
x = np.linspace(0, 1.2, 400)
y = np.linspace(0, 1.2, 400)
x, y = np.meshgrid(x, y)

A,B,C,D,E,F = 2.36686391, -2.59041463, 2.36686391, -3.336589, -3.336589,  2.27140236
a, b, c, d, e, f =  2.2542, -0.0514,  2.2502, -2.2711, -2.2655,  0.6487
assert b**2 - 4*a*c < 0

known = plt.contour(x, y,(A*x**2 + B*x*y + C*y**2 + D*x + E*y + F), [0], colors='k')
fit = plt.contour(x, y,(a*x**2 + b*x*y + c*y**2 + d*x + e*y + f), [0], colors='b')
h1,_ = known.legend_elements()
h2,_ = fit.legend_elements()
plt.legend([h1[0], h2[0]], ['Known Ellipse', 'Fit'])

plt.scatter(X, Y, alpha=0.10, s=2)
plt.gca().set_aspect('equal')

plt.show()


