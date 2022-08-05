# from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/linear_regression/main.py#L22-L23

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 500
learning_rate = 0.001

#load in ellipse data
import csv

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

n,m = (train_X.T).shape #(n,m) = (500,30)
X = train_X[:,1] # X [30,500]
y = train_Y[:,1]
X = np.reshape(X, (30,1))
y = np.reshape(y, (30,1))
x_train = X.astype(float)
y_train = y.astype(float)

# Linear regression model
model = nn.Linear(input_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train).float()
    targets = torch.from_numpy(y_train).float()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Plot the graph
predicted = model(torch.from_numpy(x_train).float()).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')