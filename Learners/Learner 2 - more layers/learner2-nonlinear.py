# *** Moddifying github tutorial for ellipse data *** 
import torch
from torch import nn
from torch.utils.data import DataLoader
#from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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
# need data in Xw = y format where X[500,30,30] and Y[500,6].
# I'll combine train_X and train_Y into one 3d array
X = np.empty((n,m,2))
X[:,:,0] = train_X.T # X [500,30]
X[:,:,1] = train_Y.T

y = train_Labels

class Dataset(torch.utils.data.Dataset):
  '''
  Prepare the dataset for regression
  '''

  def __init__(self, X, y, scale_data=False):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      # Apply scaling if necessary
      if scale_data:
          X = StandardScaler().fit_transform(X)
      self.X = torch.from_numpy(X.astype(float))
      self.y = torch.from_numpy(y.astype(float))

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]
      
class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super(MLP, self).__init__()
    self.flatten = nn.Flatten() # differs from learner1 in that it flattens the 3D array using nn.Flatten()
    self.layer = nn.Sequential(
      nn.Linear(30*2, 60),
      nn.ReLU(),
      nn.Linear(60, 120),
      nn.ReLU(),
      nn.Linear(120, 120),
      nn.ReLU(),
      nn.Linear(120, 60),
      nn.ReLU(),
      nn.Linear(60, 60),
      nn.ReLU(),
      nn.Linear(60, 6)
    )

  def forward(self, x):
    '''
      Forward pass
    '''
    x = self.flatten(x)

    return self.layer(x)

  
if __name__ == '__main__':
  
  # Set fixed random number seed
  torch.manual_seed(42)
  
  # Load Boston dataset
  # X, y = load_boston(return_X_y=True)
  
  # Prepare dataset
  dataset = Dataset(X, y)
  trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
  
  # Initialize the MLP
  mlp = MLP()
  
  # Define the loss function and optimizer
  loss_function = nn.MSELoss()
  optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-1) # this affect step size? 
  
  # Run the training loop
  for epoch in range(0, 5): # 5 epochs at maximum
    
    # Print epoch
    print(f'Starting epoch {epoch+1}')
    
    # Set current loss value
    current_loss = 0.0
    
    # Iterate over the DataLoader for training data
    for i, data in enumerate(trainloader, 0):
      
      # Get and prepare inputs
      inputs, targets = data
      inputs, targets = inputs.float(), targets.float()
      targets = targets.reshape((targets.shape[0], 6))
      
      # Zero the gradients
      optimizer.zero_grad()
      
      # Perform forward pass
      outputs = mlp(inputs)
      
      # Compute loss
      loss = loss_function(outputs, targets)
      
      # Perform backward pass
      loss.backward()
      
      # Perform optimization
      optimizer.step()
      
      # Print statistics
      current_loss += loss.item()
      if i % 10 == 0:
          print('Loss after mini-batch %5d: %.3f' %   #UNCOMMENT TO PRINT LOSS
                (i + 1, current_loss / 500))          #UNCOMMENT TO PRINT LOSS
          current_loss = 0.0

  # Process is complete.
  
  # PLOTTING scatter plot of subset of all data, and fit for last training sample 
  X_data = train_X.T.astype(float) # [500,30]
  Y_data = train_Y.T.astype(float)
  X_sub = []
  Y_sub = []
  for i in range(100):
      X_sub = np.append(X_sub, X_data[5*i,:])
      Y_sub = np.append(Y_sub, Y_data[5*i,:])
  scatter = plt.scatter(X_sub, Y_sub, s=2, label = 'Simulated Data')

  # Contour plot of fit 
  x = np.linspace(-0.20, 1.2, 400)
  y = np.linspace(-0.20, 1.2, 400)
  x, y = np.meshgrid(x, y)

  A,B,C,D,E,F = targets[8,:].numpy()
  a, b, c, d, e, f =  outputs[8,:].detach().numpy()
  assert b**2 - 4*a*c < 0

  known = plt.contour(x, y,(A*x**2 + B*x*y + C*y**2 + D*x + E*y + F), [0], colors='k')
  fit = plt.contour(x, y,(a*x**2 + b*x*y + c*y**2 + d*x + e*y + f), [0], colors='b')
  h1,_ = known.legend_elements()
  h2,_ = fit.legend_elements()
  plt.legend([h1[0], h2[0]], ['Known Ellipse', 'Neural Net. Fit', 'Simulated Data'],bbox_to_anchor=(1.04,1), loc="upper left")

  plt.gca().set_aspect('equal')

  plt.title('Subset of Simulated Data vs. Fit')
  plt.xlabel('P1')
  plt.ylabel('P2')

  plt.show()