# *** Adding wandb to github tutorial for experiment tracking *** 

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import wandb

# files I (Nico) wrote: 
from loadCSVdata import get_train_data, get_x_train_coords, get_y_train_coords, get_test_data, get_x_test_coords, get_y_test_coords
from plot_nine import plot_nine

wandb.login()

# (most) hyperparameters logged in config 
config = dict(
    epochs=25,
    milestones=[5,10,15],
    batch_size=10,
    starting_LR=0.1,
    architecture="MLP")

# load data
X,y = get_train_data()
train_X = get_x_train_coords()
train_Y = get_y_train_coords()

X_test,y_test = get_test_data()
test_X = get_x_test_coords()
test_Y = get_y_test_coords()


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
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(60, 60),
      nn.ReLU(),
      nn.Linear(60, 30),
      nn.ReLU(),
      nn.Linear(30, 6)
    )

  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)


# TODO: implement ONNX format in test()
def test(mlp, test_loader):
  mlp.eval()
  print('\n********Beginning testing********* \n')

  # Run the model on the test examples
  with torch.no_grad():
    test_sample_ct = 0
    for i, data in enumerate(test_loader, 0):
      inputs, targets = data
      inputs, targets = inputs.float(), targets.float()
      targets = targets.reshape((targets.shape[0], 6))

      outputs = mlp(inputs)
      loss_function = nn.MSELoss()
      loss = loss_function(outputs, targets)

      # Print/log statistics
      test_sample_ct += 1
      #if i % 10 == 0:
      wandb.log({"test loss": loss}, step=test_sample_ct, commit=True)
      print(f"Loss after " + str(test_sample_ct).zfill(5) + f" TEST examples: {loss:.3f}")

  # *****  log plots  *****
  # create subplot of 9 fits 
  m,n = 3,3 # 3x3 subplot (9 total ellipse fits) 
  figure, axis = plt.subplots(m,n, sharex='all', sharey = 'all')
  nine_plot = plot_nine(figure, axis, inputs, targets, outputs) # type PIL image

  image = wandb.Image(nine_plot)
  wandb.log({"Known Ellipse (black) vs. Fit (blue) for 9 testing samples": image}, commit=True)

  # TODO: implement the ONNX format after exploring sweeps 
  # Save the model in the exchangeable ONNX format 
  #torch.onnx.export(mlp, inputs, "model.onnx")
  #wandb.save("model.onnx")


if __name__ == '__main__':
  with wandb.init(project="ellipse_fitting", config=config):
    # access all HPs through wandb.config, so logging matches execution!
    config = wandb.config

    # Set fixed random number seed - for reproducability 
    #torch.manual_seed(42)
    
    # Prepare dataset
    dataset = Dataset(X, y)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)
    
    dataset_test = Dataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=config.batch_size, shuffle=True, num_workers=1)

    # Initialize the MLP
    mlp = MLP()
    
    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=config.starting_LR) # this affect step size? 

    milestones = config.milestones # at these ephochs the learning rate decreases 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5) 

    wandb.watch(mlp, loss_function, log="all", log_freq=1)

    sample_ct = 0
    # Run the training loop
    for epoch in range(0, config.epochs): 
      
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
        scheduler.step()

        # Print statistics
        #current_loss += loss.item()
        sample_ct += 1

        if i % 10 == 0:
          wandb.log({"epoch": epoch, "loss": loss, "LR": scheduler.get_last_lr()}, step=sample_ct, commit=True)
          print(f"Loss after " + str(sample_ct).zfill(5) + f" examples: {loss:.3f}")
          #print('Loss after mini-batch %5d: %.3f' %   #UNCOMMENT TO PRINT LOSS
                #(i + 1, current_loss / 500))          #UNCOMMENT TO PRINT LOSS
          #current_loss = 0.0
    
    print('\nTraining process has finished.')
    test(mlp, test_loader)
  
  # Process is complete.
  print('\nTraining and testing has finished.')


"""   # PLOTTING scatter plot of subset of all data, and fit for last training sample 
  # create subplot of 9 fits from the last batch
  m,n = 3,3 # 3x3 subplot (9 total ellipse fits) 
  figure, axis = plt.subplots(3,3, sharex='all')

  for k in range(m):
      for h in range(n):
          X_data = train_X.T.astype(float) # [500,30]
          Y_data = train_Y.T.astype(float)
          X_sub = []
          Y_sub = []
          for i in range(100):
              X_sub = np.append(X_sub, X_data[5*i,:])
              Y_sub = np.append(Y_sub, Y_data[5*i,:])
          #scatter = axis[k,h].scatter(X_sub, Y_sub, s=2, label = 'Simulated Data')

          # Contour plot of fit 
          x = np.linspace(-0.20, 1.2, 400)
          y = np.linspace(-0.20, 1.2, 400)
          x, y = np.meshgrid(x, y)

          A,B,C,D,E,F = targets[(k+1)*(h+1),:].numpy()
          a, b, c, d, e, f =  outputs[(k+1)*(h+1),:].detach().numpy()
          assert b**2 - 4*a*c < 0

          # Scatter plot of ellipse points 

          data = axis[k,h].scatter(X_data, Y_data, s=10, label='scatter')
          known = axis[k,h].contour(x, y,(A*x**2 + B*x*y + C*y**2 + D*x + E*y + F), [0], colors='k')
          fit = axis[k,h].contour(x, y,(a*x**2 + b*x*y + c*y**2 + d*x + e*y + f), [0], colors='b')
          h1,_ = known.legend_elements()
          h2,_ = fit.legend_elements()

  # Make super plot title/lavel axes
  plt.suptitle('Known Ellipse (black) vs. Fit (blue)', fontsize=14)
  figure.text(0.5, 0.04, 'P1', ha='center')
  figure.text(0.04, 0.5, 'P2', va='center', rotation='vertical')

  plt.show() """