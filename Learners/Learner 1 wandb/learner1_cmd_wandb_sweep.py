# *** Adding wandb to github tutorial for experiment tracking *** 

import functools
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import wandb
import os

# set logging
import logging
logging.getLogger().setLevel(logging.INFO)

# files I (Nico) wrote: 
from loadCSVdata import get_train_data, get_x_train_coords, get_y_train_coords, get_test_data, get_x_test_coords, get_y_test_coords
from plot_nine import plot_nine

#wandb.login()

""" def config_params():

  sweep_config = {
      'method': 'random'
      }

  metric = {
      'name': 'loss',
      'goal': 'minimize'   
      }

  sweep_config['metric'] = metric

  parameters_dict = {
      'epochs': {
          'value': 5      # change this to >15 later
          },
      'batch_size': {
          # integers between 5 and 30
          # with evenly-distributed logarithms 
          'distribution': 'q_log_uniform_values',
          'q': 5,
          'min': 5,
          'max': 30,
        },
      'optimizer': {
          'values': ['adam', 'sgd']
          },
      'second_layer_size': {
          'values': [128, 256, 512]
          },
      'starting_lr': {
          # a flat distribution between 0.01 and 1
          'distribution': 'uniform',
          'min': 0.01,
          'max': 1
        },
      'milestones' : {
            'values': [[5,10,15], [5,7,10], [3,7,15]]
          },
      }

  sweep_config['parameters'] = parameters_dict

  parameters_dict.update({
      'gamma': {
          # a flat distribution between 0.01 and 1
          'distribution': 'uniform',
          'min': 0.01,
          'max': 1
        }
      })

  import pprint
  #pprint.pprint(sweep_config)

  sweep_id = wandb.sweep(sweep_config, project="ellipse_fitting")

  return sweep_id
 """

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class from https://gist.github.com/amaarora/9b867f1868f319b3f2e6adb6bfe2373e#file-how-to-save-all-your-trained-model-weights-locally-after-every-epoch-ipynb
class CheckpointSaver:
    def __init__(self, dirpath, decreasing=True, top_n=1):
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n 
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf
        
    def __call__(self, model, metric_val):
        model_path = os.path.join(self.dirpath, model.__class__.__name__ + f'weights_sweep{sweep_id}.pt')
        save = metric_val<=self.best_metric_val if self.decreasing else metric_val>=self.best_metric_val
        if save: 
            logging.info(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}, & logging model weights to W&B.")
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            self.log_artifact(f'best-mpl-sweep-{sweep_id}.pt', model_path, metric_val)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)
        if len(self.top_model_paths)>self.top_n: 
            self.cleanup()
    
    def log_artifact(self, filename, model_path, metric_val):
        artifact = wandb.Artifact(filename, type='model', metadata={'loss': metric_val})
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)        
    
    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]

# set hyperparemeters
sweep_id = config_params()

# create pathname to save weights to
# wandb path
wandbpath = r"C:\Users\Nicor\OneDrive\Documents\KolkowitzLab\Ellipse fitting\Learners\wandb"
# sweep path
pathname = os.path.join(wandbpath, 'sweep-'+sweep_id)

# instantiate CheckpointSaver object with pathname
checkpoint_saver = CheckpointSaver(dirpath=pathname, decreasing=True, top_n=1)

#def train(checkpoint_saver, config=None):
def train(config=None):

    # set seed for reproducibility - see if we can reconstruct saved model to and produce same results 
    torch.manual_seed(42)

    # Initialize a new wandb run
    with wandb.init(config=config) as run: # jump into to change init directory
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        trainloader = build_dataset(config.batch_size)
        network = build_network(config.second_layer_size)
        optimizer = build_optimizer(network, config.optimizer, config.starting_lr)
        scheduler = build_scheduler(optimizer, config.milestones, config.gamma)

        # find particular sweep in wandb 
        api = wandb.Api()
        sweep = api.sweep(f"{'nicoranabhat'}/{'ellipse_fitting'}/{sweep_id}")

        # # create pathname to save weights to
        # pathname = os.path.join(run.dir, "model_weights")

        # # instantiate CheckpointSaver object with pathname (commented out b/c we only want 1 cps object)
        #checkpoint_saver = CheckpointSaver(dirpath=pathname, decreasing=True, top_n=1) # jump-into to see if pathname is initialized correctly

        for epoch in range(config.epochs):

            avg_loss = train_epoch(network, trainloader, optimizer, scheduler)

            # after epoch log loss to wandb
            wandb.log({"loss": avg_loss, "epoch": epoch})

            # update best_loss for next call to checkpoint_saver
            best_loss = sweep.best_run().summary_metrics['loss'] 

            # save weights of current best epoch (will save best model for whole network over training if top_n=1)
            if avg_loss <= best_loss:
              checkpoint_saver(network, avg_loss)

        # # SAVE BEST CURRENT MODEL ATTEMPT 1 as artifact
        # # checkpointSaver class replaces this code. 
        # api = wandb.Api()
        # sweep = api.sweep(f"{'nicoranabhat'}/{'ellipse_fitting'}/{sweep_id}")
        # best_loss = sweep.best_run().summary_metrics['loss']

        # model_version = 0
        # if float(avg_loss) <= float(best_loss):

        #   # Store model artifact
        #   trained_model_artifact = wandb.Artifact('sweep_'+sweep.name+'_best_current_mlp', type="model", metadata = dict(config))

        #   wandb.save(glob_str = os.path.join(run.dir, "model.v"+str(model_version)), base_path = run.dir, policy = 'end') 
        #   trained_model_artifact.add_dir(run.dir, "model.v"+str(model_version))
        #   trained_model_artifact.add_file(trained_model_artifact)
        #   run.log_artifact(trained_model_artifact)
        #   model_version = model_version + 1


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
      

def build_dataset(batch_size):
    # load data
    X,y = get_train_data()

    dataset = Dataset(X, y)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    return trainloader


def build_network(second_layer_size):
    network = nn.Sequential(  # fully-connected, single hidden layer
        nn.Linear(60, second_layer_size),
        nn.ReLU(),
        nn.Linear(second_layer_size, 30),
        nn.ReLU(),
        nn.Linear(30, 6))

    return network.to(device)
        

def build_optimizer(network, optimizer, starting_lr):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(network.parameters(),
                              lr=starting_lr, momentum=0.9)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(network.parameters(),
                              lr=starting_lr) 

    return optimizer


def build_scheduler(optimizer, milestones, gamma):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    return scheduler

# if plot is true there's code to attempt to log 9 fitting test samples 
def train_epoch(network, trainloader, optimizer, scheduler):
    cumu_loss = 0
    loss_function = nn.MSELoss()

    for i, data in enumerate(trainloader, 0):
        
        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.reshape((targets.shape[0], 6))
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        outputs = network(inputs)
        
        # Compute loss
        loss = loss_function(outputs, targets)
        cumu_loss += loss.item()
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        scheduler.step()

        # log loss 
        wandb.log({"batch loss": loss.item()})

        # if plot:
        #   print('Sweep complete. Now logging plots for the sweep\n')
        #   # *****  log plots  *****
        #   # create subplot of 9 fits 
        #   #m,n = 3,3 # 3x3 subplot (9 total ellipse fits) 
        #   #figure, axis = plt.subplots(m,n, sharex='all', sharey = 'all')
        #   nine_plot = plot_nine(inputs, targets, outputs) # type PIL image

        #   image = wandb.Image(nine_plot)
        #   wandb.log({"Known Ellipse (black) vs. Fit (blue) for 9 testing samples": image}, commit=True)

    return cumu_loss / len(trainloader)


if __name__ == '__main__':

  
  print('before login\n')
  wandb.login()
  print('after login')

  # # set hyperparemeters
  # sweep_id = config_params()

  # # create pathname to save weights to
  # # wandb path
  # wandbpath = r"C:\Users\Nicor\OneDrive\Documents\KolkowitzLab\Ellipse fitting\Learners\wandb"
  # # sweep path
  # pathname = os.path.join(wandbpath, 'sweep-'+sweep_id)

  # # instantiate CheckpointSaver object with pathname
  # checkpoint_saver = CheckpointSaver(dirpath=pathname, decreasing=True, top_n=1)
  # checkpoint_saver = functools.partial(CheckpointSaver, dirpath=pathname, decreasing=True, top_n=1)
  
  # COUNT = NUMBER OF SWEEPS!!
  count = 3
  print('\nStarting '+str(count)+' runs(s)...\n')
  #wandb_train_func = functools.partial(train, checkpoint_saver)
  wandb.agent(sweep_id, train, count=count)

  print('Sweep finished!')

  # print best run configs to see if it matches wandb runs online 
  api = wandb.Api()
  sweep = api.sweep(f"{'nicoranabhat'}/{'ellipse_fitting'}/{sweep_id}")

  best_run = sweep.best_run()
  print(best_run.config)