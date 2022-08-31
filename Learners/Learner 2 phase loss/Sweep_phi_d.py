# *** Adding wandb to github tutorial for experiment tracking *** 

import functools
import shutil
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import wandb
import os
import logging
import loadCSVdata_phi_d
from plot_nine_phi_d import plot_nine
logging.getLogger().setLevel(logging.INFO) # used to print useful checkpoints

NUM_TRAINING_ELLIPSES = 500 # number of ellipses used for training in each run of sweep
NUM_POINTS = 30
CONTRAST = 0.65
CLAMP_EPSILON = -0.0000001

wandb.login()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WANDBPATH = r"C:\Users\Nicor\OneDrive\Documents\KolkowitzLab\ellipse_fitting\Learners\wandb"
#WANDBPATH = r"D:\Nico Ranabhat\Ellipse Fitting\ellipse_fitting\Learners\wandb"

def config_params():

  sweep_config = {
      'method': 'bayes'
      }

  metric = {
      'name': 'loss',
      'goal': 'minimize'   
      }

  sweep_config['metric'] = metric

  parameters_dict = {
      'sweep_epochs': {
          'values': [1]      # change this to >15 later
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
          'values': [128, 256, 512, 1024]
          },
      'starting_lr': {
          'distribution': 'uniform',
          'min': 0.0003,
          'max': 0.006
        },
      'milestones' : {
            'values': [[10]]
          },
      }

  sweep_config['parameters'] = parameters_dict

  parameters_dict.update({
      'gamma': {
          # a flat distribution between 0.01 and 1
          'distribution': 'uniform',
          'min': 0.01,
          'max': 0.7 
        }
      })

  import pprint
  pprint.pprint(sweep_config)

  sweep_id = wandb.sweep(sweep_config, project="ellipse_fitting")

  return sweep_id


# class from https://gist.github.com/amaarora/9b867f1868f319b3f2e6adb6bfe2373e\#file-how-to-save-all-your-trained-model-weights-locally-after-every-epoch-ipynb
class CheckpointSaver:
    def __init__(self, dirpath, sweep_id, decreasing=True, top_n=1):
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.sweep_id = sweep_id
        self.top_n = top_n 
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf
        
    def __call__(self, api, model, metric_val, test_loss, config, epoch, optimizer, scheduler):
        model_path = os.path.join(self.dirpath, 'weights_tensor.pt')
        save = metric_val<=self.best_metric_val if self.decreasing else metric_val>=self.best_metric_val
        if save: 
            logging.info(f"Current metric value {metric_val} better than {self.best_metric_val} \n\
Saving model at {model_path}, & logging model weights to W&B.")
            self.best_metric_val = metric_val
            #torch.save(model.state_dict(), model_path)
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': metric_val, 
                        'test loss': test_loss},
                        model_path)

            print('\nmodel weights saved to '+str(self.sweep_id)+'\n')

            sweep_or_run = ''
            if 'sweep-' in self.dirpath: sweep_or_run = 'sweep'
            else: sweep_or_run = 'run'

            artifact_location_path = f'best-mlp-'+sweep_or_run+'-phi-' +str(self.sweep_id)+'.pt'
            current_lr = optimizer.param_groups[0]['lr']
            self.log_artifact(artifact_location_path, model_path, metric_val, test_loss, epoch, config, current_lr)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)

        if len(self.top_model_paths)>self.top_n:
            self.cleanupLocal()
    
    def log_artifact(self, filename, model_path, metric_val, test_loss, epoch, config, current_lr):
        config_string={k:str(v) for k,v in config.items()}
        config_string['loss'] = metric_val
        config_string['test loss'] = test_loss
        config_string['epoch'] = epoch + 1
        config_string['current_lr'] = current_lr
        config_string['#ellipses (sweep)'] = NUM_TRAINING_ELLIPSES

        artifact = wandb.Artifact(filename, type='model', metadata=config_string)
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)        
    
    def cleanupLocal(self):
        # cleaning up local disc
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]


def get_test_loss(batch_size, network):
    # first two parameters of build_dataset don't rly matter if train=False.
    testloader = build_dataset(int(batch_size), int(NUM_TRAINING_ELLIPSES), train=False)

    # test once manually: 
    loss_function = nn.MSELoss()
    for i, data in enumerate(testloader, 0): # should just be one big batch of all the data (for testing)
    
        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.reshape((targets.shape[0], 6))
        
        # Perform forward pass w/o training gradients 
        with torch.no_grad():
            outputs = network(inputs)
        
        # Compute loss
        total_loss = loss_function(outputs[:,0], targets[:,1])
        avg_loss = total_loss/len(testloader)   # double check exactly what this does (is it just one batch in the loop?)

    return avg_loss


def train(checkpoint_saver, sweep_id, config=None):

    # Initialize a new wandb run
    with wandb.init(config=config, reinit=True) as run: 
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        trainloader = build_dataset(config.batch_size, int(NUM_TRAINING_ELLIPSES), train=True)
        network = build_network(config.second_layer_size, clamp_output=False)
        optimizer = build_optimizer(network, config.optimizer, config.starting_lr)
        scheduler = build_scheduler(optimizer, config.milestones, config.gamma)

        # find particular sweep in wandb 
        api = wandb.Api()
        sweep = api.sweep(f"{'nicoranabhat'}/{'ellipse_fitting'}/{sweep_id}")

        for epoch in range(config.sweep_epochs):

            avg_loss = train_epoch(network, trainloader, optimizer, scheduler)
            avg_test_loss = get_test_loss(config.batch_size, network)
            # after epoch log loss to wandb
            wandb.log({"loss": avg_loss, "test loss": avg_test_loss, "epoch": epoch}, commit=True)
            print('EPOCH: ' + str(epoch+1)+'  LOSS: '+str(avg_loss)+'  TEST LOSS: '+str(avg_test_loss))
            print('optimizer LR: '+str(optimizer.param_groups[0]['lr']))

            # if it's the first or last epoch, wait 3 seconds for wandb to log the loss
            if (epoch==0 or epoch==config.sweep_epochs-1):
                time.sleep(3)

            # update best_loss for next call to checkpoint_saver
            best_loss = sweep.best_run().summary_metrics['loss'] 

            # save weights of current best epoch (will save best model for whole network over training if top_n=1)
            if avg_loss <= best_loss:
                checkpoint_saver(api, network, avg_loss, avg_test_loss, config, epoch, optimizer, scheduler)

        run.finish()


class Dataset(torch.utils.data.Dataset):
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
      

def build_dataset(batch_size, num_ellipses, train):
    # load train(or testing) data
    loader = loadCSVdata_phi_d.loadCSVdata(num_ellipses, NUM_POINTS)
    if train: X,y = loader.get_train_data()
    else: X,y = loader.get_test_data()

    dataset = Dataset(X, y)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=X.shape[0], shuffle=False, num_workers=1)
    
    if train: return trainloader
    else: return testloader


def build_network(second_layer_size, clamp_output):
    # simply define a custom activation function
    def clamp(input):
        '''
        Applies a clamp function to constrian the a2 output based on a fixed contrast: 

        B = [-2/((contrast/2)^2), 2/((contrast/2)^2)]
        '''

        e = CLAMP_EPSILON # amount of error that can be allowed on constraints of parameters
        common_factor = (CONTRAST/2)**2
        fixedAB = 1/common_factor

        output = torch.clone(input) # should fix gradient modification RuntimeError message? 
        output[:] = torch.clamp(output[:].clone(), min=-2*fixedAB-e, max=2*fixedAB+e)

        return output

    # create a class wrapper from PyTorch nn.Module, so
    # the function now can be easily used in models
    class ParameterClamp(nn.Module):
        '''
        Applies the parameter_clamp function

        Shape:
            - Input: (N, *) where * means, any number of additional
            dimensions
            - Output: (N, *), same shape as the input
        '''
        def __init__(self):
            super().__init__() # init the base class

        def forward(self, input):
            '''
            Forward pass of the function.
            '''
            return clamp(input) # simply apply already implemented parameter_clamp

    clamp_activation_function = ParameterClamp()
    if clamp_output:
        network = nn.Sequential(  # fully-connected, single hidden layer
            nn.Linear(60, second_layer_size),
            nn.ReLU(),
            nn.Linear(second_layer_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            clamp_activation_function)
    else: 
        network = nn.Sequential(  # fully-connected, single hidden layer
            nn.Linear(60, second_layer_size),
            nn.ReLU(),
            nn.Linear(second_layer_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),)

    return network.to(device)
        

def build_optimizer(network, optimizer, starting_lr):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(network.parameters(), lr=starting_lr, momentum=0.9)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(network.parameters(), lr=starting_lr) 

    return optimizer


def build_scheduler(optimizer, milestones, gamma):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    return scheduler


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
        loss = loss_function(outputs[:,0], targets[:,1]) 
        cumu_loss += loss.item()
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()

        # log loss 
        wandb.log({"batch loss": loss.item()})

    scheduler.step()
    return cumu_loss / len(trainloader)


def test_and_plot(model_locaiton, sweep_or_run_id, num_training_ellipses, is_sweep):

  api = wandb.Api()

  # not sure if this if-else block runs smooth
  if is_sweep:
    # print best run configs to see if it matches wandb runs online 
    sweep = api.sweep(f"{'nicoranabhat'}/{'ellipse_fitting'}/{sweep_or_run_id}")
    best_run = sweep.best_run()
    print(best_run.config) # sanity check

  # Initialize a new wandb (validation) run
  with wandb.init(project='ellipse_fitting', reinit=True) as run: 

        artifact = run.use_artifact(model_locaiton, type='model') # should only be 1 artifact if cleanup() works 
        artifact_dir = artifact.download()

        #now need to get config
        config = artifact.metadata

        testloader = build_dataset(int(config['batch_size']), int(num_training_ellipses), train=False)
        # previous network build: 
        network = build_network(int(config['second_layer_size']), clamp_output=True)
        # new network built:
        weights_path = os.path.join(artifact_dir, 'weights_tensor.pt')
        network.load_state_dict(torch.load(weights_path)['model_state_dict'])
        # don't need to load scheduler and optimizer because we only run through one ~non-training~ epoch for validation

        # test once manually: 
        loss_function = nn.MSELoss()
        for i, data in enumerate(testloader, 0): # should just be one big batch of all the data (for testing)
        
            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.reshape((targets.shape[0], 6))
            
            # Perform forward pass w/o training gradients 
            with torch.no_grad():
                outputs = network(inputs)
            
            # Compute loss
            target_B = targets[:,1]
            total_loss = loss_function(outputs[:,0], target_B)
            avg_loss = total_loss/len(testloader)   # double check exactly what this does (is it just one batch in the loop?)

            # after epoch, log loss to wandb
            wandb.log({"test set average loss": avg_loss})
            print('\ntest set average loss: ' + str(avg_loss))
            train_loss = str(config['loss'])
            print('train set average loss: ' + train_loss)

        # create subplot of 9 fits 
        nine_plot = plot_nine(inputs, targets, outputs, avg_loss, train_loss, CLAMP_EPSILON) # type PIL image
        image = wandb.Image(nine_plot)

        # log the plots 
        avg_loss_float = avg_loss.detach().numpy()
        if is_sweep: sweep_or_run = 'sweep'
        else: sweep_or_run = 'run'
        image_artifact = wandb.Artifact(f''+sweep_or_run+'-'+str(sweep_or_run_id)+'-'+str(num_training_ellipses)+'ellipses'+\
        '-avgtestloss-'+str(avg_loss_float), type='plot')
        image_artifact.add(obj=image, name='Fit (blue) vs. Truth (black) for 9 testing samples')
        wandb.run.log_artifact(image_artifact)

        run.finish()


def main():

    sweep_id = config_params()
    
    # sweep path
    pathname = os.path.join(WANDBPATH, 'sweep-'+sweep_id)

    # instantiate CheckpointSaver object with sweep path
    checkpoint_saver = CheckpointSaver(dirpath=pathname, sweep_id=sweep_id, decreasing=True, top_n=1)
    
    # COUNT = NUMBER OF RUNS!!
    count = 1
    print('\nStarting '+str(count)+' runs(s)...\n')

    wandb_train_func = functools.partial(train, checkpoint_saver, sweep_id)

    wandb.agent(sweep_id, function=wandb_train_func, count=count)

    # delete all artifacts that aren't top 5
    time.sleep(3)
    api = wandb.Api()
    artifact_location_path = f'best-mlp-sweep-phi-' +str(sweep_id)+'.pt'
    artifact_type, artifact_name = 'model', artifact_location_path # fill in the desired type + name
    for version in api.artifact_versions(artifact_type, artifact_name):
        if len(version.aliases) == 1: #has aliase 'latest'
            best_version_num = int(version.version[1])
        if int(version.version[1]) < (best_version_num - 4):
            version.delete()
    print('\nSweep finished!\n')
    print('Begining validation...')
    wandb.finish()

    # ________ sweep is complete __________ # 

    # test best model, plot results from best model (sanity check)...
    # save plot as artifact
    model_location = 'nicoranabhat/ellipse_fitting/best-mlp-sweep-phi-' + sweep_id + '.pt:latest'
    test_and_plot(model_location, sweep_id, num_training_ellipses=NUM_TRAINING_ELLIPSES, is_sweep=True)

    # delete any files saved to local machine
    if os.path.isdir(pathname): shutil.rmtree(pathname) 

    print('\nALL PROCESSES COMPLETE! (for sweep '+sweep_id+')\n')

        
if __name__ == "__main__":
    main()