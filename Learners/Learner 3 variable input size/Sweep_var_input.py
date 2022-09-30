# *** Adding wandb to github tutorial for experiment tracking *** 

import functools
import math
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
import loadCSVdata_var_input
from plot_nine_var_input import plot_nine
from matplotlib.patches import Ellipse
from ellipse import LsqEllipse
logging.getLogger().setLevel(logging.INFO) # used to print useful checkpoints

NUM_TRAINING_ELLIPSES = 500 # number of ellipses used for training in each run of sweep
MAX_SHOTS = 30
CONTRAST = 0.65
CLAMP_EPSILON = -0.0000001

wandb.login()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WANDBPATH = r"C:\Users\Nicor\OneDrive\Documents\KolkowitzLab\ellipse_fitting\Learners\wandb"
#WANDBPATH = r"D:\Nico Ranabhat\Ellipse Fitting\el\Learners\wandb"

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
          'max': 60,
        },
      'optimizer': {
          'values': ['adam', 'sgd']
          },
      'second_layer_size': {
          'values': [1000, 1500, 2000]
          },
      'starting_lr': {
          'distribution': 'uniform',
          'min': 0.00001,
          'max': 0.005
        },
      'milestones' : {
            'values':  [[10]]
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

            artifact_location_path = f'best-mlp-'+sweep_or_run+'-phase-' +str(self.sweep_id)+'.pt'
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
    # **first two parameters of build_dataset don't rly matter if train=False.**
    testloader = build_dataset(int(batch_size), int(NUM_TRAINING_ELLIPSES), train=False)

    # test once manually: 
    loss_function = nn.MSELoss()
    for i, data in enumerate(testloader, 0): # should just be one big batch of all the data (for testing)
    
        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.reshape((targets.shape[0], 1))
        
        # Perform forward pass w/o training gradients 
        with torch.no_grad():
            outputs = network(inputs)
        
        # Compute loss
        total_loss = loss_function(outputs[:,0], targets[:,0])
        avg_loss = total_loss

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
    loader = loadCSVdata_var_input.loadCSVdata(num_ellipses, MAX_SHOTS)
    if train:
        X,y = loader.get_train_data()
        phi = loader.get_train_phi_d()

    else:
        X,y = loader.get_test_data()
        phi = loader.get_test_phi_d()

    dataset = Dataset(X, phi)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=X.shape[0], shuffle=False, num_workers=1)
    
    if train: return trainloader
    else: return testloader


def build_network(second_layer_size, clamp_output):
    # simply define a custom activation function
    def clamp(input):
        '''
        Applies a clamp function to constrian the phi_d output: 

        phi_d in [0, pi/2]
        '''

        e = CLAMP_EPSILON # amount of error that can be allowed on constraints of parameters
        #common_factor = (CONTRAST/2)**2
        #fixedAB = 1/common_factor

        output = torch.clone(input) # should fix gradient modification RuntimeError message? 
        output[:] = torch.clamp(output[:].clone(), min=0-e, max=(math.pi/2+e))

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
            nn.Linear(MAX_SHOTS, second_layer_size),
            nn.ReLU(),
            nn.Linear(second_layer_size, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            clamp_activation_function)
    else: 
        network = nn.Sequential(  # fully-connected, single hidden layer
            nn.Linear(MAX_SHOTS, second_layer_size),
            nn.ReLU(),
            nn.Linear(second_layer_size, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 32),
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
        targets = targets.reshape((targets.shape[0], 1))            # only one phase value for output
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        outputs = network(inputs)
        
        # Compute loss
        loss = loss_function(outputs[:,0], targets[:,0])
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
            targets = targets.reshape((targets.shape[0], 1))
            
            # Perform forward pass w/o training gradients 
            with torch.no_grad():
                outputs = network(inputs)
            
            # Compute neural net loss
            total_loss = loss_function(outputs[:,0], targets[:,0])
            avg_loss = total_loss

            # after epoch, log loss to wandb
            wandb.log({"test set average loss": avg_loss})
            print('\ntest set average loss: ' + str(avg_loss))
            train_loss = str(config['loss'])
            print('train set average loss: ' + train_loss)

            # compute LS loss
            LS_test_loss, Phi_LS = get_LS_test_loss(inputs, targets)

        # create subplot of 9 fits 
        nine_plot = plot_nine(inputs, targets, outputs, avg_loss, config['loss'], LS_test_loss, Phi_LS, CLAMP_EPSILON) # type PIL image
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

def get_LS_test_loss(inputs, targets):
    inputs = inputs.detach().numpy()
    targets = targets.detach().numpy()

    def make_test_ellipse(X,i):
        """Generate Elliptical

        Returns
        -------
        data:  list:list:float
            list of two lists containing the x and y data of the ellipse.
            of the form [[x1, x2, ..., xi],[y1, y2, ..., yi]]
        """
        x_y_arrays = np.split(X, 2, axis=1)
        ellipse_x = x_y_arrays[0]
        ellipse_y = x_y_arrays[1]
        x_coords = np.asarray(ellipse_x[i,:], dtype=float)
        y_coords = np.asarray(ellipse_y[i,:], dtype=float)
        for i in range(MAX_SHOTS):
            if x_coords[i] == 0: # contrast will never be exactly 1, so points will never be exactly on [0,0].
                end_index = i
                break
        x_coords = np.delete(x_coords, slice(end_index,MAX_SHOTS))
        y_coords = np.delete(y_coords, slice(end_index,MAX_SHOTS))
        return [x_coords, y_coords]
    
    Phi_LS = np.empty(shape=len(targets))
    for i in range(len(targets)):
        X1, X2 = make_test_ellipse(inputs,i)
        X_single_ellipse = np.array(list(zip(X1, X2)))
        fitter = LsqEllipse()
        reg = fitter.fit(X_single_ellipse)
        center, width, height, phi = reg.as_parameters()
        LS_output = fitter.coefficients

        # finding target (a1-a6):
        loader = loadCSVdata_var_input.loadCSVdata(NUM_TRAINING_ELLIPSES, MAX_SHOTS)
        X_target, y_target = loader.get_test_data()

        # finding LS (a1-a6) loss:
        a = LS_output[0];       A = float(y_target[i,0])
        b = LS_output[1];       B = float(y_target[i,1])
        c = LS_output[2];       C = float(y_target[i,2])
        # d = LS_output[3];       D = float(targets[i,3])
        # e = LS_output[4];       E = float(targets[i,4])
        # f = LS_output[5];       F = float(targets[i,5])

        acos_arg_targets = -B/(2*math.sqrt(A*C))
        acos_arg_model = -b/(2*math.sqrt(a*c))
        if np.sign(acos_arg_targets) != np.sign(acos_arg_model):
            acos_arg_model = -acos_arg_model
        #target_phi_d = 0.5*acos(acos_arg_targets)
        model_phi_d = 0.5*math.acos(acos_arg_model)
        #Phi_targets[i] = target_phi_d
        Phi_LS[i] = model_phi_d

    Phi_LS = np.reshape(Phi_LS, (Phi_LS.shape[0], 1))
    LS_test_loss = np.linalg.norm(targets-Phi_LS)**2 / len(targets)
    return LS_test_loss, Phi_LS


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
    artifact_location_path = f'best-mlp-sweep-phase-' +str(sweep_id)+'.pt'
    artifact_type, artifact_name = 'model', artifact_location_path # fill in the desired type + name
    for version in api.artifact_versions(artifact_type, artifact_name):
        if len(version.aliases) == 1: #has aliase 'latest'
            best_version_num = int(version.version[1])
        if int(version.version[1]) < (best_version_num - 4):
            version.delete()
    print('\nSweep finished!\n')
    print('Begining validation...\n')
    wandb.finish()

    # ________ sweep is complete __________ # 

    # test best model, plot results from best model (sanity check)...
    # save plot as artifact
    model_location = 'nicoranabhat/ellipse_fitting/best-mlp-sweep-phase-' + sweep_id + '.pt:latest'
    test_and_plot(model_location, sweep_id, num_training_ellipses=NUM_TRAINING_ELLIPSES, is_sweep=True)

    # delete any files saved to local machine
    if os.path.isdir(pathname): shutil.rmtree(pathname) 

    print('\nALL PROCESSES COMPLETE! (for sweep '+sweep_id+')\n')

        
if __name__ == "__main__":
    main()