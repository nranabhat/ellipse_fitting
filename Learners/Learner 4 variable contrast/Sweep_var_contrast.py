""" Weights and Biases sweep for ANN
@author: nranabhat  """

#imports 
from audioop import avg
from distutils.log import error
import functools
import math
import shutil
import time
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau, SequentialLR, CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
from torch import avg_pool1d, nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import wandb
import os
import logging
import loadCSVdata_var_contrast
from plot_nine_var_contrast import plot_nine, plot_errors
from matplotlib.patches import Ellipse
from ellipse import LsqEllipse
import ellipse_fitting_api
logging.getLogger().setLevel(logging.INFO) # used to print useful checkpoints

#set constants
BAYESIAN_SWEEP = True # bayesian or grid?
PLOT_MLE = True
NUM_TRAINING_ELLIPSES = 10000 # number of ellipses used for training in each run of sweep.   
                            # Can change but make sure the dataset exists!
MAX_SHOTS = 500
MAX_CONTRAST = 0.98
MIN_CONTRAST = 0.1
CLAMP_EPSILON = -0.0000001 
DROPOUT_PROBABILITY = 0 # probability for a neuron to be zeroed. e.g.) p=0: no neurons are dropped. range:[0,1]
FULL_PHI_RANGE = True # If false, range will be [0,0.15] and [pi/2-0.15, pi/2]. 
                      # Can change but make sure the dataset exists!
LAB_COMP = False # change to False if running on Nico's machine. Specifies local file paths 
VARIABLE_CONTRAST = False # constant vs. variable contrast dataset
SCHEDULER_TYPE = 'LRPlateau' # can be 'LRPlateau' or 'CosineAnnealing'

if VARIABLE_CONTRAST: var_cons = 'Var'
else: var_cons = 'Constant'
if FULL_PHI_RANGE: all_phi = 'allPhi'
else: all_phi = 'fewPhi'
LOG_NEW_ARTIFACT_TO = '-1hl-1000n-'+all_phi+'-'+var_cons+'Contrast-'+SCHEDULER_TYPE+'-' 
# 1hl-1000n... => '1 hidden layer, 1000 neurons, phi range, contrast range, LR'

# Constants for calculating loss
phase_range = math.pi/2
c_x_range = MAX_CONTRAST-MIN_CONTRAST
c_y_range = MAX_CONTRAST-MIN_CONTRAST
K_PHI = (2/phase_range)**2
K_CX = (2/c_x_range)**2
K_CY =  (2/c_y_range)**2

wandb.login()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if LAB_COMP:
    WANDBPATH = r"D:\Nico Ranabhat\Ellipse Fitting\ellipse_fitting\Learners\wandb"
else: 
    WANDBPATH = r"C:\Users\Nicor\OneDrive\Documents\KolkowitzLab\ellipse_fitting\Learners\wandb"


def config_params():
  # configureation of hyperparameters for bayesian sweep

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
          'values': [20]      # change this to >15 later
          },
      'batch_size': {
          'values': [100, 2500, 5000]
          },
      'optimizer': {
          'values': ['adam', 'sgd']
          },
      'second_layer_size': {
          'values': [512, 1024, 2048]
          },
      'starting_lr': {
          'distribution': 'uniform',
          'min': 0.0001,
          'max': 0.5
        },
      'milestones' : {
            'values':  [[0]]
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


def config_params_grid():
  # configureation of hyperparameters for grid sweep

  second_layer_size = []
  for i in range(0,1000,20):
    layer_size = i+1
    second_layer_size.append(layer_size)

  sweep_config = {
      'method': 'grid'
      }

  metric = {
      'name': 'loss',
      'goal': 'minimize'   
      }

  sweep_config['metric'] = metric

  parameters_dict = {
      'sweep_epochs': {
          'values': [20]     
          },
      'batch_size': {
          'values': [40]
        },
      'optimizer': {
          'values': ['sgd']
          },
      'second_layer_size': {
          'values': second_layer_size
          },
      'starting_lr': {
          'values': [0.004672124070515444]
        },
      'milestones' : {
            'values':  [[0]]
          },
      'gamma': {
            'values':  [0.1]
        }
      }
      
  sweep_config['parameters'] = parameters_dict

  import pprint
  pprint.pprint(sweep_config)

  sweep_id = wandb.sweep(sweep_config, project="ellipse_fitting")

  return sweep_id


class CheckpointSaver:
    """     CheckpointSaver class class from https://gist.github.com/amaarora/9b867f1868f319b3f2e6adb6bfe2373e\#
    file-how-to-save-all-your-trained-model-weights-locally-after-every-epoch-ipynb

    ----*** only used in sweep...py! Not used in load_and_test...py ***----#  
    
    Saves model weights, architecture, and status to wandb 

    """

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
        
    def __call__(self, api, model, metric_val, test_loss, phase_loss, config, epoch, optimizer, scheduler):
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
                        'test loss': test_loss,
                        'phase loss': phase_loss},
                        model_path)

            print('\nmodel weights saved to '+str(self.sweep_id)+'\n')

            sweep_or_run = ''
            if 'sweep-' in self.dirpath: sweep_or_run = 'sweep'
            else: sweep_or_run = 'run'

            artifact_location_path = f'mlp-'+sweep_or_run+'-'+str(self.sweep_id)+LOG_NEW_ARTIFACT_TO+'.pt'
            current_lr = optimizer.param_groups[0]['lr']
            self.log_artifact(artifact_location_path, model_path, metric_val, test_loss, phase_loss, epoch, config, current_lr)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)

        if len(self.top_model_paths)>self.top_n:
            self.cleanupLocal(api, artifact_location_path)
    
    def log_artifact(self, filename, model_path, metric_val, test_loss, phase_loss, epoch, config, current_lr):
        config_string={k:str(v) for k,v in config.items()}
        config_string['loss'] = metric_val
        config_string['test loss'] = test_loss
        config_string['phase loss'] = phase_loss
        config_string['epoch'] = epoch + 1
        config_string['current_lr'] = current_lr
        config_string['#ellipses (sweep)'] = NUM_TRAINING_ELLIPSES

        artifact = wandb.Artifact(filename, type='model', metadata=config_string)
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)        
    
    def cleanupLocal(self, api, artifact_name):
        # cleaning up local disc
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]

        # cleaning up wandb model artifacts
        for version in api.artifact_versions('model', artifact_name):
        # Clean up all versions that don't have an alias such as 'latest'.
            # NOTE: You can put whatever deletion logic you want here.
            if len(version.aliases) == 0:
                version.delete()


def get_test_loss(batch_size, network):
    # ****   first two parameters of build_dataset don't rly matter if train=False.   ****
    testloader = build_dataset(int(batch_size), int(NUM_TRAINING_ELLIPSES), train=False)

    # test once manually: 
        # nn.MSELoss is the mean squared error (squared L2 norm) between each element in the input xx and target yy
    MSE_loss = nn.MSELoss(reduction='mean')
    for i, data in enumerate(testloader, 0): # should just be one big batch of all the data (for testing)
    
        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.reshape((targets.shape[0], 3))
        
        # Perform forward pass w/o training gradients 
        # **** HERE WE USE THE SAME NETWORK WE USED FOR TRAINING (NO CLAMPING) 
        # **** BECAUSE OF UNKNOWN BEHAVIOR OF CREATING NEW WANDB RUN FOR TESTING
        with torch.no_grad():
            outputs = network(inputs)
        
        # Compute loss
        phi_loss = K_PHI*MSE_loss(outputs[:,0], targets[:,0])
        cx_loss = K_CX*MSE_loss(outputs[:,1], targets[:,1])
        cy_loss = K_CY*MSE_loss(outputs[:,2], targets[:,2])
        total_loss = phi_loss + cx_loss + cy_loss

    return total_loss, phi_loss


def train(checkpoint_saver, sweep_id, config=None):

    # Initialize a new wandb run
    with wandb.init(config=config, reinit=True) as run: 
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        trainloader = build_dataset(config.batch_size, int(NUM_TRAINING_ELLIPSES), train=True)
        network = build_network(config.second_layer_size, train=True)
        optimizer = build_optimizer(network, config.optimizer, config.starting_lr)
        scheduler = build_scheduler(optimizer, config.milestones, config.gamma, scheduler_type=SCHEDULER_TYPE)

        # find particular sweep in wandb 
        api = wandb.Api()
        sweep = api.sweep(f"{'nicoranabhat'}/{'ellipse_fitting'}/{sweep_id}")

        for epoch in range(config.sweep_epochs):
            batch_size = config.batch_size
            avg_loss, avg_tot_test_loss, avg_phase_test_loss = train_epoch(network, trainloader, optimizer, scheduler, batch_size)

            # after epoch log loss to wandb
            wandb.log({"loss": avg_loss, "test loss": avg_tot_test_loss, "test phase loss": avg_phase_test_loss, 
                "epoch": epoch, "learning rate": optimizer.param_groups[0]['lr']}, commit=True)
            print('EPOCH: ' + str(epoch+1)+'  LOSS: '+str(avg_loss)+'  TEST LOSS: '+str(avg_tot_test_loss)+
                '   TEST PHASE LOSS: '+str(avg_phase_test_loss))
            print('optimizer LR: '+str(optimizer.param_groups[0]['lr']))

            # if it's the first or last epoch, wait 3 seconds for wandb to log the loss
            if (epoch==0 or epoch==config.sweep_epochs-1):
                print('sleeping for 3s...')
                time.sleep(3)

            # update best_loss for next call to checkpoint_saver
            best_loss = sweep.best_run().summary_metrics['loss'] 
            
            # save weights of current best epoch (will save best model for whole network over training if top_n=1)
            if avg_loss <= best_loss:
                checkpoint_saver(api, network, avg_loss, avg_tot_test_loss, avg_phase_test_loss, 
                config, epoch, optimizer, scheduler)

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
    loader = loadCSVdata_var_contrast.loadCSVdata(num_ellipses, MAX_SHOTS, FULL_PHI_RANGE, LAB_COMP, VARIABLE_CONTRAST)
    if train:
        X,y = loader.get_train_data()

    else:
        X,y = loader.get_test_data()

    dataset = Dataset(X, y)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=X.shape[0], shuffle=False, num_workers=1)
    
    if train: return trainloader
    else: return testloader


def build_network(second_layer_size, train):
    # simply define a custom activation function
    def clamp(input):
        '''
        Applies a clamp function to constrian the phi_d, c_x, and c_y output: 

        phi_d in [0, pi/2]
        c_x, c_y in [0.1/2, 0.98/2]
        '''
        e = CLAMP_EPSILON # amount of error that can be allowed on constraints of parameters

        output = torch.clone(input) # should fix gradient modification RuntimeError message? 
        output[:,0] = torch.clamp(output[:,0].clone(), min=0-e, max=(math.pi/2+e))
        output[:,1] = torch.clamp(output[:,1].clone(), min=MIN_CONTRAST/2-e, max=MAX_CONTRAST/2+e)
        output[:,2] = torch.clamp(output[:,2].clone(), min=MIN_CONTRAST/2-e, max=MAX_CONTRAST/2+e)

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
    p = DROPOUT_PROBABILITY
    if train:
        network = nn.Sequential(  # fully-connected, single hidden layer
            nn.Linear(MAX_SHOTS*2, second_layer_size),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(second_layer_size, second_layer_size*2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(second_layer_size*2, 128),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, 3))

    # clamp the output to physical ranges for testing
    else: 
        network = nn.Sequential(  # fully-connected, single hidden layer
            nn.Linear(MAX_SHOTS*2, second_layer_size),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(second_layer_size, second_layer_size*2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(second_layer_size*2, 128),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, 3),
            clamp_activation_function)

    return network.to(device)
        

def build_optimizer(network, optimizer, starting_lr):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(network.parameters(), lr=starting_lr, momentum=0.9)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(network.parameters(), lr=starting_lr) 

    return optimizer


def build_scheduler(optimizer, milestones, gamma, scheduler_type):
    if scheduler_type == 'LRPlateau':
        scheduler = ReduceLROnPlateau(optimizer, factor=gamma, threshold=0.0001, patience=5, verbose=True)
    elif scheduler_type == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optimizer, T_max=4, verbose=True)
    return scheduler


def train_epoch(network, trainloader, optimizer, scheduler, batch_size):
    cumu_loss = 0
    MSE_loss = nn.MSELoss(reduction='mean')

    for i, data in enumerate(trainloader, 0):
        
        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.reshape((targets.shape[0], 3))          
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        outputs = network(inputs)
        
        # Compute loss
        phi_loss = K_PHI*MSE_loss(outputs[:,0], targets[:,0])
        cx_loss = K_CX*MSE_loss(outputs[:,1], targets[:,1])
        cy_loss = K_CY*MSE_loss(outputs[:,2], targets[:,2])
        loss = phi_loss + cx_loss + cy_loss
        cumu_loss += loss.item()
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()

        # log loss 
        wandb.log({"batch loss": loss.item()})
    
    avg_tot_test_loss, avg_phase_test_loss = get_test_loss(batch_size, network)

    if str(scheduler.__class__) == "<class 'torch.optim.lr_scheduler.SequentialLR'>":
        scheduler.step()
    elif str(scheduler.__class__) == "<class 'torch.optim.lr_scheduler.CosineAnnealingLR'>":
        scheduler.step()
    
    return cumu_loss / len(trainloader), avg_tot_test_loss, avg_phase_test_loss 


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
        network = build_network(int(config['second_layer_size']), train=False)
        # new network built:
        weights_path = os.path.join(artifact_dir, 'weights_tensor.pt')
        network.load_state_dict(torch.load(weights_path)['model_state_dict'])
        # don't need to load scheduler and optimizer because we only run through one ~non-training~ epoch for validation

        # test once manually: 
        MSE_loss = nn.MSELoss(reduction='mean')
        for i, data in enumerate(testloader, 0): # should just be one big batch of all the data (for testing)
        
            # Get and prepare inputs (X array containing 500 x-coords and 500 y-coords)
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.reshape((targets.shape[0], 3))
            
            # Perform forward pass w/o training gradients 
            with torch.no_grad():
                outputs = network(inputs)
            
            # Compute neural net loss
            phi_loss = K_PHI*MSE_loss(outputs[:,0], targets[:,0])
            cx_loss = K_CX*MSE_loss(outputs[:,1], targets[:,1])
            cy_loss = K_CY*MSE_loss(outputs[:,2], targets[:,2])
            avg_loss = phi_loss + cx_loss + cy_loss
            phase_loss = phi_loss

            # after epoch, log loss to wandb
            wandb.log({"test set average loss": avg_loss, "test phase loss":phase_loss})
            print('\ntest set average loss: ' + str(avg_loss))
            train_loss = str(config['loss'])
            print('train set average loss: ' + train_loss)
            print('\ntest set average phase loss: '+str(phase_loss))

            # compute LS and MLE loss
            LS_test_loss, Phi_LS = get_LS_test_loss(inputs, targets[:,0])
            if PLOT_MLE:
                MLE_test_loss, Phi_MLE = get_MLE_testloss_and_phi(inputs, targets)
            else: 
                MLE_test_loss = None
                Phi_MLE = None

        # create subplot of 9 fits 
        nine_plot = plot_nine(PLOT_MLE, inputs, targets, 
                              outputs, Phi_LS, 
                              avg_loss, phase_loss, config['loss'], LS_test_loss, MLE_test_loss,
                              CLAMP_EPSILON) 
                              # returns type PIL image
        
        image = wandb.Image(nine_plot)

        # create plot of errors
        error_plot = plot_errors(targets, outputs, Phi_LS, Phi_MLE, PLOT_MLE)
        image_errors = wandb.Image(error_plot)

        # log the plots 
        avg_loss_float = avg_loss.detach().numpy()
        if is_sweep: sweep_or_run = 'sweep'
        else: sweep_or_run = 'run'
        # log 9 plot
        image_artifact = wandb.Artifact(f''+sweep_or_run+'-'+str(sweep_or_run_id)+'-'+str(num_training_ellipses)+'ellipses'+\
                                        '-avgtestloss-'+str(avg_loss_float), type='plot')
        image_artifact.add(obj=image, name='Fit (blue) vs. Truth (black) for 9 testing samples')
        image_artifact.add(obj=image_errors, name='Errors')
        wandb.run.log_artifact(image_artifact)
        # log error plot

        run.finish()


def get_LS_test_loss(inputs, targets):
    """     
    Parameters
    ----------
    inputs : pytorch tensor
        [100, 1000] tensor of coordinates of experimental data points. First 500: x-coords. Next 500: y-coords
    targets : tensor
        [100] target phi values for 100 test ellipses 

    Returns
    -------
    LS_test_loss: float
        phi loss (cost)
    Phi_LS: numpy array
        [100] Least-squares phi estimate
     """
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
    
    def get_LS_testloss_and_phi():
        Phi_LS = np.empty(shape=len(targets))
        for i in range(len(targets)):
            X1, X2 = make_test_ellipse(inputs,i)
            X_single_ellipse = np.array(list(zip(X1, X2)))
            fitter = LsqEllipse()
            reg = fitter.fit(X_single_ellipse)
            center, width, height, phi = reg.as_parameters()
            LS_output = fitter.coefficients

            # finding target (a1-a6):
            loader = loadCSVdata_var_contrast.loadCSVdata(NUM_TRAINING_ELLIPSES, MAX_SHOTS, FULL_PHI_RANGE, LAB_COMP, VARIABLE_CONTRAST)
            y_target = loader.get_test_labels()

            # finding LS (a1-a6) loss:
            a = LS_output[0];       A = float(y_target[i,0])
            b = LS_output[1];       B = float(y_target[i,1])
            c = LS_output[2];       C = float(y_target[i,2])

            acos_arg_targets = -B/(2*math.sqrt(A*C))
            acos_arg_model = -b/(2*math.sqrt(a*c))
            if np.sign(acos_arg_targets) != np.sign(acos_arg_model):
                acos_arg_model = -acos_arg_model
            #target_phi_d = 0.5*acos(acos_arg_targets)
            model_phi_d = 0.5*math.acos(acos_arg_model)
            #Phi_targets[i] = target_phi_d
            Phi_LS[i] = model_phi_d

        Phi_LS = np.reshape(Phi_LS, (Phi_LS.shape[0],))
        LS_test_loss = K_PHI*np.linalg.norm(targets-Phi_LS)**2 / len(targets)
        return LS_test_loss, Phi_LS

    LS_test_loss, Phi_LS = get_LS_testloss_and_phi()

    return LS_test_loss, Phi_LS


def get_MLE_testloss_and_phi(X, targets):
    """ Obtain test loss and phi estimates for MLE algo.
        Note: MLE requires contrast inputs. 
    Parameters
    ----------
    X : pytorch tensor
        [100, 1000] tensor of coordinates of experimental data points. First 500: x-coords. Next 500: y-coords
    targets : tensor
        [100,3] target phi, C_x, C_y values for 100 test ellipses 

    Returns
    -------
    MLE_test_loss : float
        phi loss (cost)
    Phi_MLE : numpy array
        [100] MLE phi estimate
     """

    X = X.detach().numpy()
    targets = targets.detach().numpy()
    #contrast = 0.65
    num_atoms = 1000
    Phi_MLE = np.zeros(targets.shape[0])
    num_shots = 0
    # loop through 100 ellipses in testing set
    for i in range(targets.shape[0]):
        # loop through 500 x-points in X
        for j in range(MAX_SHOTS):
            # find the number of non-zero entries (how many points the ellipse has)
            if X[i,j] == 0: 
                num_shots = j+1
                break
        points_x = X[i, 0:num_shots-1]
        points_y = X[i, MAX_SHOTS:MAX_SHOTS+num_shots-1]
        points = [points_x, points_y] # need the transpose of this
        points = [list(k) for k in zip(*points)]
        
        contrast = 2 * targets[i,1]
        phi_estimate = ellipse_fitting_api.main(points, contrast, num_atoms)
        Phi_MLE[i] = phi_estimate
    
    Phi_MLE = np.reshape(Phi_MLE, (Phi_MLE.shape[0],))
    MLE_test_loss = K_PHI*np.linalg.norm(targets[:,0]-Phi_MLE)**2 / len(targets[:,0])

    return MLE_test_loss, Phi_MLE


def main():

    if BAYESIAN_SWEEP:
        sweep_id = config_params()
    else:
        sweep_id = config_params_grid()
        
    # sweep path
    pathname = os.path.join(WANDBPATH, 'sweep-'+sweep_id)

    # instantiate CheckpointSaver object with sweep path
    checkpoint_saver = CheckpointSaver(dirpath=pathname, sweep_id=sweep_id, decreasing=True, top_n=1)
    
    if BAYESIAN_SWEEP:
        # COUNT = NUMBER OF RUNS!!
        count = 25
        print('\nTraining '+str(count)+' models...\n')

    wandb_train_func = functools.partial(train, checkpoint_saver, sweep_id)

    if BAYESIAN_SWEEP:
        wandb.agent(sweep_id, function=wandb_train_func, count=count)
    else:  wandb.agent(sweep_id, function=wandb_train_func)
    
    model_location = f'mlp-sweep-'+str(sweep_id)+LOG_NEW_ARTIFACT_TO+'.pt:latest'
    test_and_plot(model_location, sweep_id, num_training_ellipses=NUM_TRAINING_ELLIPSES, is_sweep=True)

    # delete any files saved to local machine
    if os.path.isdir(pathname): shutil.rmtree(pathname) 

    print('\nALL PROCESSES COMPLETE! (for sweep '+sweep_id+')\n')

        
if __name__ == "__main__":
    main()