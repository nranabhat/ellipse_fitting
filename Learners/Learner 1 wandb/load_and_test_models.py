# loading trained models and plotting their fit estimations

from cmath import inf
import logging
import stat
import time
import torch
import wandb
import os
from ast import literal_eval
import numpy as np
from learner1_wandb_Sweep1 import CheckpointSaver,Dataset,build_dataset,build_network,build_optimizer,build_scheduler,train_epoch,test_and_plot

RUN_ID = 'vj2p751m'
VERSION_NUM = '295'
NUM_TRAINING_ELLIPSES = '10000'
NAME_OF_ARTIFACT_TO_USE = 'nicoranabhat/ellipse_fitting/best-run-'+RUN_ID+'.pt:v'+str(VERSION_NUM)
LOG_NEW_ARTIFACT_TO = f'test-run-'+str(RUN_ID)+'-'+NUM_TRAINING_ELLIPSES+'-trainingellipses.pt'

wandbpath = r"C:\Users\Nicor\OneDrive\Documents\KolkowitzLab\Ellipse fitting\Learners\wandb"   
pathname = os.path.join(wandbpath, 'best-'+NUM_TRAINING_ELLIPSES+'-trainingellipses-run-for-sweep-'+RUN_ID)
MODEL_PATH = os.path.join(pathname, 'weights_tensor.pt')

EPOCHS = 3

wandb.login()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__': 
    with wandb.init(project='ellipse_fitting') as run:
        
        def log_artifact(filename, model_path, best_loss, config):
            config_string={k:str(v) for k,v in config.items()}
            config_string['loss'] = best_loss

            artifact = wandb.Artifact(filename, type='model', metadata=config_string)
            artifact.add_file(model_path)
            wandb.run.log_artifact(artifact)   

        artifact = run.use_artifact(NAME_OF_ARTIFACT_TO_USE, type='model')
        artifact_dir = artifact.download()
        weights_path = os.path.join(artifact_dir, 'weights_tensor.pt')
        #config = artifact.metadata
        config = {'loss': 0.5, 
        'gamma': '0.5454507300590375', 
        'epochs': '35+', 
        'optimizer': 'sgd', 
        'batch_size': '15',
        'milestones': '[10]', 
        'starting_lr': '0.01402435040728651', 
        'second_layer_size': '512'}
        print('config:\n'+str(config))

        # instantiate CheckpointSaver object with run path
        #checkpoint_saver = CheckpointSaver(dirpath=pathname, sweep_id=RUN_ID, decreasing=True, top_n=1)
        trainloader = build_dataset(int(config['batch_size']), True)
        network = build_network(int(config['second_layer_size']))
        network.load_state_dict(torch.load(weights_path))
        optimizer = build_optimizer(network, config['optimizer'], float(config['starting_lr']))
        scheduler = build_scheduler(optimizer, np.array(literal_eval(config['milestones'])), float(config['gamma']))

        best_loss = float(config['loss'])

        for epoch in range(EPOCHS):
            print('starting epoch '+str(epoch+1))
            avg_loss = train_epoch(network, trainloader, optimizer, scheduler)

            # after epoch log loss to wandb
            wandb.log({"avg loss over epoch": avg_loss, "epoch": epoch}, commit=True)
            print('average loss for this epoch: '+str(avg_loss))

            # if it's the first or last epoch, wait 3 seconds for wandb to log the loss
            if (epoch==0 or epoch==EPOCHS-1):
                time.sleep(3)

            # save weights of current best epoch (will save best model for whole network over training if top_n=1)
            if avg_loss <= best_loss:
                #checkpoint_saver(network, avg_loss, epoch, optimizer, config)

                # save network on local drive and as artifact on wandb
                logging.info(f"Current metric value {avg_loss} better than {best_loss}, "+ \
                "saving model at "+MODEL_PATH+' , & logging model weights to W&B.')
                best_loss = avg_loss
                
                if not os.path.exists(pathname): os.makedirs(pathname)
                # remove any previous networks weights that were worse:
                if os.path.isdir(MODEL_PATH):
                    os.rmdir(MODEL_PATH) 
                if os.path.isfile(MODEL_PATH):
                    os.chmod(MODEL_PATH, stat.S_IWRITE)
                    os.remove(MODEL_PATH)
                
                # save new better model weights
                torch.save(network.state_dict(), MODEL_PATH)

                print('\nmodel weights saved to '+str(RUN_ID)+'\n')

                log_artifact(LOG_NEW_ARTIFACT_TO, MODEL_PATH, best_loss, config)     
            
        # test and plot
        model_location = 'nicoranabhat/ellipse_fitting/'+LOG_NEW_ARTIFACT_TO+':latest'
        test_and_plot(model_location, RUN_ID, NUM_TRAINING_ELLIPSES, False)

        run.finish()