# loading trained models and plotting their fit estimations

from audioop import avg
from cmath import inf
import logging
import shutil
import stat
import time
import torch
import wandb
import os
from ast import literal_eval
import numpy as np
from Sweep_phi_d import CheckpointSaver,Dataset,\
build_dataset,build_network,build_optimizer,build_scheduler,train_epoch,get_test_loss,test_and_plot

RUN_ID = 'ipxi4flh'
VERSION_NUM = 'latest'
NUM_TRAINING_ELLIPSES = '1000000'
NAME_OF_ARTIFACT_TO_USE = 'nicoranabhat/ellipse_fitting/best-run8-phi-'+RUN_ID+'-'+NUM_TRAINING_ELLIPSES+'-trainingEllipses.pt:'+str(VERSION_NUM)
#NUM_TRAINING_ELLIPSES = '1000000'
#NAME_OF_ARTIFACT_TO_USE = 'nicoranabhat/ellipse_fitting/best-mlp-sweep-phi-'+RUN_ID+'.pt:'+str(VERSION_NUM)
LOG_NEW_ARTIFACT_TO = f'best-run9-phi-'+str(RUN_ID)+'-'+NUM_TRAINING_ELLIPSES+'-trainingEllipses.pt'

#wandbpath = r"C:\Users\Nicor\OneDrive\Documents\KolkowitzLab\ellipse_fitting\Learners\wandb"   
wandbpath = r"D:\Nico Ranabhat\Ellipse Fitting\ellipse_fitting\Learners\wandb"
pathname = os.path.join(wandbpath, 'best-'+NUM_TRAINING_ELLIPSES+'-trainingellipses-run-for-sweep-'+RUN_ID)
MODEL_PATH = os.path.join(pathname, 'weights_tensor.pt')

NUM_NEW_EPOCHS = 10

SAVE_MODEL = True  # If True, save model perormance as wandb artifact. If just running to debug, set to False 

wandb.login()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__': 
    with wandb.init(project='ellipse_fitting') as run:
        
        def log_artifact(artifact_location_path, model_path, best_loss, test_loss, config, actual_epoch, current_lr, adjusted_milestones):
            config_string={k:str(v) for k,v in config.items()}
            config_string['loss'] = str(best_loss)
            config_string['test loss'] = str(test_loss)
            config_string['epoch'] = str(actual_epoch)
            config_string['current_lr'] = str(current_lr)
            config_string['adjusted_milestones'] = adjusted_milestones
            config_string['#ellipses (training run)'] = str(NUM_TRAINING_ELLIPSES)

            artifact = wandb.Artifact(artifact_location_path, type='model', metadata=config_string)
            artifact.add_file(model_path)
            wandb.run.log_artifact(artifact)   

        print("Loading artifact at: "+NAME_OF_ARTIFACT_TO_USE)
        artifact = run.use_artifact(NAME_OF_ARTIFACT_TO_USE, type='model')
        artifact_dir = artifact.download()
        state_dicts_path = os.path.join(artifact_dir, 'weights_tensor.pt')
        config = artifact.metadata
        config['loss'] = 1.5
        config['batch_size'] = '5000'
        # config = {'loss': 0.5, 
        # 'gamma': '0.2454507300590375', 
        # 'epoch': '15', 
        # 'optimizer': 'sgd', 
        # 'batch_size': '10',
        # 'milestones': '[10]', 
        # 'starting_lr': '0.05402435040728651', 
        # 'second_layer_size': '512'}
        # print('config:\n'+str(config))

        trainloader = build_dataset(int(config['batch_size']), int(NUM_TRAINING_ELLIPSES), True)
        network = build_network(int(config['second_layer_size']),clamp_output=True)
        network.load_state_dict(torch.load(state_dicts_path)['model_state_dict'])
        optimizer = build_optimizer(network, config['optimizer'], float(config['current_lr']))
        #optimizer.load_state_dict(torch.load(state_dicts_path)['optimizer_state_dict'])
        # adjusted milestones takes into account that the model has already been trained a bit during the sweep
        if 'adjusted_milestones' in config:
            adjusted_milestones_str = '['+config['adjusted_milestones']+']'
            adjusted_milestones = np.array(literal_eval(adjusted_milestones_str))-int(config['epoch'])
        else: adjusted_milestones = np.array(literal_eval(config['milestones']))-int(config['epoch'])-1
        scheduler = build_scheduler(optimizer, adjusted_milestones, float(config['gamma']))
        scheduler.load_state_dict(torch.load(state_dicts_path)['scheduler_state_dict'])
        # not sure if this^ loads the lr correctly...

        best_loss = float(config['loss'])

        for epoch in range(NUM_NEW_EPOCHS):
            actual_epoch_num = int(config['epoch']) + epoch + 1
            
            avg_loss = train_epoch(network, trainloader, optimizer, scheduler)
            avg_test_loss = get_test_loss(config['batch_size'], network)

            # after epoch log loss to wandb
            wandb.log({"loss": avg_loss, "test loss": avg_test_loss, "epoch": epoch}, commit=True)
            print('EPOCH: '+str(actual_epoch_num)+'     LOSS: '+str(avg_loss)+'     TEST LOSS: '+str(avg_test_loss))
            print('optimizer lr: '+str(optimizer.param_groups[0]['lr']))

            # if it's the first or last epoch, wait 3 seconds for wandb to log the loss
            if (epoch==0 or epoch==NUM_NEW_EPOCHS-1):
                time.sleep(3)

            # save weights of current best epoch (will save best model for whole network over training if top_n=1)
            if SAVE_MODEL and (avg_loss <= best_loss):
                #checkpoint_saver(network, avg_loss, epoch, optimizer, config)

                # save network on local drive and as artifact on wandb
                logging.info(f"Current metric value {avg_loss} better than {best_loss}.\n"+ \
"Saving model at "+MODEL_PATH+'\nLogging model weights to W&B artifact '+LOG_NEW_ARTIFACT_TO)

                best_loss = avg_loss
                
                if not os.path.exists(pathname): os.makedirs(pathname)
                # remove any previous networks weights that were worse:
                if os.path.isdir(MODEL_PATH):
                    os.rmdir(MODEL_PATH) 
                if os.path.isfile(MODEL_PATH):
                    os.chmod(MODEL_PATH, stat.S_IWRITE)
                    os.remove(MODEL_PATH)
                
                # save new better model weights
                #torch.save(network.state_dict(), MODEL_PATH)
                torch.save({'epoch': actual_epoch_num,
                        'model_state_dict': network.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': avg_loss}, 
                        MODEL_PATH)

                print('Model weights and state_dicts saved.\n')

                current_lr = optimizer.param_groups[0]['lr']
                log_artifact(LOG_NEW_ARTIFACT_TO, MODEL_PATH, best_loss, avg_test_loss, config, actual_epoch_num, current_lr, adjusted_milestones)     
        
        #delete all artifact versions that arne't top 5
        time.sleep(3)
        api = wandb.Api()
        artifact_type, artifact_name = 'model', 'nicoranabhat/ellipse_fitting/'+LOG_NEW_ARTIFACT_TO # fill in the desired type + name
        try:
            for version in api.artifact_versions(artifact_type, artifact_name):
                if len(version.aliases) == 1: #has aliase 'latest'
                    best_version_num = int(version.version[1])
                if int(version.version[1]) < (best_version_num - 4):
                    version.delete()
            print('\nOld wandb artifacts deleted')
            # test and plot
            model_location = 'nicoranabhat/ellipse_fitting/'+LOG_NEW_ARTIFACT_TO+':latest'
            test_and_plot(model_location, RUN_ID, NUM_TRAINING_ELLIPSES, is_sweep=False)

        except Exception:
            print('\nNo better model was found in this run... No new models saved to wandb')

        # delete any files saved to local machine
        if os.path.isdir(pathname): shutil.rmtree(pathname)


        run.finish()