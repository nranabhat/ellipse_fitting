import os
import wandb

NUM_TRAINING_ELLIPSES = 100000

wandb.login()
run = wandb.init(project='ellipse_fitting')
artifact_name = 'TrainingSet-100000-Ellipses-30-Shots-Contrast-0.65'
#artifact_name = 'TestingSet-10000-Ellipses-30-Shots-Contrast=0.65'
artifact = wandb.Artifact(artifact_name, type='dataset')

datasets_path = r"C:\Users\Nicor\OneDrive\Documents\KolkowitzLab\Ellipse fitting\Datasets\Updated Contrast Datasets"
training_set_path = os.path.join(datasets_path, "Training Set ("+str(NUM_TRAINING_ELLIPSES)+" ellipses)")
testing_set_path = os.path.join(datasets_path, 'Testing Set')

# Add a file to the artifact's contents
artifact.add_file(os.path.join(training_set_path, 'training1Labels.csv'))
artifact.add_file(os.path.join(training_set_path, 'training1Phi_d.csv'))
artifact.add_file(os.path.join(training_set_path, 'training1X.csv'))
artifact.add_file(os.path.join(training_set_path, 'training1Y.csv'))


# Save the artifact version to W&B and mark it as the output of this run
wandb.run.log_artifact(artifact)