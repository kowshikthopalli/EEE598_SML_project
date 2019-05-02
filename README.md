# EEE598_SML_project
Pytorch code for reproducing the results of our group project.

__

The main codes are in teh script folder. If you dont want to train, and just want to evaluate please use aae_pytorch_eval.py and change the checkpoint directory file to the desired prior('gaussian' or'bernoulli' or 'disc_unif'.
We have provided checkpoints for z_dim =10 for all of these three cases.
This script will run a KNN classifier, plot the confusion matrix and also plots retrieved images for query test images.

__

Requirements- 

'PyTorch 1.0' 

'Torchvision'  

and usual Python ML libraries such as  

Scipy

Numpy

scikit-learn 

pandas

seaborn

Also since this is a deep-learning project- will need a Nvidia GPU  with cuda and PyTorch configured to use this GPU if you intend to train the networks

