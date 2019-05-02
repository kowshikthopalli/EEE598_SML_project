 

# Towards Discriminative Latent Spaces for Compact Hashing and Image Retrieval
Pytorch code for reproducing the results of our group project for EEE598 course (Statistical Machine Learning: From Theory to Algorithms) taught by Profesor Gautam Dasarathy
 
# Group members-

1.Kowshik Thopalli 

2.Tejas Gokhale

The main codes are in the script folder. If you dont want to train, and just want to evaluate please use aae_pytorch_eval.py and change the checkpoint directory file to the desired prior ('gaussian' or'bernoulli' or 'disc_unif').
We have provided checkpoints for z_dim =10 for all of these three cases.
This script will run a KNN classifier, plot the confusion matrix and also plots retrieved images for query test images.



Requirements- 

PyTorch 1.0  with cuda and 

Torchvision

and usual Python ML libraries such as  

Scipy, Numpy, pandas,

scikit-learn, 

seaborn and matplotlib


