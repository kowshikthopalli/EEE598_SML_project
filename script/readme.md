# Main Scripts 

aae_pytorch_'prior'.py where 'prior' could be 'gauss' or 'bernoulli' or 'disc_unif' will train the networks and save the checkpoints. 

These files will also employ a KNN algorithm, plot the confusion matrix and visualize t-SNE plots at the end.

We provide encoder and decoder neworks checkpoints for z_dim=10 for each of the priors in checkpoints folder


for other z_dim's we need to run (eg)
python aae_pytorch_bernoulli.py --zdim 3


aae_pytorch_eval.py-
Will only evaluate the networks with teh checkpoints already provided and give KNN accuracy, and also retrieve  train images from test query images. 

