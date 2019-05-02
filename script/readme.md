# Main Scripts 

aae_pytorch_'prior'.py where 'prior' could be 'gauss' or 'bernoulli' or 'disc_unif' will train the networks and save the checkpoints. will also employ a KNN algorithm, plot the confusion matrix and visualize t-SNE plots
we provide encoder and decoder checkpoints for z_dim=10
for other z_dim's we need to run (eg)
python aae_pytorch_bernoulli.py --zdim 3


aae_pytorch_eval.py-
Will only evaluate the networks with teh checkpoints already provided and give KNN accuracy, and also retrieve  train images from test query images. 

