#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 19:24:53 2019

@author: kowshik
"""

import argparse
import os
import torch
import pickle
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN
import glob
from sklearn.manifold import TSNE
# Training settings
parser = argparse.ArgumentParser(description='PyTorch semi-supervised MNIST')

parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--z_dim', type=int, default=10 ,metavar='z_dim',
                    help='dimension of latent space')


args = parser.parse_args()
cuda = torch.cuda.is_available()

seed = 10


kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
n_classes = 10
z_dim = args.z_dim
X_dim = 784
y_dim = 10
train_batch_size = args.batch_size
valid_batch_size = args.batch_size
N = 1000
epochs = args.epochs


##################################
# Load data and create Data loaders
##################################
def load_data(data_path='../data/'):
    print('loading data!')
    trainset_labeled = pickle.load(open(data_path + "train_labeled.p", "rb"))
    # print(trainset_labeled.shape)
    trainset_unlabeled = pickle.load(open(data_path + "train_unlabeled.p", "rb"))
    # Set -1 as labels for unlabeled data
   # trainset_unlabeled.train_labels = torch.from_numpy(np.array([-1] * 47000))
    trainset_unlabeled.targets = torch.from_numpy(np.array([-1] * 47000))
    validset = pickle.load(open(data_path + "validation.p", "rb"))

    train_labeled_loader = torch.utils.data.DataLoader(trainset_labeled,
                                                       batch_size=train_batch_size,
                                                       shuffle=False ,**kwargs)

    train_unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled,
                                                         batch_size=train_batch_size,
                                                         shuffle=True, **kwargs)

    valid_loader = torch.utils.data.DataLoader(validset, batch_size=valid_batch_size, shuffle=True)

    return train_labeled_loader, train_unlabeled_loader, valid_loader


##################################
# Define Networks
##################################
# Encoder
class Q_net(nn.Module):
    def __init__(self,z_dim):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        # Gaussian code (z)
        self.lin3gauss = nn.Linear(N, z_dim)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)

        return xgauss


# Decoder
class P_net(nn.Module):
    def __init__(self,z_dim):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, X_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin3(x)
        return F.sigmoid(x)


class D_net_gauss(nn.Module):
    def __init__(self):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)


        return F.sigmoid(self.lin3(x))


####################
# Utility functions
####################
def save_model(model, filename):
    print('Best model so far, saving it...')
    torch.save(model.state_dict(), filename)


def report_loss(epoch, D_loss_gauss, G_loss, recon_loss):
    '''
    Print loss
    '''
    print('Epoch-{}; D_loss_gauss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'.format(epoch,
                                                                                   D_loss_gauss.item(),
                                                                                   G_loss.item(),
                                                                                   recon_loss.item()))

#%%
def create_latent(Q, loader):
    '''
    Creates the latent representation for the samples in loader
    return:
        z_values: numpy array with the latent representations
        labels: the labels corresponding to the latent representations
    '''
    Q.eval()
    labels = []

    for batch_idx, (X, target) in enumerate(loader):

        X = X * 0.3081 + 0.1307
        X.resize_(loader.batch_size, X_dim)
        X, target = Variable(X), Variable(target)
        labels.extend(target.data.tolist())
        if cuda:
            X, target = X.cuda(), target.cuda()
        # Reconstruction phase
       
        z_sample = Q(X)
        if batch_idx > 0:
            z_values = np.concatenate((z_values, np.array(z_sample.data.tolist())))
        else:
            z_values = np.array(z_sample.data.tolist())
    labels = np.array(labels)

    return z_values, labels
#%%

def visualize_tsne_plots(data,labels):
    """
    ==========================
    tSNE to visualize digits
    ==========================
    
    Here we use :class:`sklearn.manifold.TSNE` to visualize the digits
    datasets. Indeed, the digits are vectors in a 8*8 = 64 dimensional space.
    We want to project them in 2D for visualization. tSNE is often a good
    solution, as it groups and separates data points based on their local
    relationship.
    
    """
    
    X=data
    
    ############################################################
    # Fit and transform with a TSNE
    
    tsne = TSNE(n_components=2, random_state=0)
    
    ############################################################
    # Project the data in 2D
    X_2d = tsne.fit_transform(X)
    y=labels
    ############################################################
    # Visualize the data

    
    
    plt.figure(figsize=(6, 5))
    
    plt.scatter(X_2d [:, 0],X_2d [:, 1], c=y, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('visualization of the latent variables with a gaussian prior with 10 dimensions', fontsize=12);

    plt.show()

#%%
if __name__ == '__main__':
    train_labeled_loader, train_unlabeled_loader, valid_loader = load_data()

    # Tejas - to visualize the image gemerated from a random gaussian variable.
    # more famncy explorations are there in viz.py. 
    os.chdir('/media/kowshik/Data1/gan_expts/AAE_pytorch/script/checkpoints/disc_unif')
    Q_checkpoints=  sorted(glob.glob('Q*'))
    P_checkpoints=  sorted(glob.glob('P*'))
    
    for i in range(1):#len(Q_checkpoints)):
        Q_checkpoint=Q_checkpoints[i]
        P_checkpoint=P_checkpoints[i]
        
        Z_dim= int(P_checkpoint.split('_')[1])
        
        Q = Q_net(Z_dim).cuda()
        P = P_net(Z_dim).cuda()
        
        Q.load_state_dict(torch.load(Q_checkpoint))
        P.load_state_dict(torch.load(P_checkpoint))
        clf=KNN(1)
        latent_labeled= create_latent(Q,train_labeled_loader)
        indices= np.arange(latent_labeled[0].shape[0])
        visualize_tsne_plots(latent_labeled[0],latent_labeled[1])
        X_train, X_test, y_train, y_test,idx1,idx2 = train_test_split(latent_labeled[0],latent_labeled[1] ,indices, test_size=0.33, random_state=42)
        clf.fit(X_train,y_train)
        pred=clf.predict(X_test)
        print('accuracy and Z_dim', accuracy_score(pred,y_test),Q_checkpoint,Q_checkpoint )
    
    #%%
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    import pandas as pd
    import seaborn as sns
    from scipy.spatial.distance import cdist
    full_image_data= train_labeled_loader.dataset.data
    image_train= full_image_data[idx1,:]
    image_test= full_image_data[idx2,:]
    dist_matrix= cdist(X_test,X_train)
    #%%
    k=5# no of nearest neighbors
    #p=100# no of nearest test images
    _,p=np.unique(y_test,return_index=True) #first occurence of each unique value
    nearest_neighbors= dist_matrix.argsort(axis=1)[:,:k]
    # take ten test images for image retrieval
    test_images=image_test[p,:]
    near_images= [ image_train[nearest_neighbors[i,:] ] for i in p]
    
    
    near_images_cat= torch.cat(near_images)
    test_images=image_test[p,:]
    test_and_near_images=list(zip(test_images,near_images))
    
    test_and_near_images_1=[torch.cat((i[0].unsqueeze(dim=0),i[1])) for i in test_and_near_images]
    test_and_near_images_1_cat= torch.cat(test_and_near_images_1)
    import torchvision
    
    plt.imshow(torchvision.utils.make_grid(torch.unsqueeze(test_and_near_images_1_cat,dim=1),nrow=6).permute(1,2,0),cmap='gray');
  
    plt.axis('off')
    plt.show()
    
    
    
    
    
    
    
    
    
    #%%
    
    
    def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
    
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
        df_cm = pd.DataFrame(cm, range(10),
                  range(10))
        plt.figure(figsize = (10,10))
        sns.set(font_scale=1.1)#for label size
        sns.heatmap(df_cm, annot=True,cmap="Blues")
#    
    
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    
    plot_confusion_matrix(y_test, pred, classes=np.arange(10),
                          title='Confusion matrix, without normalization')
    
    # Plot normalized confusion matrix
    plot_confusion_matrix(y_test, pred, classes=np.arange(10), normalize=True,
                          title='Normalized confusion matrix')
    
    plt.show()
    
        
