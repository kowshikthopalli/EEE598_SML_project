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

# Training settings
parser = argparse.ArgumentParser(description='PyTorch semi-supervised MNIST')

parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--z_dim', type=int, default=4, metavar='z_dim',
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
                                                       shuffle=True, **kwargs)

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
    def __init__(self):
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
    def __init__(self):
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

####################
# Train procedure
####################
def train(P, Q, D_gauss, P_decoder, Q_encoder, Q_generator, D_gauss_solver, data_loader):
    '''
    Train procedure for one epoch.
    '''
    TINY = 1e-15
    # Set the networks in train mode (apply dropout when needed)
    Q.train()
    P.train()
    D_gauss.train()

    # Loop through the labeled and unlabeled dataset getting one batch of samples from each
    # The batch size has to be a divisor of the size of the dataset or it will return
    # invalid samples
    for X, target in data_loader:

        # Load batch and normalize samples to be between 0 and 1
        X = X * 0.3081 + 0.1307
        X.resize_(train_batch_size, X_dim)
        X, target = Variable(X), Variable(target)
        if cuda:
            X, target = X.cuda(), target.cuda()

        # Init gradients
        P.zero_grad()
        Q.zero_grad()
        D_gauss.zero_grad()

        #######################
        # Reconstruction phase
        #######################
        z_sample = Q(X)
        X_sample = P(z_sample)
        recon_loss = F.binary_cross_entropy(X_sample + TINY, X.resize(train_batch_size, X_dim) + TINY)

        recon_loss.backward()
        P_decoder.step()
        Q_encoder.step()

        P.zero_grad()
        Q.zero_grad()
        D_gauss.zero_grad()

        #######################
        # Regularization phase
        #######################
        # Discriminator
        Q.eval()
        # z_real_gauss = Variable(torch.randn(train_batch_size, z_dim) * 5.)
        z_real_gauss = Variable(torch.bernoulli(0.5 * torch.ones(train_batch_size, z_dim)))
        if cuda:
            z_real_gauss = z_real_gauss.cuda()

        z_fake_gauss = Q(X)

        D_real_gauss = D_gauss(z_real_gauss)
        D_fake_gauss = D_gauss(z_fake_gauss)

        D_loss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))

        D_loss.backward()
        D_gauss_solver.step()

        P.zero_grad()
        Q.zero_grad()
        D_gauss.zero_grad()

        # Generator
        Q.train()
        z_fake_gauss = Q(X)

        D_fake_gauss = D_gauss(z_fake_gauss)
        G_loss = -torch.mean(torch.log(D_fake_gauss + TINY))

        G_loss.backward()
        Q_generator.step()

        P.zero_grad()
        Q.zero_grad()
        D_gauss.zero_grad()

    return D_loss, G_loss, recon_loss


def generate_model(train_labeled_loader, train_unlabeled_loader, valid_loader):
    torch.manual_seed(10)

    if cuda:
        Q = Q_net().cuda()
        P = P_net().cuda()
        D_gauss = D_net_gauss().cuda()
    else:
        Q = Q_net()
        P = P_net()
        D_gauss = D_net_gauss()

    # Set learning rates
    gen_lr = 0.0001
    reg_lr = 0.00005

    # Set optimizators
    P_decoder = optim.Adam(P.parameters(), lr=gen_lr)
    Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr)

    Q_generator = optim.Adam(Q.parameters(), lr=reg_lr)
    D_gauss_solver = optim.Adam(D_gauss.parameters(), lr=reg_lr)

    if not os.path.exists('../outs'):
        os.makedirs('../outs')

    if not os.path.exists('../checkpoints'):
        os.makedirs('../checkpoints')

    for epoch in range(epochs):
        D_loss_gauss, G_loss, recon_loss = train(P, Q, D_gauss, P_decoder, Q_encoder,
                                                 Q_generator,
                                                 D_gauss_solver,
                                                 train_unlabeled_loader)

        if epoch % 10 == 0:
            report_loss(epoch, D_loss_gauss, G_loss, recon_loss)
            save_model(P, '../checkpoints/P_'+str(z_dim) +'_' + str(epoch) + '.pth')
            save_model(Q, '../checkpoints/Q_'+str(z_dim) +'_' +str(epoch) + '.pth')
            # random_z = torch.bernoulli(0.5 * torch.ones(1, z_dim))

            for i in range(2**z_dim):
                corner = [int(x) for x in np.binary_repr(i, width=z_dim)]
                random_z = torch.bernoulli(torch.Tensor(corner))
                print(random_z)
                plt.imsave('../outs/im_sample_bern_' +str(z_dim) +'_' +str(epoch) +'_' +str(i) +'_.png', P((random_z).cuda()).view(28,28).detach().cpu())


            # random_z = torch.bernoulli(torch.Tensor([0, 0]))
            # plt.imsave('../outs/im_sample_bernoulli_' + str(epoch) + '_00.png', P(().cuda()).view(28,28).detach().cpu())
            # random_z = torch.bernoulli(torch.Tensor([0, 1]))
            # plt.imsave('../outs/im_sample_bernoulli_' + str(epoch) + '_01.png', P(().cuda()).view(28,28).detach().cpu())
            # random_z = torch.bernoulli(torch.Tensor([1, 0]))
            # plt.imsave('../outs/im_sample_bernoulli_' + str(epoch) + '_10.png', P(().cuda()).view(28,28).detach().cpu())
            # random_z = torch.bernoulli(torch.Tensor([1, 1]))
            # plt.imsave('../outs/im_sample_bernoulli_' + str(epoch) + '_11.png', P(().cuda()).view(28,28).detach().cpu())

    return Q, P

if __name__ == '__main__':
    train_labeled_loader, train_unlabeled_loader, valid_loader = load_data()
    Q, P = generate_model(train_labeled_loader, train_unlabeled_loader, valid_loader)
    # Tejas - to visualize the image gemerated from a random gaussian variable.
    # more famncy explorations are there in viz.py. 

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier as KNN
    
    clf=KNN(1)
    latent_labeled= create_latent(Q,train_labeled_loader)
    X_train, X_test, y_train, y_test = train_test_split(latent_labeled[0],latent_labeled[1] , test_size=0.33, random_state=42)
    clf.fit(X_train,y_train)
    pred=clf.predict(X_test)
    print('accuracy',accuracy_score(pred,y_test))
