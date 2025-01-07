import os
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.utils.data as Data
import torch.nn.functional as F

############################ Torch utils ###################################

def seed_torch(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)#as reproducibility docs
    torch.manual_seed(seed)# as reproducibility docs
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False# as reproducibility docs
    torch.backends.cudnn.deterministic = True# as reproducibility docs
    
class TLinear(nn.Module):
    def __init__(self, size_in, size_out, T):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.weights = nn.Parameter(torch.rand(T, size_in, size_out)).requires_grad_()
        self.bias = nn.Parameter(torch.rand(T, size_out)).requires_grad_()
    
    def forward(self, x):
        return torch.matmul(x.unsqueeze(2), self.weights).squeeze(2) + self.bias
    

def trainNN(model, train_dataset, lr=1e-2, epochs=200, weight_decay=1e-5, 
            T_max=200, small=1e-7, batch_size=128, verbose=True, display_intvl=20, **args):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,  T_max=T_max)
    
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

    train_loss, last_loss = 0, 0
    for epoch in range(epochs):
        for batch_idx, data in enumerate(train_loader):
            loss = model.loss(*data, **args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss
        if verbose and (epoch % display_intvl == 0 or torch.abs(last_loss - train_loss) < small):
            print('Epoch {} (lr: {:11f}):  training-Loss: {:.3f}'.format(
                epoch, scheduler.get_last_lr()[0], train_loss.item()))
        if torch.abs(last_loss - train_loss) < small:
            break
        last_loss = train_loss
        train_loss = 0
        scheduler.step()
    return model
    
    
############################ Propensity Model ###################################

class PropensityModel(torch.nn.Module):
    def __init__(self, X_dim, T, hidden_dim, seed=48, device='cpu'):
        super(PropensityModel, self).__init__()
        seed_torch(seed)
        self.lstm = nn.LSTM(X_dim + 1, hidden_dim[0]) # extra 1 as past assignment (base as 0)
        modules = []
        for i in range(1, len(hidden_dim)):
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(0.1))
            modules.append(TLinear(hidden_dim[i-1], hidden_dim[i], T))
        self.linear_layers = nn.Sequential(*modules)
        self.device = device
        
    def forward(self, X, A):
        A0 = torch.concat([torch.zeros(X.shape[0], 1).to(self.device), A[:,:-1]], axis=1)
        XA0 = torch.concat([X, A0.unsqueeze(2)], axis=2) # add past assignment as one of the covariate
        H, _ = self.lstm(XA0)
        pi = nn.Sigmoid()(self.linear_layers(H))
        return pi.flatten(1)
    
    def calculate_propensity(self, X, A):
        pi = self.forward(X, A)
        return torch.where(A>0, pi, 1-pi).detach()
    
    def loss(self, X, A, **args):
        pi = self.forward(X, A)
        return F.binary_cross_entropy(pi, (A+1)/2)  

############################ DTR Model ###################################
class DTRModel(torch.nn.Module):
    def __init__(self, X_dim, T, hidden_dim, seed=48, device='cpu'):
        super(DTRModel, self).__init__()
        seed_torch(seed)
        self.lstm = nn.LSTM(X_dim + 1, hidden_dim[0]) # extra 1 as past assignment (base as 0)
        modules = []
        for i in range(1, len(hidden_dim)):
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(0.1))
            modules.append(TLinear(hidden_dim[i-1], hidden_dim[i], T))
        self.T = T
        self.linear_layers = nn.Sequential(*modules)
        self.device = device
    
    def forward(self, X, A):
        A0 = torch.concat([torch.zeros(X.shape[0], 1).to(self.device), A[:,:-1]], axis=1)
        XA0 = torch.concat([X, A0.unsqueeze(2)], axis=2) # add past assignment as one of the covariate
        H, _ = self.lstm(XA0)
        f = self.linear_layers(H).flatten(1)
        return f
    
    def residual(self, A, R, R_hat):
        res = R - R_hat
        A_new = A * torch.sign(res.repeat_interleave(self.T).reshape(-1, self.T))
        R_new = torch.abs(res)
        return A_new, R_new
    
    def loss(self):
        NotImplementedError('loss function not implemented')

    def predict(self, X):
        Ahat = torch.zeros(X.shape[0], self.T+1).to(self.device)
        for t in range(self.T):
            Ahat[:, t+1] = torch.sign(self.forward(X, Ahat[:,:self.T])).detach()[:,t]
        return Ahat[:, 1:]

asy_indicator_larger_than_0 = lambda f, A, theta: 1 / (torch.exp(torch.clip(-theta * A * f, -80, 80)) + 1)

#############################################################
###### Stage Aware Learning - Hamming Loss
#############################################################
class SALModel(DTRModel):
    def __init__(self, X_dim, T, hidden_dim, seed=48, device='cpu'):
        super(SALModel, self).__init__(X_dim, T, hidden_dim, seed, device)
        
    def loss(self, X, A, R, pi, R_hat=None, theta=None, **args):
        A_new, R_new = self.residual(A, R, R_hat)
        f = self.forward(X, A_new)
        return (-R_new / pi.prod(1) * asy_indicator_larger_than_0(f, A_new, theta).mean(1)).sum()

#############################################################
###### Stage Weighted Learning - Weighted Hamming Loss ######
#############################################################
class SWLModel(DTRModel):
    def __init__(self, X_dim, T, hidden_dim, seed=48, device='cpu'):
        super(SWLModel, self).__init__(X_dim, T, hidden_dim, seed, device)
        
    def loss(self, X, A, R, pi, R_hat, sw, theta=None, **args):
        A_new, R_new = self.residual(A, R, R_hat)
        f = self.forward(X, A_new)
        return (-R_new / pi.prod(1) * (sw * asy_indicator_larger_than_0(f, A_new, theta)).sum(1)).sum()
    

class WeightNNModel(torch.nn.Module):
    def __init__(self, X_dim, T, hidden_dim, seed=48, device='cpu'):
        super(WeightNNModel, self).__init__()
        seed_torch(seed)
        self.lstm = nn.LSTM(X_dim + 1, hidden_dim[0])
        modules = [] # extra 1 for current assignment
        for i in range(1, len(hidden_dim)):
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(0.1))
            modules.append(TLinear(hidden_dim[i-1], hidden_dim[i], T))
        self.T = T
        self.linear_layers = nn.Sequential(*modules)
        self.beta = nn.Parameter(torch.ones(1, T)).requires_grad_()
        self.device = device

    def forward(self, X, A):
        A0 = torch.concat([torch.zeros(X.shape[0], 1).to(self.device), A[:,:-1]], axis=1)
        XA0 = torch.concat([X, A0.unsqueeze(2)], axis=2) # add past assignment as one of the covariate
        H, _ = self.lstm(XA0)

        constrained_beta = nn.Softmax(dim=1)(torch.abs(self.beta)).flatten().reshape(self.T, 1)
        HA = torch.concat([H, A.unsqueeze(2)], axis=2)*constrained_beta
        r = (self.linear_layers(HA)).flatten(1)
        R = (r).sum(1)
        return R
    
    def getWeights(self):
        return (self.beta**2 / (self.beta**2).sum()).detach()
    
    def loss(self, X, A, R, theta=None, **args):
        R_hat = self.forward(X, A)
        return torch.mean((R_hat - R)**2) + theta * torch.norm(self.beta, 1).sum()
