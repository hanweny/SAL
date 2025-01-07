import os
import torch
from torch import optim, nn
import torch.utils.data as Data
from torch.nn import functional as F
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

utils = None
def reload(ut):
    global utils
    utils = ut
    
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
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.weights = nn.Parameter(torch.rand(utils.num_measurements, size_in, size_out)).requires_grad_()
        self.bias = nn.Parameter(torch.rand(utils.num_measurements, size_out)).requires_grad_()
    
    def forward(self, x):
        return torch.matmul(x.unsqueeze(2), self.weights).squeeze(2) + self.bias

############################ WeighNN ###################################

class WeightNNModel(torch.nn.Module):
    def __init__(self, H_dim, hidden_dim, f_class=TLinear, seed=48):
        super(WeightNNModel, self).__init__()
        seed_torch(seed)
        self.lstm = nn.Linear(utils.num_variables, H_dim)
#         self.lstm = nn.LSTM(utils.num_variables + 1, H_dim) # extra 1 as past assignment (base as 0)
        modules = [f_class(H_dim + 1, hidden_dim[0])] # extra 1 for current assignment
        for i in range(1, len(hidden_dim)):
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(0.1))
            modules.append(f_class(hidden_dim[i-1], hidden_dim[i]))
        self.linear_layers = nn.Sequential(*modules)
        self.beta = nn.Parameter(torch.ones(1, utils.num_measurements)).requires_grad_()
    
    def forward(self, X, A):
        H = self.lstm(X)
        constrained_beta = nn.Softmax(dim=1)(torch.abs(self.beta)).flatten().reshape(utils.num_measurements, 1)
        HA = torch.concat([H, A.unsqueeze(2)], axis=2)*constrained_beta
        r = (self.linear_layers(HA)).flatten(1)
        R = (r).sum(1)
        return R
    
    def getWeights(self):
        return (self.beta**2 / (self.beta**2).sum()).detach().cpu().numpy()
    
    def loss(self, X, A, R, theta=0):
        R_hat = self.forward(X, A)
        return torch.mean((R_hat - R)**2) + theta * torch.norm(self.beta, 1).sum()

# Theta as l1 loss weight
WeightLoss = lambda model, X, A, R, pi=None, sw=None, theta=0, Rest=None: model.loss(X, A, R, theta)

def WeightVerbose(model, epoch, scheduler, train_loss, val_loss):
    sw_pred = model.getWeights()
    si_pred = sw_pred.argsort()[0]
    overlap = len(set(si_pred[-utils.num_important_stages:]).intersection(set(utils.si_arr)))
    print('Epoch {} (lr: {:11f}, {}/{}):  MSE: {:.3f} (Val: {:.3f})\tIS: {}\t'.format(
        epoch, scheduler.get_last_lr()[0], overlap, utils.num_important_stages, train_loss.item(), val_loss.item(), si_pred))
    print(sw_pred)

############################ Propensity Model ###################################

class propensityModel(torch.nn.Module):
    def __init__(self, H_dim, hidden_dim, f_class=None, seed=48):
        super(propensityModel, self).__init__()
        seed_torch(seed)
        self.lstm = nn.LSTM(utils.num_variables, H_dim) # extra 1 as past assignment (base as 0)
        modules = [TLinear(H_dim, hidden_dim[0])]
        for i in range(1, len(hidden_dim)):
            modules.append(nn.ReLU())
            modules.append(TLinear(hidden_dim[i-1], hidden_dim[i]))
        self.linear_layers = nn.Sequential(*modules)
    
    def forward(self, X):
        H, hidden = self.lstm(X)
        pi = nn.Sigmoid()(self.linear_layers(H))
        return pi.flatten(1)
    
    def calculate_propensity(self):
        pi = self.forward(utils.X)
        return torch.where(utils.A>0, pi, 1-pi).detach()
    
    def loss(self, X, A):
        pi = self.forward(X)
        return F.binary_cross_entropy(pi, A)
    
pLoss = lambda model, X, A, R=None, pi=None, sw=None, theta=None, Rest=None: model.loss(X, (A+1)/2)

def pVerbose(model, epoch, scheduler, train_loss, val_loss):
    with torch.no_grad():
        pi_t, pi_v = model(utils.X[utils.train_idx, ]), model(utils.X[utils.val_idx,])
        Ahat_t = torch.where(pi_t > 0.5, 1, -1).detach().cpu().numpy()
        Ahat_v = torch.where(pi_v > 0.5, 1, -1).detach().cpu().numpy()
        acc_t = (Ahat_t == utils.A[utils.train_idx,].detach().cpu().numpy()).mean() * 100
        acc_v = (Ahat_v == utils.A[utils.val_idx,].detach().cpu().numpy()).mean() * 100
        print('Epoch {} (lr: {:11f}):  Loss: {:.3f}\tAcc: {:.2f}% (Val-Loss: {:.3f}\tAcc: {:.2f}%)'.format(
            epoch, scheduler.get_last_lr()[0], train_loss.item(), acc_t, val_loss.item(), acc_v))
        
####################################################################################
############################ OWL Model ###################################
####################################################################################
from collections import Counter
class DTRModel(torch.nn.Module):
    def __init__(self, H_dim, hidden_dim, f_class=TLinear, seed=48):
        super(DTRModel, self).__init__()
        seed_torch(seed)
        self.lstm = nn.LSTM(utils.num_variables, H_dim) # extra 1 as past assignment (base as 0)
        modules = [f_class(H_dim, hidden_dim[0])]
        for i in range(1, len(hidden_dim)):
            # modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(0.01))
            modules.append(f_class(hidden_dim[i-1], hidden_dim[i]))
        self.linear_layers = nn.Sequential(*modules)
    
    def forward(self, X):
        # H, hidden = self.lstm(X)
        f = self.linear_layers(X).flatten(1)
        # f = nn.BatchNorm1d(utils.num_measurements).to(f.device)(f)
        # f = nn.Tanh()(f)
        return f
        return (f-f.mean(0)) / torch.clamp(f.std(0), 1e-2)
    
    def loss(self, loss_func, X, A, R, pi, sw=None, theta=None, Rest=None):
        f = self.forward(X)
        # print((R-R.min()*2).min(), (R-R.min()*2).max())
        # loss = loss_func(f, A, R - R.min(), pi, sw, theta)
        if not Rest:
            Rest = R.min() # Ensure positivity
        res = R - Rest
        A_new = A * torch.sign(res.repeat_interleave(utils.num_measurements).reshape(-1, utils.num_measurements))
        R_new = torch.abs(res)
        loss = loss_func(f, A_new, R_new, pi, sw, theta)
        return loss.sum() 
    
def DTRVerbose(model, epoch, scheduler, train_loss, val_loss):
    with torch.no_grad():
        Ahat_t = torch.sign(model(utils.X[utils.train_idx, ]))
        Ahat_v = torch.sign(model(utils.X[utils.val_idx, ]))
        acc_t = (Ahat_t == utils.O[utils.train_idx,]).detach().cpu().numpy().mean() * 100
        acc_v = (Ahat_v == utils.O[utils.val_idx,]).detach().cpu().numpy().mean() * 100
        accImp_t = (Ahat_t[:,utils.si_arr] == utils.O[utils.train_idx,:][:,utils.si_arr]).detach().cpu().numpy().mean() * 100 if len(utils.si_arr) > 0 else np.nan
        accImp_v = (Ahat_v[:,utils.si_arr] == utils.O[utils.val_idx,:][:,utils.si_arr]).detach().cpu().numpy().mean() * 100 if len(utils.si_arr) > 0 else np.nan
        vf_t = utils.VF_model(model, utils.X[utils.train_idx,], utils.X[utils.train_idx,], utils.O[utils.train_idx,])
        vf_v = utils.VF_model(model, utils.X[utils.val_idx,], utils.X[utils.val_idx,], utils.O[utils.val_idx,])
        outer_indicator = KIPWE_SURROGATE(model(utils.X[utils.train_idx, ]), utils.A[utils.train_idx,])
        k_acc = ((torch.abs((utils.O[utils.train_idx,] == utils.A[utils.train_idx,]).sum(1) - outer_indicator.argmax(1))) <= 2).float().mean() * 100
        print('Epoch {} (lr: {:11f}):  Loss: {:.3f} (O-Acc: {:.2f}%, Imp-Acc: {:.2f}%, K-acc: {:.2f}% VF: {:.3f})\tVal-Loss: {:.3f} (O-Acc: {:.2f}%, Imp-Acc: {:.2f}%, VF: {:.3f})'.format(
            epoch, scheduler.get_last_lr()[0], train_loss.item(), acc_t, accImp_t, k_acc, vf_t, val_loss.item(), acc_v, accImp_v, vf_v))

def DTREval(model):
    eval_dict = {}
    with torch.no_grad():
        for name, idx in zip(["train", "val", "test"], [utils.train_idx, utils.val_idx, utils.test_idx]):
            Ahat = torch.sign(model(utils.X[idx,]))
            acc = (Ahat == utils.O[idx,]).detach().cpu().numpy().mean() * 100
            acc_imp = (Ahat[:,utils.si_arr] == utils.O[idx,:][:,utils.si_arr]).detach().cpu().numpy().mean() * 100 if len(utils.si_arr) > 0 else np.nan
            vf = utils.VF_model(model, utils.X[idx,], utils.X[idx,], utils.O[idx,])
            vf_obs = utils.VF(utils.X[idx,], utils.A[idx,], utils.O[idx,])
            vf_optim = utils.VF(utils.X[idx,], utils.O[idx,], utils.O[idx,])
            eval_dict[name] = {"Acc": np.round(acc, 3), "Imp-Acc": np.round(acc_imp, 3) if len(utils.si_arr) > 0 else np.nan,
                               "VF": np.round(vf, 3), "VF-obs": np.round(vf_obs, 3), "VF-optim": np.round(vf_optim, 3)}
    return eval_dict

def DTREval2(model, seed=1, old=False):
    model.eval()
    with torch.no_grad():
        res = utils.obtain_results(model, seed, time_slice=True, old=old)
    model.train()
    return res

#############################################################
###### Stage Weighted Learning - Weighted Hamming Loss ######
#############################################################
asy_indicator_larger_than_0 = lambda f, A, theta: 1 / (torch.exp(torch.clip(-theta * A * f, -80, 80)) + 1)
# asy_indicator_larger_than_0 = lambda f, A, theta: 1 / (torch.exp(-theta * A * f) + 1)

# asy_indicator_larger_than_0 = lambda f, A, theta: (A+1)/2 * (1/((-theta*f).exp() + 1)).log() + (1 -(A+1)/2)*(1 - 1/((-theta*f).exp() + 1)).log()

SWLOSS_FUNC = lambda f, A, R, pi, sw, theta: -R / pi.prod(1) * (sw * asy_indicator_larger_than_0(f, A, theta)).sum(1)
SWLoss = lambda model, X, A, R, pi, sw, theta=None, Rest=None: model.loss(SWLOSS_FUNC, X, A, R, pi, sw, theta, Rest)

#############################################################
###### Stage Awared Learning - Hamming Loss
#############################################################
SALOSS_FUNC = lambda f, A, R, pi, sw=None, theta=None: -R / pi.prod(1) * asy_indicator_larger_than_0(f, A, theta).mean(1)
SALoss = lambda model, X, A, R, pi, sw=None, theta=None, Rest=None: model.loss(SALOSS_FUNC, X, A, R, pi, sw, theta, Rest)


#############################################################
###### K-IPWE
#############################################################
asy_indicator_equal = lambda X_arr, a=0, std=None: torch.exp(-(X_arr-a)**2 / std)
def KIPWE_SURROGATE(f, A, theta=1, std=1):
    K = asy_indicator_larger_than_0(A, f, theta).sum(1, keepdim=True).repeat_interleave(utils.num_measurements+1, axis=1) - torch.arange(utils.num_measurements+1).to(f.device)
    outer_indicator = asy_indicator_equal(K, 0, std)
    # print("K accuracy:  ", ((torch.abs((utils.O[utils.train_idx,] == A).sum(1) - outer_indicator.argmax(1))) <= 2).mean())
    return outer_indicator

KIPWE_FUNC = lambda f, A, R, pi, sw=None, theta=None, std=3: -R / pi.prod(1) * (sw * KIPWE_SURROGATE(f, A, theta, std)).sum(1)
KIPWELOSS = lambda model, X, A, R, pi, sw=None, theta=None, Rest=None: model.loss(KIPWE_FUNC, X, A, R, pi, sw, theta, Rest)

#############################################################
###### DTRESLO learning
#############################################################
def DTRESLO_SURROGATE(f, A):
    return (1 + 2/torch.pi * torch.arctan(torch.pi * (A * f) / 2)).prod(1)
DTRESLO_FUNC = lambda f, A, R, pi, sw, theta: -R/pi.prod(1) * DTRESLO_SURROGATE(f, A)
DTRESLOLoss = lambda model, X, A, R, pi, sw, theta, Rest=None: model.loss(DTRESLO_FUNC, X, A, R, pi, sw, theta, Rest)

#############################################################
###### SOWL learning （smooth version) - equivalent to our method
#############################################################
def soft_max(f, A, theta):
    x = torch.clip((1 - A*f) / theta, -80, 80)
    return (x.exp().sum(1) + 1).log() * theta

SOWLOSS_FUNC = lambda f, A, R, pi, sw, theta: R/pi.prod(1) * (soft_max(f, A, theta) - 1)
SOWLoss = lambda model, X, A, R, pi, sw, theta, Rest=None: model.loss(SOWLOSS_FUNC, X, A, R, pi, sw, theta, Rest)

OWLOSS_FUNC = lambda f, A, R, pi, sw, theta: -R * torch.min(A*f - 1, axis=1).values
OWLoss = lambda model, X, A, R, pi, sw, theta, Rest=None: model.loss(OWLOSS_FUNC, X, A, R, pi, sw, theta, Rest)

############################ Train Model ################################## 
model = None

def trainNN(modelClass, H_dim, hidden_dim, lossFunc, pi=None, sw=None, theta=5, Rest=None,
            lr=1e-2, epochs=200, weight_decay=1e-5, T_max=200, small=1e-7, f_class=TLinear, 
            verbose=True, verboseFunc=None, display_intvl=20, num_batch=3, seed=48, device="cpu"):
    global model
    model = modelClass(H_dim, hidden_dim, f_class=f_class, seed=seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,  T_max=T_max)
    if pi is None:
        pi = torch.clone((utils.A+1)/2)
        for t in range(utils.num_measurements):
            idx1, idx0 = pi[:,t]==1, pi[:,t]==0
            prob_one = pi[:,t].mean()
            pi[idx1,t], pi[idx0,t] = prob_one, 1-prob_one

        pi = torch.ones_like(utils.A).to(utils.A.device)

    # R = (utils.R[utils.train_idx,] - utils.R[utils.train_idx,].mean()) / utils.R[utils.train_idx,].std()
    R = utils.R[utils.train_idx,]
    # R = utils.R[utils.train_idx,] / utils.R[utils.train_idx,].std()

    train_dataset = Data.TensorDataset(utils.X[utils.train_idx,], utils.A[utils.train_idx,], R, pi[utils.train_idx,])
    train_loader = Data.DataLoader(train_dataset, batch_size=min(1028, len(train_dataset)), shuffle=True) 
    Xv,Av,Rv,Pv = utils.X[utils.val_idx,], utils.A[utils.val_idx,], utils.R[utils.val_idx,], pi[utils.val_idx]

    train_loss, last_loss = 0, 0
    for epoch in range(epochs):
        for batch_idx, (Xb, Ab, Rb, pib, ) in enumerate(train_loader):
            loss = lossFunc(model, Xb, Ab, Rb, pib, sw, theta, Rest)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss
        if verbose and (epoch % display_intvl == 0 or torch.abs(last_loss - train_loss) < small):
            with torch.no_grad():
                val_loss = lossFunc(model, Xv, Av, Rv, Pv, sw, theta)
            if verboseFunc is None:
                print('Epoch {} (lr: {:11f}):  Loss: {:.3f}'.format(epoch, scheduler.get_last_lr()[0], train_loss.item()))
            else:
                verboseFunc(model, epoch, scheduler, train_loss, val_loss)
        if torch.abs(last_loss - train_loss) < small:
            break
        last_loss = train_loss
        train_loss = 0
        scheduler.step()
    if verbose:
        with torch.no_grad():
            print("Validation Loss:  {:.3f}".format(lossFunc(model, Xv,Av,Rv,Pv, sw, theta)))
    return model