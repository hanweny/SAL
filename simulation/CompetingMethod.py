import numpy as np

utils = None
def set_utils(ut):
    global utils
    utils = ut

kernel = None
def set_kernel(k):
    global kernel
    kernel = k

def update_global_variable():
    try:
        update_R_global(immediate=False)
        R_prepare_data()
    except Exception as e:
        print("WEIRD DATA CONVERSION ERROR HAPPENED!!!", e)
    R_prepare_data()

def update_R_global(immediate=True):
    kernel.user_ns['HH_train'] = utils.X_train.cpu().numpy().transpose(1, 0, 2)
    kernel.user_ns['HH_test'] = utils.X_test.cpu().numpy().transpose(1, 0, 2)
    kernel.user_ns['AA_train'] = utils.A_train.cpu().numpy().astype('int').transpose(1, 0)
    kernel.user_ns['AA_test'] = utils.A_test.cpu().numpy().astype('int').transpose(1, 0)
    kernel.user_ns['pi_train'] = np.ones(shape = (utils.num_measurements, utils.X_train.shape[0]))
    if immediate:
        r = utils.r[utils.train_idx, ].cpu().numpy()
        # r = utils.r[utils.train_idx, ] - utils.r[utils.train_idx, ].min(0).reshape(1, -1)
        kernel.user_ns['RR_train'] = (r / utils.num_measurements).transpose(1, 0)
    else:
        T = utils.num_measurements
        R = utils.R[utils.train_idx, ].cpu().numpy()
        # R = R / R.std()
        # R = (utils.R[utils.train_idx, ] - utils.R[utils.train_idx, ].min()).cpu()
        kernel.user_ns['RR_train'] = np.repeat(R, T).reshape(-1, T)
def R_prepare_data():
    R_command = '''
        library('DynTxRegime')
        set.seed(0)
        
        T <- dim(HH_train)[1]
        N <- dim(HH_train)[2]
        P <- dim(HH_train)[3]
    '''
    R_pass_in_params = "-i HH_train -i HH_test -i AA_train -i AA_test -i RR_train -i pi_train"
    kernel.run_cell_magic('R', R_pass_in_params, R_command)
    return


###################################################################################################
# DynTxRegime Methods
###################################################################################################
def DynTxRegime_evolve(XA0, t, is_zero_one=True):
    kernel.user_ns['Heval'] = XA0.cpu().numpy()[:,:(t+1),:-1].transpose(1, 0, 2)
    kernel.user_ns['Aeval'] = (XA0.cpu().numpy()[:,:t,-1].transpose(1, 0) + 1) / 2
    kernel.user_ns['teval'] = t+1
    make_predictions = '''
    x0 <- cbind(Heval[teval,,])
    # if (teval > 1) {
    #     for (j in (teval-1):1) {
    #         x0 <- cbind(x0, Heval[j,,], Aeval[j,], matrix(Heval[j,,], nrow=dim(Heval)[2])*Aeval[j,])
    #     }
    # }
    df.train <- data.frame(x0)
    Apred <- optTx(model_list[[teval]], df.train)$optimalTx
    '''
    kernel.run_cell_magic('R', '-i teval -i Heval -i Aeval', make_predictions)
    A = kernel.run_line_magic('R', 'Apred') 
    if is_zero_one:
        A = A * 2 - 1
    return A

###################################################################################################
# QLearning
###################################################################################################
def QLearning(seed=1, old=False):
    QLearning_command = '''
    model_list <- list()
    y <- RR_train[T]
    N <- dim(HH_train)[2]
    for (t in T:1) {
        x0 <- cbind(HH_train[t,,])
        # if (t > 1) {
        #     for (j in (t-1):1) {
        #         x0 <- cbind(x0, HH_train[j,,], (AA_train[j,]+1)/2, matrix(HH_train[j,,], nrow=N)*(AA_train[j,]+1)/2)
        #     }
        # }
        df.train <- data.frame(cbind(x0, (AA_train[t,]+1)/2))
        nvar <- dim(df.train)[2]
        formula <- as.formula(paste("~ 1 + ", paste(colnames(df.train)[1:(nvar-1)], collapse = " + ")))
        moMain <- buildModelObj(model = formula, solver.method = 'lm')
        moCont <- buildModelObj(model = formula, solver.method = 'lm')
        fitSS <- qLearn(moMain = moMain, moCont = moCont, data = df.train, 
            response = y, txName = colnames(df.train)[nvar], verbose=FALSE)
        model_list[[t]] <- fitSS
    }
    '''
    kernel.run_cell_magic('R', '', QLearning_command)
    return utils.obtain_results(DynTxRegime_evolve, seed=seed, old=old)

###################################################################################################
# AIPW-Classifier
###################################################################################################
def AIPWClass(seed=1, old=False):
    AIPW_command = '''
    library("rpart")
    moPropen <- buildModelObj(model = ~1, solver.method = 'glm', solver.args = list('family'='binomial'),
        predict.method = 'predict.glm', predict.args = list(type='response'))

    model_list <- list()
    fitSS <- RR_train[,T]
    for (t in T:1) {
        x0 <- cbind(HH_train[t,,])
        df.train <- data.frame(cbind(x0, (AA_train[t,]+1)/2))
        nvar <- dim(df.train)[2]
        formula <- as.formula(paste("~", paste(colnames(df.train)[1:(nvar-1)], collapse = " + ")))
        moMain <- buildModelObj(model = formula, solver.method = 'lm')
        moCont <- buildModelObj(model = formula, solver.method = 'lm')
        moClass <- buildModelObj(model = formula, solver.method = 'rpart', solver.args = list(method="class"),
                            predict.args = list(type='class'))
        fitSS <- optimalClass(moPropen = moPropen, moMain=moMain, moCont = moCont, moClass = moClass,
                              data = df.train, response = fitSS, txName = colnames(df.train)[nvar], verbose=FALSE)
        model_list[[t]] <- fitSS
    }
    '''
    kernel.run_cell_magic('R', '', AIPW_command)
    return utils.obtain_results(DynTxRegime_evolve, seed=seed, old=old)

###################################################################################################
# BOWL
###################################################################################################
def BOWL(seed, old=False):
    BOWL_command = '''
    set.seed(1)
    moPropen <- buildModelObj(model = ~1, solver.method = 'glm', solver.args = list('family'='binomial'),
        predict.method = 'predict.glm', predict.args = list(type='response'))

    model_list <- list()
    fitSS <- NULL
    for (t in T:1) {
        x0 <- cbind(HH_train[t,,])
        # if (t > 1) {
        #     for (j in (t-1):1) {
        #         x0 <- cbind(x0, HH_train[j,,], (AA_train[j,]+1)/2, matrix(HH_train[j,,], nrow=N)*(AA_train[j,]+1)/2)
        #     }
        # }
        df.train <- data.frame(cbind(x0, (AA_train[t,]+1)/2))
        nvar <- dim(df.train)[2]
        regime.formula <- as.formula(paste("~", paste(colnames(df.train)[1:(nvar-1)], collapse = " + ")))
        fitSS <- bowl(moPropen = moPropen, data = df.train, reward = RR_train[,T], txName = colnames(df.train)[nvar], 
                      regime = regime.formula, BOWLObj = fitSS, lambdas = c(0.5, 1.0), cvFolds = 2L, verbose=FALSE)
        model_list[[t]] <- fitSS
    }

    '''
    kernel.run_cell_magic('R', '', BOWL_command)
    return utils.obtain_results(DynTxRegime_evolve, seed=seed, old=old)

###################################################################################################
# RWL
###################################################################################################
def RWL(seed, old=False):
    RWL_command = '''
    set.seed(1)
    moPropen <- buildModelObj(model = ~1, solver.method = 'glm', solver.args = list('family'='binomial'),
        predict.method = 'predict.glm', predict.args = list(type='response'))
    model_list <- list()
    fitSS <- NULL

    for (t in T:1) {
        df.train <- data.frame(cbind(HH_train[t,,], (AA_train[t,]+1)/2), RR_train[,t])
        nvar <- dim(df.train)[2]
        outcome.formula <- as.formula(
            paste(
                colnames(df.train)[nvar], 
                paste("~1+", paste(colnames(df.train)[1:(nvar-1)], collapse = " + "))
            )
        )
        outcome.fit <- lm(outcome.formula, data = df.train)
        res <- outcome.fit$residuals
        indicator <- as.numeric(res>0)*2-1
        df.train <- data.frame(cbind(HH_train[t,,], ((AA_train[t,] * indicator) + 1)/2))
        nvar <- dim(df.train)[2]
        regime.formula <- as.formula(paste("~", paste(colnames(df.train)[1:(nvar-1)], collapse = " + ")))
        fitSS <- bowl(moPropen = moPropen, data = df.train, reward = abs(res), txName = colnames(df.train)[nvar], 
                      regime = regime.formula, BOWLObj = fitSS, lambdas = c(0.5, 1.0), cvFolds = 2L, verbose=FALSE)
        model_list[[t]] <- fitSS
    }
    '''
    kernel.run_cell_magic('R', '', RWL_command)
    return utils.obtain_results(DynTxRegime_evolve, seed=seed, old=old)


###################################################################################################
# SOWL
###################################################################################################
import cvxpy as cp

SOWL_model = None 
def SOWL(homo = False, seed=1, old=False):
    global SOWL_model
    X = utils.XA0[utils.train_idx,].detach().cpu().numpy()
    A = utils.A[utils.train_idx,].detach().cpu().numpy()
    R = utils.R[utils.train_idx,].detach().cpu().numpy()

    # res = R - np.mean(R)
    # indicator = np.sign(res).reshape(-1,1)
    # A = A * indicator
    # R = np.abs(res)

    R = R - R.min()

    if homo:
        beta = cp.Variable((X.shape[-1], 1))
        f = [X[:,t,:] @ beta[:,0] for t in range(utils.num_measurements)]
    else:
        beta = cp.Variable((X.shape[-1], utils.num_measurements))
        f = [X[:,t,:] @ beta[:,t] for t in range(utils.num_measurements)] 
    xi = cp.Variable(X.shape[0])
    objective = cp.Minimize(
        -cp.mean(R @ (xi + 1)) + cp.norm(beta, 2)**2
    )
    constraints = [xi <= 0] + [xi <= A[:,t] @ f[t] - 1 for t in range(utils.num_measurements)]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    SOWL_model = beta.value
    return utils.obtain_results(SOWL_evolve, seed=seed, old=old)

def SOWL_evolve(X, t):
    if SOWL_model.shape[1] == 1: # homo
        return np.sign((X[:,t,:] @ SOWL_model[:,0]))
    else:
        return np.sign((X[:,t,:] @ SOWL_model[:,t]))
    
###################################################################################################
# C-learning
###################################################################################################
def CLearning_evolve(XA0, t):
    kernel.user_ns['Heval'] = XA0.detach().cpu().numpy()[:,t,:-1]
    kernel.user_ns['teval'] = t+1
    kernel.run_cell_magic('R', '-i teval -i Heval', 'Apred <- as.numeric(I(cbind(1, Heval)%*%CLearning_model_list[[teval]]>0))')
    A = kernel.run_line_magic('R', 'Apred*2-1')       
    return A

def CLearning(seed, nonParam='FALSE', old=False):

    CLearning_command = """
    library(rgenoud)
    library(randomForest)

    mean.ls<-function(y,X0,X1,a) {{
        fit00<-lm(y~X0+a:X1)
        beta<-summary(fit00)$coef[,1]
        m0<-cbind(1,X0)%*%beta[1:(dim(X0)[2]+1)]
        m1<-m0+X1%*%beta[(dim(X0)[2]+2):(dim(X0)[2]+2+dim(X1)[2]-1)] 
        cbind(m1,m0)
    }}

    mean.rf <- function(y, x, a) {{
        ht <- cbind(x, a)
        rf <- randomForest(ht, y, ntree=500)
        m0 <- predict(rf, cbind(x, 0))
        m1 <- predict(rf, cbind(x, 1))
        cbind(m1, m0)
    }}

    z.wt.Clearning<-function(y,a,ph,m1,m0) {{
        ym<-a*y/ph-(a-ph)/ph*m1-(1-a)*y/(1-ph)-(a-ph)/(1-ph)*m0
        b<-ifelse(ym>=0,1,0)
        wt<-abs(ym)
        cbind(b,wt)
    }}

    obq<-function(X,eta) {{
        g<-as.numeric(I(X%*%eta>0))
        mean(wt*(z-g)^2)
    }}

    genetic.search<-function(coef) {{
        nvars<-length(coef)
        Domains<-cbind(c(-100,rep(-1,nvars-1)),c(0,rep(1,nvars-1)))
        pop.size<-1000
        wait<-5
        est<-genoud(fn=obq,X=X,nvars=nvars,print.level=0,max=FALSE,pop.size=pop.size,wait.generations=wait,gradient.check=FALSE,
        BFGS=TRUE,P1=50, P2=50, P3=50, P4=50, P5=50, P6=50, P7=50, P8=50, P9=0,Domains=Domains,starting.values=coef,solution.tolerance=0.00001,optim.method="Nelder-Mead")
        return(est$par)
    }}

    CLearning_model_list = list()

    y <- RR_train[,T]
    for (t in T:1) {{
        x0 <- cbind(HH_train[t,,])
        x1 <- cbind(1, HH_train[t,,])
        At <- (AA_train[t,] + 1 ) /2

        if ({nonParam}) {{
            meanF <- mean.rf(y, x0, At)
        }} else {{
            meanF <- mean.ls(y, x0, x1, At)
        }}
        m1 <- meanF[,1]
        m0 <- meanF[,2]

        pi <- sum(At) / N
        cdata <- z.wt.Clearning(y, At, pi, m1, m0)
        z <- cdata[,1]
        wt <- cdata[,2]

        X <- cbind(1, HH_train[t,,])
        nvars <- dim(X)[2]
        coef <- rep(0,nvars)
        # policy 
        eta.C <- genetic.search(coef)
        CLearning_model_list[[t]] <- eta.C

        g <- as.numeric(I(X%*%CLearning_model_list[[t]]>0))
        C <- m1 - m0
        y <- y + (g-At)^2 * abs(C)
    }}
    """.format(nonParam=nonParam)

    kernel.run_cell_magic('R', '', CLearning_command)
    return utils.obtain_results(CLearning_evolve, seed=seed, old=old)


###################################################################################################
# Naive Learning
###################################################################################################
def naive_model_pos1(X=None, t=None):
    return 1
def naive_model_neg1(X=None, t=None):
    return -1


###################################################################################################
# SHARED PARAMETERS MODEL
###################################################################################################

coef, coef_shared = None, None

import torch
from sklearn.linear_model import LinearRegression

def QL_unshared(seed, old=False):
    global coef
    R = utils.R[utils.train_idx,].cpu()
    X = utils.X[utils.train_idx,].cpu()
    A = (utils.A[utils.train_idx,].unsqueeze(-1).cpu() + 1) / 2
    XAS = torch.cat([X, X*A], dim=2)
    coef = torch.zeros((utils.num_measurements, 2*utils.num_variables+1))
    for t in range(utils.num_measurements-1, -1, -1):
        lr = LinearRegression()
        lr.fit(XAS[:,t,:], R)
        coef[t,] = torch.tensor(np.hstack([lr.intercept_, lr.coef_]))
        main_effect = (coef[t,1:utils.num_variables+1] * X[:,t,:]).sum(1)
        interaction_effect = (coef[t,utils.num_variables+1:] * X[:,t,:] * A[:,t,:]).sum(1)
        R = coef[t,0] + main_effect + torch.clamp(interaction_effect, 0)

    def QLunshared_evolve(XA, t):
        X = XA[:,t,:-1]
        b0, beta, psi = coef[t,0], coef[t,1:1+utils.num_variables], coef[t,1+utils.num_variables:]
        R0, R1 = (b0 + beta*X).sum(1), (b0 + beta*X + psi*X).sum(1)
        Ahat = torch.argmax(torch.vstack([R0, R1]), dim=0)
        return 2*Ahat - 1
    return utils.obtain_results(QLunshared_evolve, seed=seed, old=old)

def QL_shared(seed, MAX_ITER=500, old=False):
    global coef_shared
    iter_num = 0

    R = utils.R[utils.train_idx,]
    X = utils.X[utils.train_idx,]
    A = (utils.A[utils.train_idx,].unsqueeze(-1) + 1) / 2

    beta = coef[:,:utils.num_variables+1]
    psi = coef[:,utils.num_variables+1:].mean(0)
    theta = torch.concat([beta.flip(0).flatten(), psi])
    while True:
        Z = torch.concat([torch.ones(X[:,-1,:].shape[0], 1), X[:,-1,:]], dim=1)
        Z_psi = X[:,-1,:] * A[:,-1]
        for t in range(utils.num_measurements-2, -1, -1):
            Xt = torch.concat([torch.zeros((X[:,t,:].shape[0], Z.shape[1])), 
                            torch.concat([torch.ones(X[:,t,:].shape[0], 1), X[:,t,:]], dim=1)], dim=1)
            Z = torch.concat([Z, torch.zeros((Z.shape[0], X[:,t,:].shape[1]+1))], dim=1)
            Z = torch.concat([Z, Xt], dim=0)
            Z_psi = torch.concat([Z_psi, X[:,t,:] * A[:,t]], dim=0)
        Z = torch.concat([Z, Z_psi], dim=1)
        Ystar = (Z[:,:-psi.shape[0]] @ theta[:-psi.shape[0]]) + (Z[:,-psi.shape[0]:] @ theta[-psi.shape[0]:]).clamp(0)
        Ystar = torch.concat([R, Ystar[:-X[:,0,:].shape[0]]])
        theta_next = torch.linalg.solve(Z.t() @ Z, Z.t() @ Ystar)
        if torch.norm(theta-theta_next)  < 1e-7 or iter_num >= MAX_ITER:
            print("Converged at iteration ", iter_num, " (theta mean:  ", theta.mean().item(), ")" )
            break
        iter_num += 1
        theta = theta_next

    coef_shared = torch.concat([
        theta[:-psi.shape[0],].view(utils.num_measurements, -1).flip(0),
        theta[-psi.shape[0]:].unsqueeze(0).repeat(utils.num_measurements, 1)
    ], dim=1)
    def QLshared_evolve(XA, t):
        X = XA[:,t,:-1]
        b0, beta, psi = coef_shared[t,0], coef_shared[t,1:1+utils.num_variables], coef_shared[t,1+utils.num_variables:]
        R0, R1 = (b0 + beta*X).sum(1), (b0 + beta*X + psi*X).sum(1)
        Ahat = torch.argmax(torch.vstack([R0, R1]), dim=0)
        return 2*Ahat - 1
    return utils.obtain_results(QLshared_evolve, seed, old=old)


###################################################################################################
# DQN
###################################################################################################
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

class DQN:
    def __init__(self, hidden_dim, seed=1, device='cpu'):
        utils.seed_torch(seed)
        self.QNet = _QNET(hidden_dim, seed)
        self.QNet_optimizer = optim.AdamW(self.QNet.parameters(), lr=0.001, amsgrad=True)

        self.targetNet = _QNET(hidden_dim, seed)
        self.targetNet.load_state_dict(self.QNet.state_dict())
                
    def optimize_model(self, state, action, reward, next_state, update_target=False):
        Q = self.QNet(state).gather(1, action.unsqueeze(1))
        expected_Q = self.targetNet(next_state).max(1).values * 0.99 + reward
        
        # Update weights
        criterion = nn.SmoothL1Loss()
        loss = criterion(Q, expected_Q.unsqueeze(1)) 
        self.QNet_optimizer.zero_grad()
        loss.backward( retain_graph=True)
        self.QNet_optimizer.step()
        Q = self.QNet(state).gather(1, action.unsqueeze(1))
        if update_target:
            self._update_target_net(self.targetNet, self.QNet)
        
    def _update_target_net(self, target_net, net, TAU=0.005):
        target_net_state_dict = target_net.state_dict()
        net_state_dict = net.state_dict()
        for key in net_state_dict:
            target_net_state_dict[key] = net_state_dict[key] * TAU + target_net_state_dict[key] * (1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        
class _QNET(nn.Module):
    def __init__(self, hidden_dim, seed=1):
        utils.seed_torch(seed)
        super(_QNET, self).__init__()      
        modules = [nn.Linear(utils.num_variables, hidden_dim[0])]
        for i in range(1, len(hidden_dim)):
            modules.append(nn.LeakyReLU(0.001))
            modules.append(nn.Dropout(0.05))
            modules.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)
    
    def predict_prob(self, x):
        prob = nn.functional.softmax(self.forward(x), dim=1)
        return prob
    
global policy
def train_DQN(seed, old=False):
    X, A, R = utils.X[utils.train_idx,], (utils.A[utils.train_idx,] + 1)/2, utils.R[utils.train_idx,]
    curr = X[:,:-1,:].reshape(X.shape[0]*(X.shape[1]-1), -1)
    next = X[:,1:,:].reshape(X.shape[0]*(X.shape[1]-1), -1)
    a = A[:,:-1].flatten().long()
    r = torch.repeat_interleave(R / utils.num_measurements, utils.num_measurements-1)
    train_data = TensorDataset(curr, a, r, next)
    policy = DQN([16, 2], seed=1)
    for epoch in range(max(50*X.shape[1], 500)):
        for (state, action, reward, next_state) in DataLoader(train_data, batch_size=258, shuffle=True):
            policy.optimize_model(state, action, reward, next_state, update_target=(epoch % 5 == 0))

    print("train done")
    def DQN_evolve(XA, t, policy=policy):
        X = XA[:,t,:-1]
        Ahat = policy.QNet(X).argmax(1)
        return Ahat * 2 - 1
    return utils.obtain_results(DQN_evolve, seed=seed, old=old)
