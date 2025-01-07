import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.stats import binom
import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################################# Generate Simulation Data #######################################################

IS_WEIGHT = 10

X_eval_mode = None
n, num_measurements, num_variables, num_important_stages = [None] * 4
X,O,A,A_mask,r,R,R_mean,sw_arr,si_arr,XA0 = [None] * 10

def define_global_variables(n_param, nm_param, nv_param, ni_param):
    global n, num_measurements, num_variables, num_important_stages, X_eval_mode
    n, num_measurements, num_variables, num_important_stages = n_param, nm_param, nv_param, ni_param
    
def generate_random_matrix(row_sum, nc, label_true, label_false, seed=48):
    X = np.zeros((len(row_sum), nc))
    X[:,:] = label_false
    for i, K in enumerate(row_sum):
        idx = np.random.choice(list(range(nc)), K, replace=False)
        X[i, idx] = label_true
    return X

def generate_matching_probability(match_mode = None, match_prob = None, seed=48):
    global num_measurements
    np.random.seed(seed)
    matching_prob = None
    if match_mode == 'binom': # phi(K) ~ binom(T, k, p)
        matching_prob = [binom.pmf(i, num_measurements, float(match_prob)) for i in range(num_measurements+1)]
    elif match_mode == 'linear': # phi(K) ~ k / T
        matching_prob = [k/num_measurements for k in range(num_measurements + 1)]
    elif match_mode == 'quadratic':  # phi(K) ~ (k / T)**2
        matching_prob = [(k/num_measurements)**2 for k in range(num_measurements + 1)]
    elif match_mode == 'nonlinear': # phi(K) ~ random dirichlet + last
        matching_prob = np.hstack([
            np.random.dirichlet(np.ones(num_measurements)) * (1-float(match_prob)), float(match_prob)])
    matching_prob = np.array(matching_prob)
    matching_prob = matching_prob / matching_prob.sum()
    matching_prob[0] = 1 - np.sum(matching_prob[1:]) 
    return matching_prob

def mask_opposite_assignment(matching_prob, seed=48):
    matching_n = np.random.multinomial(n, matching_prob)
    row_sum = [i for i, k in enumerate(matching_n) for j in range(k)]
    A_mask = generate_random_matrix(row_sum, num_measurements, 1, -1, seed)
    return A_mask

def construct_stage_weights_list(sw=None, si=None, seed=48):    
    np.random.seed(seed)
    if num_important_stages == 0:
        stage_weights_list = np.ones(num_measurements) / num_measurements
        stage_idx_list = np.array([])
    elif sw is None:
        stage_weights_list = []
        idx = np.random.choice(list(range(num_measurements)), size = num_important_stages, replace=False)

        if num_important_stages == 3:
            stage_weights_list = [
                np.random.uniform(0, 0.01) if i not in idx else np.random.uniform(0.99, 1) for i in range(num_measurements)
            ]
        else:
            stage_weights_list = [
                np.random.uniform(0, 0.2) if i not in idx else np.random.uniform(0.8, 1) for i in range(num_measurements)
            ]
        stage_weights_list = np.array(stage_weights_list)
        stage_weights_list = stage_weights_list / stage_weights_list.sum()
        stage_idx_list = sorted(idx)
    else:
        stage_weights_list = sw
        stage_idx_list = si
    print("Stage Weights: ", stage_weights_list)
    return stage_weights_list, stage_idx_list


############# Linear Rules  ######################
### D(x) = Beta * X[:,selected]
### r(x) = sw * D * A
X_basis, optimal_betas, optimal_decision_rule = [None] * 3
RX_basis, optimal_Rbetas, reward_function, IR_function = [None] * 4
num_var, interactions_num, interaction_select, function_select = [None] * 4
HOMO_RULE = None

def generate_immediate_reward(seed = 48):
    global RX_basis, optimal_Rbetas, reward_function, IR_function
    RX_basis = generate_random_matrix(
        np.random.randint(low=5, high=num_variables, size=num_measurements), num_variables, 1, 0, seed=seed+1
    )
    optimal_Rbetas = np.random.normal(0, 1, size = (num_measurements, 2*num_variables))

    def reward_function(x, ft, homo=True, noise=True):
        ft = 0 if homo else ft
        base_reward = (optimal_Rbetas[ft,:num_variables] * x * RX_basis[ft,:num_variables]).sum(axis=1)
        base_reward = base_reward + np.random.normal(0, 1, size = x.shape[0]) if noise else base_reward
        # base_reward = (base_reward - base_reward.min()) / max(base_reward.max() - base_reward.min(), 1e-5)
        return base_reward

    def IR_function(X, t, At, Ot, is_scale, homo=True, noise=True):
        base_rewards = reward_function(X[:,t,:], t, homo, noise)
        is_scale = sw_arr[t]
        rewards = is_scale * (base_rewards / 10 + At * Ot)
        return rewards 
    
    
def linear_decision_rules(seed=48):
    global X_basis, optimal_betas, optimal_decision_rule
    np.random.seed(seed)
    X_basis = generate_random_matrix(np.random.randint(low=5, high=num_variables, size=num_measurements), 
                                     num_variables, 1, 0, seed=seed)
    optimal_betas = np.random.normal(0, 1, size = (num_measurements, num_variables))
     
    def optimal_decision_rule(x, ft, homo=True):
        ft = 0 if homo else ft
        return (optimal_betas[ft,] * x * X_basis[ft,]).sum(axis=1)

def nonlinear_decision_rules(seed=48):
    global optimal_decision_rule, num_var, interactions_num, interaction_select, function_select
    np.random.seed(seed)
    nonlinear_functions = np.array([lambda x: x**i for i in range(1, 4)] + \
                                   [lambda x: np.sign(x), lambda x: np.arctan(x)])

    # define how many interaction terms for each stage
    num_var = np.random.choice(list(range(10, 16)), size=num_measurements) 
    # define which var will be selected for the interaction term
    interactions_num = [np.random.choice([i for i in range(1, 3)], size = v) for v in num_var] 
    interaction_select = [generate_random_matrix(v, num_variables, 1, 0, seed=seed) 
                          for v in interactions_num]
    funcion_select = [generate_random_matrix(v, len(nonlinear_functions), 1, 0, seed=seed) 
                      for v in interactions_num]
    def optimal_decision_rule(x, ft, homo=True):
        ft = 0 if homo else ft
        result = np.zeros(shape=(x.shape[0], num_var[ft]))
        for i in range(num_var[ft]):
            X_sub = x[:,np.where(interaction_select[ft][i,] == 1)]
            function_sub = nonlinear_functions[np.where(funcion_select[ft][i,] == 1)]
            result[:,i] = np.sum([function_sub[j](X_sub[:,0,j]) for j in range(len(function_sub))], axis=0)
        assert(result.shape[1] == num_var[ft])
        result = (result - result.mean(0)) / result.std(0)
        return result.sum(1)

def X_evolve(X, A, t, mode=1, seed=1):
    global X_eval_mode
    X_eval_mode = mode
    
    np.random.seed(seed)
    if t == X.shape[1]-1:
        return X
    
    A0_idx, A1_idx = np.where(A[:,t] == -1)[0], np.where(A[:,t] == 1)[0]

    X_next = np.zeros((X.shape[0], num_variables))
    if mode == 1:
        A0_noise = np.random.normal(0, 1, size=(len(A0_idx), num_variables))
        A1_noise = np.random.normal(0, 1, size=(len(A1_idx), num_variables))
        X_next[A0_idx,] = 0.6 * X[A0_idx,t,:num_variables] + 0.8 * A0_noise
        X_next[A1_idx,] = 0.8 * X[A1_idx,t,:num_variables] + 0.6 * A1_noise
    elif mode == 2:
        A0_noise = np.random.normal(0, 1, size=(len(A0_idx), num_variables))
        A1_noise = np.random.normal(0, 1, size=(len(A1_idx), num_variables))
        X_next[A0_idx,] = 0.2 * X[A0_idx,t,:num_variables] + 0.8 * A0_noise
        X_next[A1_idx,] = 0.8 * X[A1_idx,t,:num_variables] + 0.2 * A1_noise
    elif mode == 3:
        A0_noise = np.random.normal(0.1, 1, size=(len(A0_idx), num_variables))
        A1_noise = np.random.normal(-0.1, 1, size=(len(A1_idx), num_variables))
        X_next[A0_idx,] = X[A0_idx,t,:num_variables] + A0_noise
        X_next[A1_idx,] = X[A1_idx,t,:num_variables] + A1_noise
    elif mode == 4:
        A0_noise = np.random.normal(0.5, 1, size=(len(A0_idx), num_variables))
        A1_noise = np.random.normal(-0.5, 1, size=(len(A1_idx), num_variables))
        X_next[A0_idx,] = X[A0_idx,t,:num_variables] + A0_noise
        X_next[A1_idx,] = X[A1_idx,t,:num_variables] + A1_noise
    else:
        raise ValueError("Invalid mode")
    X[:,t+1,:num_variables] = torch.tensor(X_next).to(X.device) if type(X) == torch.Tensor else X_next
    if X.shape[-1] == num_variables + 1:
        X[:,t+1,-1] = torch.tensor(A[:,t]).to(X.device) if type(X) == torch.Tensor else A[:,t] 
    return X

def generate_data(matching_prob = None, homo=True, linear=True, X_evolve_mode=1, seed=48, sw=None, si=None, device="cpu"):
    global X,O,A,A_mask,r,R,sw_arr, si_arr, XA0, HOMO_RULE
    HOMO_RULE = homo
    if linear:
        print("Using linear optimal rule")
        linear_decision_rules(seed=seed)
    else:
        print("Using nonlinear optimal rule")
        nonlinear_decision_rules(seed=seed)

    if HOMO_RULE:
        print("Using homogeneous optimal rule")
    else:
        print("Using heterogeneous optimal rule")
        
    generate_immediate_reward(seed = seed)
    
    np.random.seed(seed)
    X = np.random.normal(0, 1, size = [n, num_measurements, num_variables])
    O,A,r = np.zeros((n, num_measurements)), np.zeros((n, num_measurements)), np.zeros((n, num_measurements))
    A_mask = mask_opposite_assignment(matching_prob, seed=seed)
    sw_arr, si_arr = construct_stage_weights_list(sw, si, seed=seed)
    for t in range(num_measurements):
        O_values = optimal_decision_rule(X[:,t,:], t, homo=HOMO_RULE)
        Ot = np.sign(O_values)
        At = Ot.copy() * A_mask[:,t]
        O[:,t] = Ot; A[:,t] = At
        is_scale = IS_WEIGHT if t in si_arr else 1  # scale even more for the important stages
        r[:,t] = IR_function(X, t, At, Ot, is_scale, homo=HOMO_RULE, noise=True)
        X = X_evolve(X, A, t, mode=X_evolve_mode)
    # R = r.mean(axis=1)
    R = r.sum(axis=1)
    # R = (sw_arr * r).sum(axis=1)
    X,O,A,A_mask,r,R = shuffle(X,O,A,A_mask,r,R)
    X, A, O, R = torch.tensor(X).float(), torch.tensor(A).float(), torch.tensor(O).float(), torch.tensor(R).float()
    X, A, O, R = X.to(device), A.to(device), O.to(device), R.to(device)
    A0 = torch.concat([torch.zeros(n, 1).to(device), A[:,1:]], axis=1)
    XA0 = torch.concat([X, A0.unsqueeze(2)], axis=2)
    sw_arr, si_arr = np.array(sw_arr), np.array(si_arr)
    
flatten_tensor = lambda x: x.detach().cpu().numpy().flatten()
train_idx, val_idx, test_idx = [None] * 3
X_train, R_train, X_val, R_val, X_test, R_test = [None] * 6
O_train, A_train, O_val, A_val, O_test, A_test = [None] * 6

def train_test_split_data(train_size = None, seed=48):
    global train_idx, val_idx, test_idx
    global X_train, R_train, X_val, R_val, X_test, R_test
    global O_train, A_train, O_val, A_val, O_test, A_test
    global R_mean
    
    train_size = 0.8 if train_size is None else train_size
    train_idx, test_idx = train_test_split(list(range(n)), train_size = train_size, random_state = seed)
    val_idx, test_idx = train_test_split(test_idx, train_size = 0.2, random_state = seed)
    
    X_train, X_val, X_test = X[train_idx, ], X[val_idx,], X[test_idx,]
    R_train, R_val, R_test = R[train_idx, ], R[val_idx,], R[test_idx,]
    O_train, O_val, O_test = O[train_idx, ], O[val_idx,], O[test_idx,]
    A_train, A_val, A_test = A[train_idx, ], A[val_idx,], A[test_idx,]
    
    R_mean = R_train.mean()

def VF_NP(X, A, O):
    r_hat = np.zeros_like(A)
    for t in range(num_measurements):
        is_scale = IS_WEIGHT if t in si_arr else 1
        r_hat[:,t] = IR_function(X, t, A[:,t], O[:,t], is_scale, homo=HOMO_RULE, noise=False)
    # rescale_R = r_hat.mean(1) 
    rescale_R = r_hat.sum(1) 
#     + (-torch.min(R).detach().cpu().numpy())
    return rescale_R.mean()

def VF(X, A, O):
    r_hat = np.zeros_like(A.cpu())
    for t in range(num_measurements):
        is_scale = IS_WEIGHT if t in si_arr else 1
        r_hat[:,t] = IR_function(X.detach().cpu().numpy(), t, A[:,t].detach().cpu().numpy(), O[:,t].detach().cpu().numpy(), is_scale, homo=HOMO_RULE, noise=False)
    # rescale_R = r_hat.mean(1) 
    rescale_R = r_hat.sum(1)
#     + (-torch.min(R).detach().cpu().numpy())
    return rescale_R.mean()

def VF_model(model,XA0,X,O):
    A_hat = torch.sign(model(XA0))
    return VF(X, A_hat, O)

def seed_torch(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)#as reproducibility docs
    torch.manual_seed(seed)# as reproducibility docs
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False# as reproducibility docs
    torch.backends.cudnn.deterministic = True# as reproducibility docs

def obtain_results(model, seed=1, time_slice=False, old=False):
    if not old:
        return obtain_results_new(model, seed, time_slice)
    else:
        return obtain_results_old(model, seed, time_slice)
    
def obtain_results_new(model, seed=1, time_slice=False):
    eval_dict = {}
    seed_torch(seed)
    for name, idx in zip(["train", "val", "test"], [train_idx, val_idx, test_idx]):
        Ahat = np.zeros((len(idx), num_measurements))
        Ote = np.zeros((len(idx), num_measurements))
        Rte = np.zeros((len(idx), num_measurements))
        XA0_cp = torch.clone(XA0[idx,]).cpu()
        XA0_cp[:,1:,:] = 0 # Only know the baseline covariates
    
        # If everything follows the optimal
        Ooptte = np.zeros((len(idx), num_measurements))
        ROpt = np.zeros((len(idx), num_measurements))
        device = XA0.device
        XA0_optim_cp = torch.clone(XA0[idx,]).cpu()
        
        for t in range(num_measurements):
            if time_slice:
                Ahat[:,t] = np.sign(model(XA0_cp[:,:,:-1].to(device)).detach().cpu().numpy())[:,t]
            else:
                Ahat[:,t] = model(XA0_cp, t)
            Ote[:,t] = np.sign(optimal_decision_rule(XA0_cp[:,t,:-1].cpu().detach().numpy(), t, homo=HOMO_RULE))
            is_scale = IS_WEIGHT if t in si_arr else 1
            Rte[:,t] = IR_function(XA0_cp[:,:,:-1].detach().cpu().numpy(), t, Ahat[:,t], Ote[:,t], is_scale, homo=HOMO_RULE, noise=False)
            
            # everything is assigned optimal
            Ooptte[:,t] = np.sign(optimal_decision_rule(XA0_optim_cp[:,t,:-1].cpu().detach().numpy(), t, homo=HOMO_RULE))
            ROpt[:,t] = IR_function(XA0_optim_cp[:,:,:-1].detach().cpu().numpy(), t, Ooptte[:,t], Ooptte[:,t], is_scale, homo=HOMO_RULE, noise=False)
        
            if t + 1 == num_measurements:
                break
            # Evolve based on d(X)
            XA0_cp = X_evolve(XA0_cp, Ahat, t, mode=X_eval_mode, seed=seed)
            assert all(XA0_cp[:,t+1,-1].detach().cpu().numpy()== Ahat[:,t])
            # Evolve based on O
            XA0_optim_cp = X_evolve(XA0_optim_cp, Ooptte, t, mode=X_eval_mode, seed=seed)

        acc = (Ahat == Ote).mean() * 100
        acc_imp = (Ahat[:,si_arr] == Ote[:,si_arr]).mean() * 100 if len(si_arr) > 0 else np.nan
        # vf = Rte.mean(1).mean() # Total reward if every state is assigned according to the treatment rule (X evolves accoridng to d(X))
        vf = Rte.sum(1).mean()
        # vf_optim = ROpt.mean(1).mean() # total reward if every state is assigned optimal (X evolve accoding to O)
        vf_optim = ROpt.sum(1).mean()
        eval_dict[name] = {"Acc": np.round(acc, 3), "Imp-Acc": np.round(acc_imp, 3) if len(si_arr) > 0 else np.nan,
                            "VF": np.round(vf, 3), "VF-obs": np.round(R[idx,].mean().item(), 3), "VF-optim": np.round(vf_optim, 3)}
    return eval_dict


def obtain_results_old(model, seed=1, time_slice=False):
    eval_dict = {}
    seed_torch(seed)
    for name, idx in zip(["train", "val", "test"], [train_idx, val_idx, test_idx]):
        Ahat = np.zeros((len(idx), num_measurements))
        Ote = np.zeros((len(idx), num_measurements))
        Rte = np.zeros((len(idx), num_measurements))
        ROpt = np.zeros((len(idx), num_measurements))
        # Fixed covariates
        device = XA0[idx,].device
        XA0_cp = torch.clone(XA0[idx,]).cpu()
        
        for t in range(num_measurements):
            if time_slice:
                Ahat[:,t] = np.sign(model(XA0_cp[:,:,:-1].to(device)).detach().cpu().numpy())[:,t]
            else:
                Ahat[:,t] = model(XA0_cp, t)
            Ote[:,t] = np.sign(optimal_decision_rule(XA0_cp[:,t,:-1].cpu().detach().numpy(), t, homo=HOMO_RULE))
            is_scale = IS_WEIGHT if t in si_arr else 1
            Rte[:,t] = IR_function(XA0_cp[:,:,:-1].cpu().numpy(), t, Ahat[:,t], Ote[:,t], is_scale, homo=HOMO_RULE, noise=False)
            ROpt[:,t] = IR_function(XA0_cp[:,:,:-1].cpu().numpy(), t, Ote[:,t], Ote[:,t], is_scale, homo=HOMO_RULE, noise=False)

            if t + 1 == num_measurements:
                break

        acc = (Ahat == Ote).mean() * 100
        acc_imp = (Ahat[:,si_arr] == Ote[:,si_arr]).mean() * 100 if len(si_arr) > 0 else np.nan
        # vf = Rte.mean(1).mean() # Total reward if every state is assigned according to the treatment rule (X evolves accoridng to d(X))
        vf = Rte.sum(1).mean()
        # vf_optim = ROpt.mean(1).mean() # total reward if every state is assigned optimal (X evolve accoding to O)
        vf_optim = ROpt.sum(1).mean()
        eval_dict[name] = {"Acc": np.round(acc, 3), "Imp-Acc": np.round(acc_imp, 3) if len(si_arr) > 0 else np.nan,
                            "VF": np.round(vf, 3), "VF-obs": np.round(R[idx,].mean().item(), 3), "VF-optim": np.round(vf_optim, 3)}
    return eval_dict