import time
import warnings
import os
from copy import deepcopy
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.quasirandom import SobolEngine
import botorch
from botorch import fit_gpytorch_model
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.fully_bayesian import MIN_INFERRED_NOISE_LEVEL
from botorch.models.transforms import Normalize
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.utils.warnings import NumericalWarning
from optimize import optimize_acqf_categorical_local_search
from minisom import MiniSom
import multiprocessing as mp

my_idx = 0
max_evals = 500
n_initial_points = 100
batch_size = 4

Cp = 0.01
N_SOM = 5
N_SUCC = 5
N_FAIL = 10
MAX_DEPTH = 3
N_RS = 2
min_num_variables = 3
max_num_variables = 20
p_imp = 0.5

id_list = np.loadtxt("id_list.txt").astype(int)
id_list = id_list.reshape(-1,batch_size)
id_list = id_list[my_idx]
ub_x = np.loadtxt("ub.txt")
ub_x = ub_x.astype(int)
num_dims = len(ub_x)
num_objectives = 3


class SOMT_Node:
    def __init__(self, parent=None, Xbest=None, Ybest=None, hvc=None, SOM=None):
        self.success_counter = 0
        self.failure_counter = 0
        self.rs_counter = 0
        self.hvc = hvc
        self.parent = parent
        self.child_list = []
        self.PS_list = []
        self.PS_list_str = []
        self.PF_list = []
        self.PF_list_str = []
        self.overlapped = False
        self.SOM_list = []
        self.ROI_list_list = []
        self.ROI_list = []
        self.dim_visits = np.ones(num_dims)
        if parent is None:
            self.depth = 0
            self.SOM = None
            self.Xbest = None
            self.Ybest = None
        else:
            self.Xbest = Xbest.numpy()
            self.Ybest = Ybest.numpy()
            self.PS_list.append(self.Xbest)
            self.PS_list_str.append(str(self.Xbest))
            self.PF_list.append(self.Ybest)
            self.PF_list_str.append(str(self.Ybest))
            self.depth = parent.depth + 1
            self.SOM = SOM
            center = self.SOM.winner(self.Xbest)
            self.ROI_list.append(center)
            for i in range(N_SOM):
                for j in range(N_SOM):
                    if (i,j) != center:
                        if np.abs(center[0]-i) < 2 and np.abs(center[1] - j) < 2:
                            self.ROI_list.append((i,j))
            for neighbour in self.parent.child_list:
                if neighbour.overlapped:
                    continue
                for ROI in self.ROI_list.copy():
                    if ROI in neighbour.ROI_list:
                        neighbour.overlapped = True
                        if neighbour.hvc > self.hvc:
                            self.hvc = neighbour.hvc
                            self.Xbest = neighbour.Xbest
                            self.Ybest = neighbour.Ybest
                        if str(neighbour.Xbest) not in self.PS_list_str:
                            self.PS_list.append(neighbour.Xbest)
                            self.PS_list_str.append(str(neighbour.Xbest))
                            self.PF_list.append(neighbour.Ybest)
                            self.PF_list_str.append(str(neighbour.Ybest))
                        for ROI_neighbour in neighbour.ROI_list:
                            if ROI_neighbour not in self.ROI_list:
                                self.ROI_list.append(ROI_neighbour)
                        break
        
    def append_child(self, child):
        self.child_list.append(child)

    def remove_child(self, child):
        self.child_list.remove(child)

    def clean_child(self):
        useless_child = []
        for child in self.child_list:
            if child.overlapped:
                useless_child.append(child)
        for child in useless_child:
            self.child_list.remove(child)

    def is_leaf(self):
        if len(self.child_list) == 0:
            return True
        else:
            return False

    def get_ROI_from_pareto(self, X_PS):
        ROI_list = []
        center = self.SOM.winner(np.array(X_PS[0]))
        ROI_list.append(center)
        for i in range(N_SOM):
            for j in range(N_SOM):
                if (i,j) != center:
                    if np.abs(center[0]-i) < 2 and np.abs(center[1] - j) < 2:
                        ROI_list.append((i,j))
        for k in range(1,len(X_PS)):
            X = X_PS[k]
            center = self.SOM.winner(np.array(X))
            for i in range(N_SOM):
                for j in range(N_SOM):
                    if np.abs(center[0]-i) < 2 and np.abs(center[1] - j) < 2:
                        if (i,j) not in ROI_list:
                            ROI_list.append((i,j))
        return ROI_list

    def update(self, X, Y, ref_point):
        X_PS_all, Y_PF_all = pareto(X.tolist(),Y.tolist())
        HV = calcHypervolume(ref_point, Y_PF_all)
        SOMT_node_list = self.parent.child_list
        for node in SOMT_node_list:
            if node.overlapped:
                continue
            X_ROI, Y_ROI = node.get_XY_in_ROI(X, Y)
            X_PS_ROI, Y_PF_ROI = pareto(X_ROI.tolist(), Y_ROI.tolist())
            real_X_PS_ROI = []
            real_Y_PF_ROI = []
            for i_ROI in range(len(X_PS_ROI)):
                for X_PS in X_PS_all:
                    if (np.array(X_PS_ROI[i_ROI])==np.array(X_PS)).all():
                        real_X_PS_ROI.append(X_PS_ROI[i_ROI])
                        real_Y_PF_ROI.append(Y_PF_ROI[i_ROI])
                        break
            if len(real_X_PS_ROI) == 0:
                node.overlapped = True
                continue
            ROI_list = node.get_ROI_from_pareto(real_X_PS_ROI)
            for neighbour in SOMT_node_list:
                if node is neighbour:
                    continue
                if neighbour.overlapped:
                    continue
                for ROI in ROI_list.copy():
                    if ROI in neighbour.ROI_list:
                        neighbour.overlapped = True
                        node.success_counter = max(node.success_counter, neighbour.success_counter)
                        node.failure_counter = min(node.failure_counter, neighbour.failure_counter)
                        for ROI_neighbour in neighbour.ROI_list:
                            if ROI_neighbour not in ROI_list:
                                ROI_list.append(ROI_neighbour)
                        break
            node.ROI_list = ROI_list
            node.ROI_list_list[-1] = ROI_list
            new_X_ROI, new_Y_ROI = node.get_XY_in_ROI(X, Y)
            new_X_PS_ROI, new_Y_PF_ROI = pareto(new_X_ROI.tolist(), new_Y_ROI.tolist())
            real_X_PS_ROI = []
            real_Y_PF_ROI = []
            for i_ROI in range(len(new_X_PS_ROI)):
                for X_PS in X_PS_all:
                    if (np.array(X_PS)==np.array(new_X_PS_ROI[i_ROI])).all():
                        real_X_PS_ROI.append(new_X_PS_ROI[i_ROI])
                        real_Y_PF_ROI.append(new_Y_PF_ROI[i_ROI])
                        break
            if len(real_X_PS_ROI) == 0:
                node.overlapped = True
                continue
            node.PS_list = real_X_PS_ROI
            node.PF_list = real_Y_PF_ROI
            HVC = []
            for Y_PF in real_Y_PF_ROI:
                Y_PF_all_removed = Y_PF_all.copy()
                Y_PF_all_removed.remove(Y_PF)
                if len(Y_PF_all_removed) == 0:
                    HV_removed = 0
                else:
                    HV_removed = calcHypervolume(ref_point, Y_PF_all_removed)
                HVC.append(HV-HV_removed)
            HVC = np.array(HVC)
            i_max_hvc = np.argmax(HVC)
            node.hvc = HVC[i_max_hvc]
            node.Xbest = np.array(node.PS_list[i_max_hvc])
            node.Ybest = np.array(node.PF_list[i_max_hvc])
        self.parent.clean_child()

    def is_in_ROI(self, X):
        if self.depth != len(self.SOM_list):
            self.SOM_list.append(self.SOM)
            self.ROI_list_list.append(self.ROI_list)
        i = -1
        while i >= -len(self.SOM_list):
            SOM = self.SOM_list[i]
            ROI_list = self.ROI_list_list[i]
            i_X = SOM.winner(np.array(X))
            if i_X not in ROI_list:
                return False
            i -= 1
        return True

    def get_XY_in_ROI(self, Xdataset, Ydataset):
        if self.depth != len(self.SOM_list):
            self.SOM_list.append(self.SOM)
            self.ROI_list_list.append(self.ROI_list)
        Xdataset = Xdataset.numpy()
        Ydataset = Ydataset.numpy()
        i = -1
        while i >= -len(self.SOM_list):
            SOM = self.SOM_list[i]
            ROI_list = self.ROI_list_list[i]
            wm = SOM.win_map(Xdataset, return_indices=True)
            i_X = []
            for region in ROI_list:
                i_X.extend(wm[region])
            Xdataset = Xdataset[i_X]
            Ydataset = Ydataset[i_X]
            i -= 1
        return torch.from_numpy(Xdataset), torch.from_numpy(Ydataset)
            
    def get_X_in_ROI(self, Xdataset):
        if self.depth != len(self.SOM_list):
            self.SOM_list.append(self.SOM)
            self.ROI_list_list.append(self.ROI_list)
        Xdataset = Xdataset.numpy()
        i = -1
        while i >= -len(self.SOM_list):
            SOM = self.SOM_list[i]
            ROI_list = self.ROI_list_list[i]
            wm = SOM.win_map(Xdataset, return_indices=True)
            i_X = []
            for region in ROI_list:
                i_X.extend(wm[region])
            Xdataset = Xdataset[i_X]
            i -= 1
        return torch.from_numpy(Xdataset)
    
    def split(self, X, Y, ref_point):
        X_ROI, Y_ROI = self.get_XY_in_ROI(X, Y)
        som = MiniSom(N_SOM, N_SOM, len(X_ROI[0]))
        som.train_random(X_ROI.numpy(), 10000)
        X_PS_ROI, Y_PF_ROI = pareto(X_ROI.tolist(), Y_ROI.tolist())
        HV = calcHypervolume(ref_point, Y_PF_ROI)
        for i in range(len(X_PS_ROI)):
            Y_PF_ROI_removed = Y_PF_ROI.copy()
            Y_PF_ROI_removed.remove(Y_PF_ROI[i])
            if len(Y_PF_ROI_removed) == 0:
                HV_removed = 0
            else:
                HV_removed = calcHypervolume(ref_point, Y_PF_ROI_removed)
            hvc = HV - HV_removed
            node = SOMT_Node(parent=self, Xbest=torch.tensor(X_PS_ROI[i]), Ybest=torch.tensor(Y_PF_ROI[i]), hvc=hvc, SOM=som)
            self.append_child(node)
            self.clean_child()


def calcHypervolume(refpoint, pareto): 
    model = Hypervolume(-torch.tensor(refpoint))
    return model.compute(-torch.tensor(pareto))

def dominate(a, b): 
    assert len(a) == len(b)
    domin1 = True
    domin2 = False
    for idx in range(len(a)): 
        if a[idx] > b[idx]: 
            domin1 = False
        elif a[idx] < b[idx]: 
            domin2 = True
    return domin1 and domin2

def newParetoSet(paretoParams, paretoValues, newParams, newValue): 
    assert len(paretoParams) == len(paretoValues)
    dupli = False
    removed = set()
    indices = []
    for idx, elem in enumerate(paretoValues): 
        if str(paretoParams[idx]) == str(newParams): 
            dupli = True
            break
        if dominate(newValue, elem): 
            removed.add(idx)
    if dupli: 
        return paretoParams, paretoValues
    for idx, elem in enumerate(paretoValues): 
        if not idx in removed: 
            indices.append(idx)
    newParetoParams = []
    newParetoValues = []
    for index in indices: 
        newParetoParams.append(paretoParams[index])
        newParetoValues.append(paretoValues[index])
    bedominated = False
    for idx, elem in enumerate(newParetoValues): 
        if dominate(elem, newValue): 
            bedominated = True
    if len(removed) > 0:
        assert not bedominated
    if len(removed) > 0 or len(paretoParams) == 0 or not bedominated: 
        newParetoParams.append(newParams)
        newParetoValues.append(newValue)
    return newParetoParams, newParetoValues

def pareto(params, values): 
    paretoParams = []
    paretoValues = []

    for var, objs in zip(params, values): 
        paretoParams, paretoValues = newParetoSet(paretoParams, paretoValues, var, objs)

    return paretoParams, paretoValues

def calcHV(refpoint, initParetoParams, initParetoValues, tmp1, tmp2): 
    newParetoParams, newParetoValues = newParetoSet(initParetoParams, initParetoValues, tmp1, tmp2)
    newHV = calcHypervolume(refpoint, newParetoValues)
    return newHV

def calcHV2(refpoint, newParams, newValues): 
    newParetoParams, newParetoValues = pareto(newParams, newValues)
    newHV = calcHypervolume(refpoint, newParetoValues)
    return newHV

def get_hvc_all(refpoint, trainX, trainY):
    initX = trainX.numpy()
    initY = trainY.numpy()
    initParams = []
    initValues = []
    for idx, param in enumerate(list(initX)): 
        initParams.append(list(param))
        initValues.append(list(initY[idx]))
    initParetoParams, initParetoValues = pareto(initParams, initValues)
    initHV = calcHypervolume(refpoint, initParetoValues)
    procs = []
    pool = mp.Pool(processes=1)
    for idx in range(initX.shape[0]): 
        newParams = []
        newValues = []
        for jdx, param in enumerate(list(initX)): 
            if idx == jdx: 
                continue
            newParams.append(list(param))
            newValues.append(list(initY[jdx]))
        proc = pool.apply_async(calcHV2, (refpoint, newParams, newValues))
        procs.append(proc)
    pool.close()
    pool.join()
    hvc = []
    for idx in range(initX.shape[0]): 
        newHV = procs[idx].get()
        newHVC = initHV - newHV
        hvc.append(newHVC)
    hvc = np.array(hvc)
    return hvc

def sensitivity_matrix(X,Y,lb=0.0,ub=1.0,p_vr=1):
    n = X.shape[1]
    m = Y.shape[1]
    S = np.zeros((n,m))
    c_mean = np.mean(Y, axis=0)
    n_l = 10 ** p_vr
    dom = ub - lb
    step = dom / n_l
    cl = [[[] for l in range(n_l)] for i in range(n)]
    for i_X in range(X.shape[0]):
        for i in range(n):
            if int((X[i_X][i]-lb)/step) == n_l:
                cl[i][int((X[i_X][i]-lb)/step)-1].append(Y[i_X])
            else:
                cl[i][int((X[i_X][i]-lb)/step)].append(Y[i_X])
    
    for j in range(m):
        c_sum_j = np.sum((Y[:,j] - c_mean[j]) ** 2)
        for i in range(n):
            for l in range(n_l):
                if len(cl[i][l]) == 0:
                    S[i][j] += 0
                else:
                    S[i][j] += len(cl[i][l]) * ((np.mean(np.array(cl[i][l]),axis=0)[j]-c_mean[j]) ** 2)
            S[i][j] /= (c_sum_j + 1e-6)
    
    return S

def sim(ROI_i, X, Y, ref_point, pid_set, id_list, next_x, result_p, lock, time_slot, last_time_slot):
    print("Start a simulation. ROI_i:", ROI_i)
    pid = os.getpid()
    if pid in pid_set:
        index = pid_set[pid]
    else:
        pid_set[pid] = None      
        index = list(pid_set.keys()).index(pid)
        pid_set[pid] = index
    idx = id_list[index]
    sim_start_time = time.time()
    np.savetxt(str(idx)+'.input', next_x)
    shell = "" # Here is the command that simulates the candidate design. It is specified by your EDA tools and environment. 
    os.system(shell)
    output = np.loadtxt(str(idx)+'.output')
    sim_time = time.time() - sim_start_time
    next_y = output[:3].reshape(-1)
    X_PS, Y_PF = pareto(X.tolist(), Y.tolist())
    HV = calcHypervolume(ref_point, Y_PF)
    new_HV = calcHV(ref_point, X_PS, Y_PF, next_x.tolist(), next_y.tolist())
    if new_HV > HV:
        hvc = new_HV - HV
        success = True
    else:
        hvc = 0
        success = False
    result_p.put((torch.from_numpy(next_x),torch.from_numpy(next_y),ROI_i,success,hvc))
    print("Simulation is Over. ROI_i:", ROI_i)
    time_slot[index] = time.time() - last_time_slot[index]
    last_time_slot[index] = time.time()
    record = np.hstack((next_x, output.reshape(-1), np.array(sim_time), np.array(time_slot[index])))
    with open("record"+str(idx)+".txt","a+") as fr:
        fr.write(str(record.tolist())+"\n")

def print_err(err):
    print(err)

def RS_in_ROI(ROI_i_list, ROI_i, X, Y, ub_x, ref_point, B, pool_sim, pid_set, id_list, result_p, lock, X_pending, time_slot, last_time_slot):
    ROI = ROI_i_list[ROI_i]
    B = int(B)
    ub_x = ub_x.astype(int)
    num_dims = len(ub_x)
    if ROI.depth == 0:
        new_x = np.random.randint(0,ub_x+1,(B,num_dims))
    else:
        new_x = []
        new_x_str = []
        while len(new_x) < B:
            x_rnd = np.random.randint(0,ub_x+1,(num_dims))
            while not ROI.is_in_ROI(torch.tensor(x_rnd)):
                x_rnd = np.random.randint(0,ub_x+1,(num_dims))
            if str(x_rnd) not in new_x_str:
                new_x.append(x_rnd)
                new_x_str.append(str(x_rnd))
    for x in new_x:
        X_pending.append(x.tolist())
        pool_sim.apply_async(sim,(ROI_i, X, Y, ref_point, pid_set, id_list, x.reshape(-1), result_p, lock, time_slot, last_time_slot),error_callback=print_err)

def BO_in_ROI(ROI_i_list, ROI_i, X, Y, ub_x, ref_point, B, pool_sim, pid_set, id_list, result_p, lock, X_pending, time_slot, last_time_slot):
    ROI = ROI_i_list[ROI_i]
    print("IN ROI:",ROI)
    num_skip = 0
    B = int(B)
    ub_x = ub_x.astype(int)
    num_dims = len(ub_x)
    X_ROI, Y_ROI = ROI.get_XY_in_ROI(X, Y)
    if len(X_pending) == 0:
        X_pending_ROI = torch.tensor([])
    else:
        X_pending_ROI = ROI.get_X_in_ROI(torch.tensor(X_pending))
    X_PS_ROI, Y_PF_ROI = pareto(X_ROI.tolist(), Y_ROI.tolist())
    X_PS_ROI_np = np.array(X_PS_ROI)
    np_X_ROI = X_ROI.numpy()
    np_Y_ROI = Y_ROI.numpy()
    n_obj = np_Y_ROI.shape[1]
    try:
        S = sensitivity_matrix(np_X_ROI/ub_x, np_Y_ROI)
    except:
        S = sensitivity_matrix(X.numpy()/ub_x, Y.numpy())
    Y_min = np.zeros(n_obj)
    Y_dis = np.zeros(n_obj)
    for i in range(n_obj):
        Y_min[i] = np.min(np_Y_ROI[:,i])
        Y_dis[i] = (ROI.Ybest[i] - Y_min[i])/Y_min[i]
    w_S = np.zeros(n_obj)
    for i in range(n_obj):
        w_S[i] = (np.sum(Y_dis) - Y_dis[i])/np.sum(Y_dis)
    scores = np.zeros(num_dims)
    for i in range(n_obj):
        scores = scores + w_S[i] * S[:,i] + Cp * np.sqrt(1/ROI.dim_visits)
    print("original S:",S)
    print("w_S:",w_S)
    print("scores:",scores)
    max_scores = np.max(scores)
    if max_scores == 0:
        S = sensitivity_matrix(X.numpy()/ub_x, Y.numpy())
        scores = np.zeros(num_dims)
        for i in range(n_obj):
            scores = scores + w_S[i] * S[:,i]
    idx_sort = np.argsort(-scores)
    active_dims_idx = []
    i_other_dims = 0
    for i in range(num_dims):
        if i < min_num_variables:
            active_dims_idx.append(idx_sort[i])
        elif i > max_num_variables:
            i_other_dims = i
            break
        elif scores[idx_sort[i]] > max_scores * p_imp:
            active_dims_idx.append(idx_sort[i])
        else:
            i_other_dims = i
            break
    print("active_dims_idx:",active_dims_idx)
    active_dims_idx = np.sort(active_dims_idx)
    all_dims_idx = np.arange(num_dims)
    inactive_dims_idx = np.delete(all_dims_idx, active_dims_idx)
    for i in active_dims_idx:
        ROI.dim_visits[i] += 1
    ipt_X_ROI = X_ROI[:, active_dims_idx]
    torch_X_PS_ROI = torch.tensor(X_PS_ROI)
    ipt_X_PS_ROI = torch_X_PS_ROI[:, active_dims_idx]
    fixed_PS_ROI = torch_X_PS_ROI[:, inactive_dims_idx]
    fixed_PS_ROI = torch.unique(fixed_PS_ROI, dim=0)
    if len(X_pending_ROI) == 0:
        ipt_X_ROI_all = ipt_X_ROI
    else:
        ipt_X_pending_ROI = X_pending_ROI[:, active_dims_idx]
        ipt_X_ROI_all = torch.vstack((ipt_X_ROI, ipt_X_pending_ROI))
    mu = Y_ROI.mean(axis=0)
    sigma = Y_ROI.std(axis=0)
    reference_point = -(ref_point - mu) / sigma
    other_dims_ROI = torch.zeros((len(X_ROI),1))
    dis_PS = np.zeros((len(X_ROI),len(fixed_PS_ROI)))
    for i in range(len(X_ROI)):
        for j in range(len(fixed_PS_ROI)):
            dis_PS[i,j] = np.linalg.norm(X_ROI[i, inactive_dims_idx]/ub_x[inactive_dims_idx]-np.array(fixed_PS_ROI[j])/ub_x[inactive_dims_idx])
        other_dims_ROI[i] = np.argmin(dis_PS[i])
    ipt_X_full_ROI = torch.hstack((ipt_X_ROI, other_dims_ROI))
    if len(X_pending_ROI) == 0:
        ipt_X_full_ROI_all = ipt_X_full_ROI
    else:
        other_dims_pending_ROI = torch.zeros((len(X_pending_ROI),1))
        dis_PS_pending = np.zeros((len(X_pending_ROI),len(fixed_PS_ROI)))
        for i in range(len(X_pending_ROI)):
            for j in range(len(fixed_PS_ROI)):
                dis_PS_pending[i,j] = np.linalg.norm(X_pending_ROI[i, inactive_dims_idx]/ub_x[inactive_dims_idx]-np.array(fixed_PS_ROI[j])/ub_x[inactive_dims_idx])
            other_dims_pending_ROI[i] = np.argmin(dis_PS_pending[i])
        ipt_X_full_pending_ROI = torch.hstack((ipt_X_pending_ROI, other_dims_pending_ROI))
        ipt_X_full_ROI_all = torch.vstack((ipt_X_full_ROI, ipt_X_full_pending_ROI))
    bounds_ROI = ub_x[active_dims_idx]
    bounds_ROI = np.hstack((bounds_ROI, len(fixed_PS_ROI)-1))
    likelihood = GaussianLikelihood(
        noise_prior=GammaPrior(torch.tensor(0.9, **tkwargs), torch.tensor(10.0, **tkwargs)),
        noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
    )
    covar_module = ScaleKernel(
        base_kernel=MaternKernel(nu=2.5),
        outputscale_prior=GammaPrior(torch.tensor(2.0, **tkwargs), torch.tensor(0.15, **tkwargs)),
        outputscale_constraint=GreaterThan(1e-6)
    )
    models = []
    train_Y = -(Y_ROI - mu) / sigma
    for i_y in range(Y_ROI.shape[1]):
        gp_model = SingleTaskGP(
            train_X=ipt_X_full_ROI,
            train_Y=train_Y[:,i_y].unsqueeze(-1),
            covar_module=covar_module,
            input_transform=Normalize(d=ipt_X_full_ROI.shape[-1]),
            likelihood=likelihood,
        )
        models.append(gp_model)
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model=model, likelihood=model.likelihood)
    fit_gpytorch_model(mll)
    sampler = SobolQMCNormalSampler(128)
    acqf = qNoisyExpectedHypervolumeImprovement(
        X_baseline=ipt_X_full_ROI_all,
        model=model,
        ref_point=reference_point,
        sampler=sampler,
    )
    s_cat = bounds_ROI.flatten() + 1
    size_categorical = torch.from_numpy(s_cat)
    afo_config = {
        "n_initial_candts": 2000,
        "n_restarts": 5,
        "n_categorical": len(active_dims_idx)+1,
        "category_size": size_categorical,
        "add_spray_points": True,
        "num_ls_steps": 50,
        "som": ROI.SOM_list[-1],
        "PF_area": ROI.ROI_list_list[-1],
        "best_X": torch.tensor(X_PS_ROI),
        "active_dims_idx": active_dims_idx,
        "ub_x": ub_x,
        "fixed_PS_ROI": fixed_PS_ROI
    }
    other_dims_PS = torch.zeros((len(torch_X_PS_ROI),1))
    dis_PS_ = np.zeros((len(torch_X_PS_ROI),len(fixed_PS_ROI)))
    for i in range(len(torch_X_PS_ROI)):
        for j in range(len(fixed_PS_ROI)):
            dis_PS_[i,j] = np.linalg.norm(torch_X_PS_ROI[i, inactive_dims_idx]/ub_x[inactive_dims_idx]-np.array(fixed_PS_ROI[j])/ub_x[inactive_dims_idx])
        other_dims_PS[i] = np.argmin(dis_PS_[i])
    ipt_X_PS_full_ROI = torch.hstack((ipt_X_PS_ROI,other_dims_PS.reshape(-1,1)))
    pareto_points = ipt_X_PS_full_ROI.clone()
    opt_start_time = time.time()
    with warnings.catch_warnings(): 
        warnings.filterwarnings("ignore", category=NumericalWarning)
        next_x, acq_val, flag_skip = optimize_acqf_categorical_local_search(acqf, afo_config=afo_config, pareto_points=pareto_points, q=B)
    print("Time to get the candidate:",time.time() - opt_start_time)
    if flag_skip:
        num_skip += B
        return num_skip
    next_x_list = next_x.reshape(B,-1)
    for i in range(B):    
        next_x = next_x_list[i].numpy().reshape(-1)
        for j in range(num_dims):
            if next_x[j] < 0:
                next_x[j] = 0
            elif next_x[j] > ub_x[j]:
                next_x[j] = ub_x[j]
        if (next_x.tolist() in X.tolist()) or (next_x.tolist() in X_pending):
            num_skip += 1
        else:
            X_pending.append(next_x.tolist())
            pool_sim.apply_async(sim,(ROI_i, X, Y, ref_point, pid_set, id_list, next_x, result_p, lock, time_slot, last_time_slot),error_callback=print_err)
    return num_skip

all_start_time = time.time()
device = torch.device("cpu")
tkwargs = {"dtype": torch.double, "device": device}
print("Read Init Data")
dataset = np.loadtxt("initial_dataset.txt")
init_X = dataset[:,:num_dims]
init_Y = dataset[:,num_dims:num_dims+num_objectives]
X = torch.from_numpy(init_X[:n_initial_points])
Y = torch.from_numpy(init_Y[:n_initial_points])
ref_point = torch.tensor([torch.max(Y[:, 0]), torch.max(Y[:, 1]), torch.max(Y[:, 2])])

som = MiniSom(N_SOM, N_SOM, num_dims)
som.train_random(init_X, 10000)

pool_sim = mp.Pool(processes=batch_size)
manager = mp.Manager()
pid_set = manager.dict()
time_slot = manager.list()
last_time_slot = manager.list()
lock = manager.Lock()
result_p = manager.Queue()
X_pending = []
ROI_i_list = [[]] * batch_size
real_batch_size = 0

start_loop_time = time.time()
for i in range(batch_size):
    last_time_slot.append(start_loop_time)
    time_slot.append(start_loop_time - all_start_time)
X_PS, Y_PF = pareto(X.tolist(), Y.tolist())
HV = calcHypervolume(ref_point, Y_PF)
root = SOMT_Node()
for i in range(len(X_PS)):
    Y_PF_removed = Y_PF.copy()
    Y_PF_removed.remove(Y_PF[i])
    if len(Y_PF_removed) == 0:
        HV_removed = 0
    else:
        HV_removed = calcHypervolume(ref_point, Y_PF_removed)
    hvc = HV - HV_removed
    print("Y_PF[",i,"]:",Y_PF[i],", HVC:",hvc)
    node = SOMT_Node(parent=root, Xbest=torch.tensor(X_PS[i]), Ybest=torch.tensor(Y_PF[i]), hvc=hvc, SOM=som)
    root.append_child(node)
    root.clean_child()
ROIs = root.child_list
HVC = []
for ROI in ROIs:
    HVC.append(ROI.hvc)
HVC = np.array(HVC)
print("HVC list:",HVC)
i_HVC_sort = np.argsort(-HVC)
n_ROIs = len(HVC.reshape(-1))
if n_ROIs >= batch_size:
    for i in range(batch_size):
        idx = i_HVC_sort[i]
        ROI_i_list[i] = ROIs[idx]
        real_batch_size += BO_in_ROI(ROI_i_list, i, X, Y, ub_x, ref_point, 1, pool_sim, pid_set, id_list, result_p, lock, X_pending, time_slot, last_time_slot)
else:
    B_ROIs = np.zeros(n_ROIs, dtype=int)
    B_ROIs += int(batch_size / n_ROIs)
    if batch_size % n_ROIs > 0:
        B_ROIs[:batch_size % n_ROIs] += 1
    for i in range(n_ROIs):
        idx = i_HVC_sort[i]
        ROI_i_list[i] = ROIs[idx]
        print("OUT ROI:",ROI)
        real_batch_size += BO_in_ROI(ROI_i_list, i, X, Y, ub_x, ref_point, B_ROIs[i], pool_sim, pid_set, id_list, result_p, lock, X_pending, time_slot, last_time_slot)
i_HVC = 0
while real_batch_size != 0:
    i_HVC = i_HVC % n_ROIs
    idx = i_HVC_sort[i_HVC]
    ROI_i_list[i] = ROIs[idx]
    old_real_batch_size = real_batch_size
    real_batch_size += BO_in_ROI(ROI_i_list, i, X, Y, ub_x, ref_point, old_real_batch_size, pool_sim, pid_set, id_list, result_p, lock, X_pending, time_slot, last_time_slot)
    real_batch_size -= old_real_batch_size
    i_HVC += 1

restart_depth = MAX_DEPTH
while True:
    if time.time() - all_start_time >= 86400.0:
        pool_sim.close()
        pool_sim.join()
        while not result_p.empty():
            new_X, new_Y, ROI_i, success, new_hvc = result_p.get()
            X_pending.remove(new_X.tolist())
            X = torch.vstack((X, new_X))
            Y = torch.vstack((Y, new_Y))
            X_PS, Y_PF = pareto(X.tolist(), Y.tolist())
            HV = calcHypervolume(ref_point, Y_PF)
            print("new_X:",new_X)
            print("new_Y:",new_Y)
            print("HV " ,len(X) ,": " ,HV)
        break

    while result_p.empty() and real_batch_size == 0:
        pass

    while not result_p.empty():
        print("len(X):",len(X))
        if time.time() - all_start_time >= 86400.0:
            pool_sim.close()
            pool_sim.join()
            while not result_p.empty():
                new_X, new_Y, ROI_i, success, new_hvc = result_p.get()
                X_pending.remove(new_X.tolist())
                X = torch.vstack((X, new_X))
                Y = torch.vstack((Y, new_Y))
                X_PS, Y_PF = pareto(X.tolist(), Y.tolist())
                HV = calcHypervolume(ref_point, Y_PF)
                print("new_X:",new_X)
                print("new_Y:",new_Y)
                print("HV " ,len(X) ,": " ,HV)
            break
        new_X, new_Y, ROI_i, success, new_hvc = result_p.get()
        ROI = ROI_i_list[ROI_i]
        print("Get ROI:",ROI)
        if success:
            print("New result. Success.")
            ROI.rs_counter = 0
            ROI.success_counter += 1
            ROI.failure_counter = 0
        else:
            print("New result. Fail.")
            ROI.success_counter = 0
            ROI.failure_counter += 1
        X_pending.remove(new_X.tolist())
        X = torch.vstack((X, new_X))
        Y = torch.vstack((Y, new_Y))
        X_PS, Y_PF = pareto(X.tolist(), Y.tolist())
        HV = calcHypervolume(ref_point, Y_PF)
        print("new_X:",new_X)
        print("new_Y:",new_Y)
        print("new_hvc:",new_hvc)
        print("HV " ,len(X) ,": " ,HV)
        real_batch_size += 1
        if ROI is root:
            if success:
                if ROI.child_list == []:
                    som = MiniSom(N_SOM, N_SOM, len(X[0]))
                    som.train_random(X.numpy(), 10000)
                    X_PS, Y_PF = pareto(X.tolist(), Y.tolist())
                    HV = calcHypervolume(ref_point, Y_PF)
                    for i in range(len(X_PS)):
                        Y_PF_removed = Y_PF.copy()
                        Y_PF_removed.remove(Y_PF[i])
                        if len(Y_PF_removed) == 0:
                            HV_removed = 0
                        else:
                            HV_removed = calcHypervolume(ref_point, Y_PF_removed)
                        hvc = HV - HV_removed
                        node = SOMT_Node(parent=ROI, Xbest=torch.tensor(X_PS[i]), Ybest=torch.tesnor(Y_PF[i]), hvc=hvc, SOM=som)
                        ROI.append_child(node)
                        ROI.clean_child()
                else:
                    node = SOMT_Node(parent=ROI, Xbest=new_X, Ybest=new_Y, hvc=new_hvc, SOM=ROI.child_list[0].SOM)
                    ROI.append_child(node)
                    ROI.clean_child()
                ROI_i_list[ROI_i] = ROI.child_list[0]
                ROI = ROI_i_list[ROI_i]
            else:
                if real_batch_size < batch_size:
                    continue
                RS_in_ROI(ROI_i_list, ROI_i, X, Y, ub_x, ref_point, real_batch_size, pool_sim, pid_set, id_list, result_p, lock, X_pending, time_slot, last_time_slot)
                real_batch_size -= real_batch_size
                break
        ROI.update(X, Y, ref_point)
        parent = ROI.parent
        ROIs = parent.child_list
        succ_count = []
        fail_count = []
        hvc_list = []
        for ROI_ in ROIs:
            succ_count.append(ROI_.success_counter)
            fail_count.append(ROI_.failure_counter)
            hvc_list.append(ROI_.hvc)
        print("succ_count:",succ_count)
        print("fail_count:",fail_count)
        print("hvc_list:",hvc_list)
        if success:
            restart_depth = MAX_DEPTH
            fom = np.array(succ_count) - np.array(fail_count) + np.array(hvc_list)/max(hvc_list)
            print("fom:",fom)
            ROI_i_list[ROI_i] = ROIs[np.argmax(fom)]
            ROI = ROI_i_list[ROI_i]
            old_real_batch_size = real_batch_size
            real_batch_size += BO_in_ROI(ROI_i_list, ROI_i, X, Y, ub_x, ref_point, old_real_batch_size, pool_sim, pid_set, id_list, result_p, lock, X_pending, time_slot, last_time_slot)
            real_batch_size -= old_real_batch_size
            if real_batch_size == old_real_batch_size:
                ROI.success_counter = 0
                ROI.failure_counter += 1
            continue
        if (np.array(fail_count) > N_FAIL).all():
            ROI_i_list[ROI_i] = ROIs[np.argmax(hvc_list)]
            ROI = ROI_i_list[ROI_i]
            if real_batch_size < batch_size:
                continue
            if ROI.rs_counter != 0 or ROI.depth == restart_depth:
                if ROI.rs_counter == 0:
                    ROI_i_list[ROI_i] = ROI.parent
                    ROI = ROI_i_list[ROI_i]
                    ROI.child_list = []
                ROI.rs_counter += 1
                if ROI.rs_counter > N_RS:
                    if restart_depth > 0:
                        ROI_i_list[ROI_i] = ROI.parent
                        ROI = ROI_i_list[ROI_i]
                        ROI.child_list = []
                        restart_depth -= 1
                RS_in_ROI(ROI_i_list, ROI_i, X, Y, ub_x, ref_point, real_batch_size, pool_sim, pid_set, id_list, result_p, lock, X_pending, time_slot, last_time_slot)
                real_batch_size -= real_batch_size
                break
            X_ROI, Y_ROI = ROI.get_XY_in_ROI(X, Y)
            if len(X_ROI) >= n_initial_points:
                print("Split!")
                ROI.split(X,Y,ref_point)
                parent = ROI
                ROIs = parent.child_list
                HVC = []
                for ROI in ROIs:
                    HVC.append(ROI.hvc)
                print("HVC after spilt:",HVC)
                HVC = np.array(HVC)
                i_HVC_sort = np.argsort(-HVC)
                n_ROIs = len(HVC.reshape(-1))
                if n_ROIs >= batch_size:
                    for i in range(batch_size):
                        idx = i_HVC_sort[i]
                        ROI_i_list[ROI_i] = ROIs[idx]
                        ROI = ROI_i_list[ROI_i]
                        real_batch_size += BO_in_ROI(ROI_i_list, ROI_i, X, Y, ub_x, ref_point, 1, pool_sim, pid_set, id_list, result_p, lock, X_pending, time_slot, last_time_slot)
                        real_batch_size -= 1
                        if real_batch_size == old_real_batch_size:
                            ROI.success_counter = 0
                            ROI.failure_counter += 1
                else:
                    B_ROIs = np.zeros(n_ROIs, dtype=int)
                    B_ROIs += int(batch_size / n_ROIs)
                    if batch_size % n_ROIs > 0:
                        B_ROIs[:batch_size % n_ROIs] += 1
                    for i in range(n_ROIs):
                        idx = i_HVC_sort[i]
                        ROI_i_list[ROI_i] = ROIs[idx]
                        ROI = ROI_i_list[ROI_i]
                        real_batch_size += BO_in_ROI(ROI_i_list, ROI_i, X, Y, ub_x, ref_point, B_ROIs[i], pool_sim, pid_set, id_list, result_p, lock, X_pending, time_slot, last_time_slot)
                        real_batch_size -= B_ROIs[i]
                        if real_batch_size == old_real_batch_size:
                            ROI.success_counter = 0
                            ROI.failure_counter += 1
            else:
                fom = np.array(succ_count) - np.array(fail_count) + np.array(hvc_list)/max(hvc_list)
                print("fom:",fom)
                ROI_i_list[ROI_i] = ROIs[np.argmax(fom)]
                ROI = ROI_i_list[ROI_i]
                old_real_batch_size = real_batch_size
                real_batch_size += BO_in_ROI(ROI_i_list, ROI_i, X, Y, ub_x, ref_point, old_real_batch_size, pool_sim, pid_set, id_list, result_p, lock, X_pending, time_slot, last_time_slot)
                real_batch_size -= old_real_batch_size
                if real_batch_size == old_real_batch_size:
                    ROI.success_counter = 0
                    ROI.failure_counter += 1
        else:
            fom = np.array(succ_count) - np.array(fail_count) + np.array(hvc_list)/max(hvc_list)
            print("fom:",fom)
            ROI_i_list[ROI_i] = ROIs[np.argmax(fom)]
            ROI = ROI_i_list[ROI_i]
            old_real_batch_size = real_batch_size
            real_batch_size += BO_in_ROI(ROI_i_list, ROI_i, X, Y, ub_x, ref_point, old_real_batch_size, pool_sim, pid_set, id_list, result_p, lock, X_pending, time_slot, last_time_slot)
            real_batch_size -= old_real_batch_size
            if real_batch_size == old_real_batch_size:
                ROI.success_counter = 0
                ROI.failure_counter += 1

    if real_batch_size > 0:
        parent = ROI.parent
        ROIs = parent.child_list
        succ_count = []
        fail_count = []
        hvc_list = []
        for ROI_ in ROIs:
            succ_count.append(ROI_.success_counter)
            fail_count.append(ROI_.failure_counter)
            hvc_list.append(ROI_.hvc)
        if (np.array(fail_count) > N_FAIL).all():
            ROI = ROIs[np.argmax(hvc_list)]
            if real_batch_size < batch_size:
                continue
            if ROI.rs_counter != 0 or ROI.depth == restart_depth:
                if ROI.rs_counter == 0:
                    ROI = ROI.parent
                    ROI.child_list = []
                ROI.rs_counter += 1
                if ROI.rs_counter > N_RS:
                    if restart_depth > 0:
                        ROI = ROI.parent
                        ROI.child_list = []
                        restart_depth -= 1
                RS_in_ROI(ROI_i_list, ROI_i, X, Y, ub_x, ref_point, real_batch_size, pool_sim, pid_set, id_list, result_p, lock, X_pending, time_slot, last_time_slot)
                real_batch_size -= real_batch_size
                break
            X_ROI, Y_ROI = ROI.get_XY_in_ROI(X, Y)
            if len(X_ROI) >= n_initial_points:
                print("Split!")
                ROI.split(X,Y,ref_point)
                parent = ROI
                ROIs = parent.child_list
                HVC = []
                for ROI in ROIs:
                    HVC.append(ROI.hvc)
                print("HVC after spilt:",HVC)
                HVC = np.array(HVC)
                i_HVC_sort = np.argsort(-HVC)
                n_ROIs = len(HVC.reshape(-1))
                if n_ROIs >= batch_size:
                    for i in range(batch_size):
                        idx = i_HVC_sort[i]
                        ROI = ROIs[idx]
                        real_batch_size += BO_in_ROI(ROI, X, Y, ub_x, ref_point, 1, pool_sim, pid_set, id_list, result_p, lock, X_pending, time_slot, last_time_slot)
                        real_batch_size -= 1
                        if real_batch_size == old_real_batch_size:
                            ROI.success_counter = 0
                            ROI.failure_counter += 1
                else:
                    B_ROIs = np.zeros(n_ROIs, dtype=int)
                    B_ROIs += int(batch_size / n_ROIs)
                    if batch_size % n_ROIs > 0:
                        B_ROIs[:batch_size % n_ROIs] += 1
                    for i in range(n_ROIs):
                        idx = i_HVC_sort[i]
                        ROI = ROIs[idx]
                        real_batch_size += BO_in_ROI(ROI, X, Y, ub_x, ref_point, B_ROIs[i], pool_sim, pid_set, id_list, result_p, lock, X_pending, time_slot, last_time_slot)
                        real_batch_size -= B_ROIs[i]
                        if real_batch_size == old_real_batch_size:
                            ROI.success_counter = 0
                            ROI.failure_counter += 1
            else:
                fom = np.array(succ_count) - np.array(fail_count) + np.array(hvc_list)/max(hvc_list)
                print("fom:",fom)
                ROI_i_list[ROI_i] = ROIs[np.argmax(fom)]
                ROI = ROI_i_list[ROI_i]
                old_real_batch_size = real_batch_size
                real_batch_size += BO_in_ROI(ROI_i_list, ROI_i, X, Y, ub_x, ref_point, old_real_batch_size, pool_sim, pid_set, id_list, result_p, lock, X_pending, time_slot, last_time_slot)
                real_batch_size -= old_real_batch_size
                if real_batch_size == old_real_batch_size:
                    ROI.success_counter = 0
                    ROI.failure_counter += 1
        else:
            fom = np.array(succ_count) - np.array(fail_count) + np.array(hvc_list)/max(hvc_list)
            print("fom:",fom)
            ROI_i_list[ROI_i] = ROIs[np.argmax(fom)]
            ROI = ROI_i_list[ROI_i]
            old_real_batch_size = real_batch_size
            real_batch_size += BO_in_ROI(ROI_i_list, ROI_i, X, Y, ub_x, ref_point, old_real_batch_size, pool_sim, pid_set, id_list, result_p, lock, X_pending, time_slot, last_time_slot)
            real_batch_size -= old_real_batch_size
            if real_batch_size == old_real_batch_size:
                ROI.success_counter = 0
                ROI.failure_counter += 1

print("Total time:", time.time() - all_start_time)
