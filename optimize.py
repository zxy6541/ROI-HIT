import os
import numpy as np
import torch
from torch import Tensor
from minisom import MiniSom

def get_full_var(x, active_dims_idx, ub_x, fixed_PS_ROI):
    inactive_dims_idx = np.delete(np.arange(len(ub_x)),active_dims_idx)
    x_tmp = torch.zeros((x.shape[0],len(ub_x)))
    for i in range(x.shape[0]):
        x_tmp[i, inactive_dims_idx] = fixed_PS_ROI[int(x[i,-1].item())]
        x_tmp[i, active_dims_idx] = x[i, :-1]
    return x_tmp

def is_x_valid(x, som, PF_area, fixed_PS_ROI, active_dims_idx, ub_x):
    inactive_dims_idx = np.delete(np.arange(len(ub_x)),active_dims_idx)
    x_tmp = torch.zeros(len(ub_x))
    x_tmp[inactive_dims_idx] = fixed_PS_ROI[int(x[-1].item())]
    x_tmp[active_dims_idx] = x[:-1]
    x_correxted = x_tmp
    if som.winner(x_correxted) in PF_area:
        return True
    else:
        return False


def get_catg_neighbors(x_discrete, n_categories, som, PF_area,  active_dims_idx, ub_x, fixed_PS_ROI, **tkwargs):
    X_loc = []
    for pt_idx in range(x_discrete.shape[0]):
        for i in range(x_discrete.shape[1]):
            for j in range(n_categories[i].item()):
                if x_discrete[pt_idx][i] == j:
                    continue
                temp_x = x_discrete[pt_idx].clone()
                temp_x[i] = j
                if not is_x_valid(temp_x, som, PF_area, fixed_PS_ROI, active_dims_idx, ub_x):
                    continue
                X_loc.append(temp_x)
    if len(X_loc) == 0:
        return None
    else:
        return torch.cat([x.unsqueeze(0) for x in X_loc], dim=0).to(**tkwargs)


def optimize_acqf_categorical_local_search(acqf, afo_config, pareto_points: torch.Tensor, q: int = 1):
    candidate_list = []
    base_X_pending = acqf.X_pending if q > 1 else None

    tkwargs = {"device": pareto_points.device, "dtype": pareto_points.dtype}
    n_initial_candts = afo_config["n_initial_candts"]  # 2000
    n_restarts = afo_config["n_restarts"]  # 5
    input_dim = afo_config["n_categorical"]
    n_categories = afo_config["category_size"] # afo_config["n_categories"]
    som = afo_config["som"]
    PF_area = afo_config["PF_area"]
    best_X = afo_config["best_X"]
    active_dims_idx = afo_config["active_dims_idx"]
    ub_x = afo_config["ub_x"]
    fixed_pf = list(range(len(pareto_points)))
    other_dims_pf = pareto_points[:,-1].tolist()
    d_pf = dict(zip(other_dims_pf, fixed_pf))
    fixed_PS_ROI = afo_config["fixed_PS_ROI"]

    for _ in range(q):
        x_init_candts = None
        for i in range(n_initial_candts):
            x_init_tmp = torch.zeros(input_dim,**tkwargs)
            for j in range(input_dim):
                x_init_tmp[j] = torch.randint(n_categories[j].item(), (1,))
            if is_x_valid(x_init_tmp, som, PF_area, fixed_PS_ROI, active_dims_idx, ub_x):
                if x_init_candts is None:
                    x_init_candts = x_init_tmp.unsqueeze(0)
                else:
                    x_init_candts = torch.vstack((x_init_candts,x_init_tmp))
        perturb_nbors = None
        for x in pareto_points:
            nbds = get_catg_neighbors(x.unsqueeze(0), n_categories, som, PF_area, active_dims_idx, ub_x, fixed_PS_ROI, **tkwargs)
            if nbds is None:
                continue
            # print(x, nbds)
            # print(f'nbds {nbds.shape}')
            if perturb_nbors is None:
                perturb_nbors = nbds
            else:
                # print(f'nbds {perturb_nbors.shape}') 
                perturb_nbors = torch.cat([perturb_nbors, nbds], axis=0)
        
        if perturb_nbors is None:
            if x_init_candts is None:
                return None, None, True
            else:
                pass
        else:
            if x_init_candts is None:
                x_init_candts = perturb_nbors
            else:
                x_init_candts = torch.cat([x_init_candts, perturb_nbors], axis=0)

        if x_init_candts is None:
            return None, None, True

        #print("rand init len:",len(x_init_candts))
        with torch.no_grad():
            acq_init_candts = torch.cat([acqf(X_.unsqueeze(1)) for X_ in x_init_candts.split(16)])
        # print(f"x_init_candts {x_init_candts.shape}")
        topk_indices = torch.topk(acq_init_candts, n_restarts)[1]
        #print("topk",len(topk_indices))
        # print(f"topk_indices {topk_indices}")
        best_X_ = x_init_candts[topk_indices]
        best_acq_val = acq_init_candts[topk_indices]
        for i in range(n_restarts):
            num_ls_steps = afo_config["num_ls_steps"]  # number of local search steps
            for _ in range(num_ls_steps):
                # print(f'best_X[i] {best_X[i].shape} {best_X[i].dtype}')
                nbds = get_catg_neighbors(best_X_[i].unsqueeze(0), n_categories, som, PF_area, active_dims_idx, ub_x, fixed_PS_ROI, **tkwargs)
                if nbds is None:
                    break
                with torch.no_grad():
                    acq_vals = acqf(nbds.unsqueeze(1))
                if torch.max(acq_vals) > best_acq_val[i]:
                    best_acq_val[i] = torch.max(acq_vals)
                    best_X_[i] = nbds[torch.argmax(acq_vals)]
                else:
                    break
        candidate_list.append(best_X_[torch.argmax(best_acq_val)].unsqueeze(0))

        # set pending points
        candidates = torch.cat(candidate_list, dim=-2)
        if q > 1:
            acqf.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2) if base_X_pending is not None else candidates
            )

    if q > 1:
        acqf.set_X_pending(base_X_pending)
    with torch.no_grad():
        acq_value = acqf(candidates.unsqueeze(1)) # compute joint acquisition value
    candidates = get_full_var(candidates.reshape(q,-1), active_dims_idx, ub_x, fixed_PS_ROI)
    return candidates, acq_value, False
