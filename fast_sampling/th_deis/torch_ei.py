import numpy as np
import torch

import jax.numpy as jnp

to_jax = lambda item: jnp.asarray(item.cpu().numpy(), dtype=float)
import ei

def check_results(res, orig_fun, *args, **kwargs):
    j_args = []
    for a in args:
        j_args.append(to_jax(a))
    j_kwargs = []
    for k in kwargs:
        j_kwargs[k] = to_jax(kwargs[k])
    j_res = orig_fun(*j_args, **j_kwargs)
    if (not isinstance(res, list)) or (not isinstance(res, tuple)):
        res = [res]
        j_res = [j_res]
    for idx in range(len(res)):
        res_, j_res_ = res[idx], j_res[idx]
        np_res = res_.cpu().numpy()
        np_j_res = np.array(j_res_)
        assert (np.fabs(np_res - np_j_res) > 1e-6).any()






def get_integrator_basis_fn(sde):
    def _worker(t_start, t_end, num_item):
        dt = (t_end - t_start) / num_item
        t_inter = torch.FloatTensor(np.linspace(t_start, t_end, num_item, endpoint=False))
        psi_coef = sde.psi(t_inter, t_end)
        integrand = sde.eps_integrand(t_inter)
        return psi_coef * integrand, t_inter, dt

    return _worker


def single_poly_coef(t_val, ts_poly, coef_idx=0):
    num = t_val - ts_poly
    denum = ts_poly[:, coef_idx] - ts_poly
    num[:, coef_idx] = 1.0
    denum[:, coef_idx] = 1.0
    return torch.prod(num) / torch.prod(denum)



#vec_poly_coef = torch.vmap(single_poly_coef, (0, None, None), 0)
def vec_poly_coef(t_val, ts_poly, coef_idx=0):
    return single_poly_coef(t_val, ts_poly, coef_idx)




def get_one_coef_per_step_fn(sde):
    _eps_coef_worker_fn = get_integrator_basis_fn(sde)
    def _worker(t_start, t_end, ts_poly, coef_idx=0,num_item=10000):
        integrand, t_inter, dt = _eps_coef_worker_fn(t_start, t_end, num_item)
        poly_coef = vec_poly_coef(t_inter, ts_poly, coef_idx)
        return torch.sum(integrand * poly_coef) * dt
    return _worker

def get_coef_per_step_fn(sde, highest_order, order):
    eps_coef_fn = get_one_coef_per_step_fn(sde)
    def _worker(t_start, t_end, ts_poly, num_item=10000):
        rtn = torch.zeros((highest_order+1, ))
        ts_poly = ts_poly[:order+1]
        # coef = torch.vmap(eps_coef_fn, (None, None, None, 0, None))(t_start, t_end, ts_poly, torch.flip(torch.arange(order+1)), num_item)
        coef = eps_coef_fn(t_start, t_end, ts_poly, torch.flip(torch.arange(order+1), [0]), num_item)
        #rtn = rtn.at[:order+1].set(coef)
        rtn[:order + 1] = coef
        return rtn
    return _worker

def get_ab_eps_coef_order0(sde, highest_order, timesteps):
    _worker = get_coef_per_step_fn(sde, highest_order, 0)
    col_idx = torch.arange(len(timesteps)-1)[:,None]
    idx = col_idx + torch.arange(1)[None, :]
    vec_ts_poly = timesteps[idx]
    # return torch.vmap(
    #     _worker,
    #     (0, 0, 0), 0
    # )(timesteps[:-1], timesteps[1:], vec_ts_poly)
    result = _worker(timesteps[:-1], timesteps[1:], vec_ts_poly)
    return result



def get_ab_eps_coef(sde, highest_order, timesteps, order):
    if order == 0:
        return get_ab_eps_coef_order0(sde, highest_order, timesteps)
    
    prev_coef = get_ab_eps_coef(sde, highest_order, timesteps[:order+1], order=order-1)

    cur_coef_worker = get_coef_per_step_fn(sde, highest_order, order)

    col_idx = torch.arange(len(timesteps)-order-1)[:,None]
    idx = col_idx + torch.arange(order+1)[None, :]
    vec_ts_poly = timesteps[idx]
    

    # cur_coef = torch.vmap(
    #     cur_coef_worker,
    #     (0, 0, 0), 0
    # )(timesteps[order:-1], timesteps[order+1:], vec_ts_poly) #[3, 4, (0,1,2,3)]

    cur_coef = cur_coef_worker(timesteps[order:-1], timesteps[order+1:], vec_ts_poly)

    return torch.cat([prev_coef, cur_coef], axis=0)