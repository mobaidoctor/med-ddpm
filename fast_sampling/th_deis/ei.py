import jax
import jax.numpy as jnp

def get_integrator_basis_fn(sde):
    def _worker(t_start, t_end, num_item):
        dt = (t_end - t_start) / num_item

        t_inter = jnp.linspace(t_start, t_end, num_item, endpoint=False)
        psi_coef = sde.psi(t_inter, t_end)
        integrand = sde.eps_integrand(t_inter)

        return psi_coef * integrand, t_inter, dt
    return _worker


def single_poly_coef(t_val, ts_poly, coef_idx=0):
    num = t_val - ts_poly
    denum = ts_poly[coef_idx] - ts_poly
    num = num.at[coef_idx].set(1.0)
    denum = denum.at[coef_idx].set(1.0)
    return jnp.prod(num) / jnp.prod(denum)

vec_poly_coef = jax.vmap(single_poly_coef, (0, None, None), 0)


def get_one_coef_per_step_fn(sde):
    _eps_coef_worker_fn = get_integrator_basis_fn(sde)
    def _worker(t_start, t_end, ts_poly, coef_idx=0,num_item=10000):
        integrand, t_inter, dt = _eps_coef_worker_fn(t_start, t_end, num_item)
        poly_coef = vec_poly_coef(t_inter, ts_poly, coef_idx)
        return jnp.sum(integrand * poly_coef) * dt
    return _worker

def get_coef_per_step_fn(sde, highest_order, order):
    eps_coef_fn = get_one_coef_per_step_fn(sde)
    def _worker(t_start, t_end, ts_poly, num_item=10000):
        rtn = jnp.zeros((highest_order+1, ), dtype=float)
        ts_poly = ts_poly[:order+1]
        coef = jax.vmap(eps_coef_fn, (None, None, None, 0, None))(t_start, t_end, ts_poly, jnp.flip(jnp.arange(order+1)), num_item)
        rtn = rtn.at[:order+1].set(coef)
        return rtn
    return _worker

def get_ab_eps_coef_order0(sde, highest_order, timesteps):
    _worker = get_coef_per_step_fn(sde, highest_order, 0)
    col_idx = jnp.arange(len(timesteps)-1)[:,None]
    idx = col_idx + jnp.arange(1)[None, :]
    vec_ts_poly = timesteps[idx]
    return jax.vmap(
        _worker,
        (0, 0, 0), 0
    )(timesteps[:-1], timesteps[1:], vec_ts_poly)

def get_ab_eps_coef(sde, highest_order, timesteps, order):
    if order == 0:
        return get_ab_eps_coef_order0(sde, highest_order, timesteps)
    
    prev_coef = get_ab_eps_coef(sde, highest_order, timesteps[:order+1], order=order-1)

    cur_coef_worker = get_coef_per_step_fn(sde, highest_order, order)

    col_idx = jnp.arange(len(timesteps)-order-1)[:,None]
    idx = col_idx + jnp.arange(order+1)[None, :]
    vec_ts_poly = timesteps[idx]
    

    cur_coef = jax.vmap(
        cur_coef_worker,
        (0, 0, 0), 0
    )(timesteps[order:-1], timesteps[order+1:], vec_ts_poly) #[3, 4, (0,1,2,3)]

    return jnp.concatenate(
        [
            prev_coef,
            cur_coef
        ],
        axis=0
    )
