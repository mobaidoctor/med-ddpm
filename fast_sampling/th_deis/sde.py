import jax
import jax.numpy as jnp
import numpy as np
import torch as th
from jax._src.numpy.lax_numpy import _promote_dtypes_inexact

from .ei import get_ab_eps_coef


def get_interp_fn(_xp, _fp):
  @jax.jit
  def _fn(x):
      if jnp.shape(_xp) != jnp.shape(_fp) or jnp.ndim(_xp) != 1:
          raise ValueError("xp and fp must be one-dimensional arrays of equal size")
      x, xp, fp = _promote_dtypes_inexact(x, _xp, _fp)

      i = jnp.clip(jnp.searchsorted(xp, x, side='right'), 1, len(xp) - 1)
      df = fp[i] - fp[i - 1]
      dx = xp[i] - xp[i - 1]
      delta = x - xp[i - 1]
      f = jnp.where((dx == 0), fp[i], fp[i - 1] + (delta / dx) * df)
      return f
  return _fn

class DisVPSDE:
    def __init__(self, discrete_alpha):
        to_jax = lambda item: jnp.asarray(item.cpu().numpy(), dtype=float)
        if isinstance(discrete_alpha, th.Tensor):
            j_alphas = to_jax(discrete_alpha)
        else:
            j_alphas = jnp.asarray(discrete_alpha)
        j_times = jnp.asarray(
            jnp.arange(len(discrete_alpha)), dtype=float
        )
        # use a piecewise linear function to fit alpha
        _alpha_fn = get_interp_fn(j_times, j_alphas)
        self.alpha_fn = lambda item: jnp.clip(
            _alpha_fn(item), 1e-7, 1.0 - 1e-7
        )
        
        self.t_start = 0
        self.t_end = len(discrete_alpha) - 1

        log_alpha_fn = lambda t: jnp.log(self.alpha_fn(t))
        grad_log_alpha_fn = jax.grad(log_alpha_fn)
        self.d_log_alpha_dtau_fn = jax.vmap(grad_log_alpha_fn)

    def psi(self, t_start, t_end):
        return jnp.sqrt(self.alpha_fn(t_end) / self.alpha_fn(t_start))

    def eps_integrand(self, vec_t):
        d_log_alpha_dtau = self.d_log_alpha_dtau_fn(vec_t)
        integrand = -0.5 * d_log_alpha_dtau / jnp.sqrt(1 - self.alpha_fn(vec_t))
        return integrand

    def get_deis_coef(self, order, rev_timesteps, highest_order=3):
        # return [x_coef, eps_coef]
        rev_timesteps = jnp.asarray(rev_timesteps)
        x_coef = self.psi(rev_timesteps[:-1], rev_timesteps[1:])
        eps_coef = get_ab_eps_coef(self, highest_order, rev_timesteps, order)
        return np.asarray(
            jnp.concatenate([x_coef[:, None], eps_coef], axis=1)
        ).copy()

    def get_ipndm_coef(self, rev_timesteps):
        # return [x_coef, eps_coef]
        rev_timesteps = jnp.asarray(rev_timesteps)  # (n+1, )
        x_coef = self.psi(rev_timesteps[:-1], rev_timesteps[1:]) #(n, )

        def get_linear_ab_coef(i):
            if i == 0:
                return jnp.asarray([1.0, 0, 0, 0]).reshape(-1,4)
            prev_coef = get_linear_ab_coef(i-1)
            cur_coef = None
            if i == 1:
                cur_coef = jnp.asarray([1.5, -0.5, 0, 0])
            elif i == 2:
                cur_coef = jnp.asarray([23, -16, 5, 0]) / 12.0
            else:
                cur_coef = jnp.asarray([55, -59, 37, -9]) / 24.0
            return jnp.concatenate(
                [prev_coef, cur_coef.reshape(-1,4)]
            )
        linear_ab_coef = get_linear_ab_coef(len(rev_timesteps) - 2) # (n, 4)

        next_ts, cur_ts = rev_timesteps[1:], rev_timesteps[:-1]
        next_alpha, cur_alpha = self.alpha_fn(next_ts), self.alpha_fn(cur_ts)
        ddim_coef = jnp.sqrt(1 - next_alpha) - jnp.sqrt(next_alpha / cur_alpha) * jnp.sqrt(1 - cur_alpha) # (n,)

        eps_coef = ddim_coef.reshape(-1,1) * linear_ab_coef

        return np.asarray(
            jnp.concatenate([x_coef[:, None], eps_coef], axis=1)
        ).copy()


    def get_rev_timesteps(self, num_timesteps, discr_method="uniform", last_step=False):
        # in discrete vpsde, some methods enforce [0,1, i, 2i, ...]
        # instead of [0, i-1, 2i-1, ...]
        # for a fair comparison, we set last_step is true when compared with them
        used_num_timesteps = num_timesteps - 1 if last_step else num_timesteps
        t_start = self.t_start + 1 if last_step else self.t_start
        if discr_method == 'uniform':
            steps_out = np.linspace(t_start, self.t_end, used_num_timesteps + 1, dtype=int)
        elif discr_method == 'quad':
            steps_out = (
                (
                    np.linspace(t_start, np.sqrt(self.t_end), used_num_timesteps+1) ** 2
                )
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{discr_method}"')

        if last_step:
            steps_out = np.array([0, *steps_out])

        return np.flip(steps_out).copy()



class CntVPSDE(DisVPSDE):
    def __init__(self, alpha_fn, t_start, t_end):
        self.alpha_fn = alpha_fn

        self.t_start = t_start
        self.t_end = t_end

        log_alpha_fn = lambda t: jnp.log(self.alpha_fn(t))
        grad_log_alpha_fn = jax.grad(log_alpha_fn)
        self.d_log_alpha_dtau_fn = jax.vmap(grad_log_alpha_fn)

    def get_rev_timesteps(self, num_timesteps, discr_method="uniform", last_step=False):
        del last_step
        if discr_method == 'uniform':
            steps_out = np.linspace(self.t_start, self.t_end, num_timesteps + 1)
        elif discr_method == 'quad':
            steps_out = (
                (
                    np.linspace(self.t_start, np.sqrt(self.t_end), num_timesteps+1) ** 2
                )
            )
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{discr_method}"')

        return np.flip(steps_out).copy()


def ei_ab_step(x, ei_coef, new_eps, eps_pred):
    x_coef, eps_coef = ei_coef[0], ei_coef[1:]
    full_eps_pred = [ new_eps, *eps_pred]
    rtn = x_coef * x
    for i in range(len(full_eps_pred)):
        cur_coef = eps_coef[i]
        cur_eps = full_eps_pred[i]
        #cur_coef, cur_eps in zip(eps_coef.tolist(), full_eps_pred):
        rtn += cur_coef * cur_eps
    return rtn, full_eps_pred[:-1]


def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def get_sampler(sde, num_timesteps, eps_fn, order, highest_order=3, discr_method="quad", method="deis", last_step=False):
    # eps_fn (x,scalar_t) -> eps
    if method == "deis":
        np_rev_timesteps = sde.get_rev_timesteps(num_timesteps, discr_method, last_step)
        np_ei_ab_coef = sde.get_deis_coef(order, np_rev_timesteps, highest_order)
    elif method == "ipndm":
        np_rev_timesteps = sde.get_rev_timesteps(num_timesteps, 'uniform', last_step)
        np_ei_ab_coef = sde.get_ipndm_coef(np_rev_timesteps)
        highest_order = 3
    else:
        raise RuntimeError(f"{method} is not supported")

    def sampler(x0):
        rev_timesteps = th.from_numpy(np_rev_timesteps).to(x0.device)
        ei_ab_coef = th.from_numpy(np_ei_ab_coef).to(x0.device)
        def ei_body_fn(i, val):
            x, eps_pred = val
            s_t= rev_timesteps[i]
            
            new_eps = eps_fn(x, s_t)
            new_x, new_eps_pred = ei_ab_step(x, ei_ab_coef[i], new_eps, eps_pred)
            return new_x, new_eps_pred


        eps_pred = [x0,] * highest_order
        img, _ = fori_loop(0, num_timesteps, ei_body_fn, (x0, eps_pred))
        return img
    return sampler
