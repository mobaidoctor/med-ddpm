#import jax
#import jax.numpy as jnp
import numpy as np
#import torch as th
#from jax._src.numpy.lax_numpy import _promote_dtypes_inexact
import torch


from .torch_ei import get_ab_eps_coef


def interpolate_linear(xp, fp, x, need_grad=False):
    if xp.shape[0] != fp.shape[0] or xp.ndim != 1 or fp.ndim != 1:
        raise ValueError("xp and fp must be one-dimensional arrays of equal size")
    i = torch.clip(torch.searchsorted(xp, x, right=True), 1, len(xp) - 1)
    df = fp[i] - fp[i - 1]
    dx = xp[i] - xp[i - 1]
    delta = x - xp[i - 1]
    k = (delta / dx)
    mask = (dx == 0)
    f = torch.where(mask, fp[i], fp[i - 1] + k * df)
    if need_grad:
        df = torch.where(mask, torch.zeros_like(fp[i]), k)
        return f, df
    return f, None


# def get_interp_fn(xp, fp):
#   def _fn(x):
#       if xp.shape[0] != fp.shape[0] or xp.ndim != 1 or fp.ndim != 1:
#           raise ValueError("xp and fp must be one-dimensional arrays of equal size")
#       i = torch.clip(torch.searchsorted(xp, x,right=True), 1, len(xp) - 1)
#       df = fp[i] - fp[i - 1]
#       dx = xp[i] - xp[i - 1]
#       delta = x - xp[i - 1]
#       f = torch.where((dx == 0), fp[i], fp[i - 1] + (delta / dx) * df)
#       return f
#   return _fn


class DiscreteVPSDE:
    def __init__(self, discrete_alpha):
        self.alphas = torch.FloatTensor(discrete_alpha)
        self.times = torch.arange(len(discrete_alpha))
        self.t_start = 0
        self.t_end = len(discrete_alpha) - 1

    def alpha_fn(self, vec_t, need_grad=False):
        alpha, d_alpha = interpolate_linear(self.times, self.alphas, vec_t, need_grad=need_grad)
        m1 = (alpha <= 1e-7)
        alpha[m1] = 1e-7
        m2 = (alpha >= 1.0 - 1e-7)
        alpha[m2] = 1.0 - 1e-7
        if need_grad:
            d_alpha[m1] = 0
            d_alpha[m2] = 0
            return alpha, d_alpha
        return alpha

    def d_log_alpha_dtau_fn(self, vec_t):
        #vec_t.requires_grad_(True)
        alpha, d_alpha = self.alpha_fn(vec_t, need_grad=True)
        log_alpha = torch.log(alpha)
        d_log_alpha = d_alpha*(1/alpha)
        #d_log_alpha = torch.autograd.grad(log_alpha[:, 0], vec_t[:, 0], only_inputs=True, is_grads_batched=True)
        #vec_t.requires_grad_(False)
        return d_log_alpha

    def psi(self, t_start, t_end):
        return torch.sqrt(self.alpha_fn(t_end) / self.alpha_fn(t_start))

    def eps_integrand(self, vec_t):
        d_log_alpha_dtau = self.d_log_alpha_dtau_fn(vec_t)
        integrand = -0.5 * d_log_alpha_dtau / torch.sqrt(1 - self.alpha_fn(vec_t))
        return integrand

    def get_deis_coef(self, order, rev_timesteps, highest_order=3):
        # return [x_coef, eps_coef]
        #rev_timesteps = jnp.asarray(rev_timesteps)
        x_coef = self.psi(rev_timesteps[:-1], rev_timesteps[1:])
        eps_coef = get_ab_eps_coef(self, highest_order, rev_timesteps, order)
        return np.asarray(
            torch.cat([x_coef[:, None], eps_coef], axis=1)
        ).copy()

    def get_ipndm_coef(self, rev_timesteps):
        # return [x_coef, eps_coef]
        #rev_timesteps = jnp.asarray(rev_timesteps)  # (n+1, )
        x_coef = self.psi(rev_timesteps[:-1], rev_timesteps[1:]) #(n, )

        def get_linear_ab_coef(i):
            if i == 0:
                return torch.FloatTensor([1.0, 0, 0, 0]).reshape(-1,4)
            prev_coef = get_linear_ab_coef(i-1)
            cur_coef = None
            if i == 1:
                cur_coef = torch.FloatTensor([1.5, -0.5, 0, 0])
            elif i == 2:
                cur_coef = torch.FloatTensor([23, -16, 5, 0]) / 12.0
            else:
                cur_coef = torch.FloatTensor([55, -59, 37, -9]) / 24.0
            return torch.cat(
                [prev_coef, cur_coef.reshape(-1,4)]
            )
        linear_ab_coef = get_linear_ab_coef(len(rev_timesteps) - 2) # (n, 4)

        next_ts, cur_ts = rev_timesteps[1:], rev_timesteps[:-1]
        next_alpha, cur_alpha = self.alpha_fn(next_ts), self.alpha_fn(cur_ts)
        ddim_coef = torch.sqrt(1 - next_alpha) - torch.sqrt(next_alpha / cur_alpha) * torch.sqrt(1 - cur_alpha) # (n,)

        eps_coef = ddim_coef.reshape(-1,1) * linear_ab_coef

        return np.asarray(
            torch.cat([x_coef[:, None], eps_coef], axis=1)
        ).copy()


    def get_rev_timesteps(self, num_timesteps, discr_method="uniform", last_step=False):
        # in discrete vpsde, some methods enforce [0,1, i, 2i, ...]
        # instead of [0, i-1, 2i-1, ...]
        # for a fair comparison, we set last_step is true when compared with them
        used_num_timesteps = num_timesteps - 1 if last_step else num_timesteps
        t_start = self.t_start + 1 if last_step else self.t_start
        if discr_method == 'uniform':
            steps_out = torch.linspace(t_start, self.t_end, used_num_timesteps + 1, dtype=torch.long)
        elif discr_method == 'quad':
            steps_out = (torch.linspace(t_start, np.sqrt(self.t_end), used_num_timesteps+1) ** 2).long()
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{discr_method}"')

        if last_step:
            steps_out = torch.LongTensor([0, *steps_out])

        return torch.flip(steps_out, [0]).clone()




def ei_ab_step(x, ei_coef, new_eps, eps_pred):
    x_coef, eps_coef = ei_coef[0], ei_coef[1:]
    full_eps_pred = [new_eps, *eps_pred]
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
        rev_timesteps = torch.from_numpy(np_rev_timesteps).to(x0.device)
        ei_ab_coef = torch.from_numpy(np_ei_ab_coef).to(x0.device)

        def ei_body_fn(i, val):
            x, eps_pred = val
            s_t = rev_timesteps[i]

            new_eps = eps_fn(x, s_t)
            new_x, new_eps_pred = ei_ab_step(x, ei_ab_coef[i], new_eps, eps_pred)
            return new_x, new_eps_pred

        eps_pred = [x0, ] * highest_order
        img, _ = fori_loop(0, num_timesteps, ei_body_fn, (x0, eps_pred))
        return img

    return sampler
