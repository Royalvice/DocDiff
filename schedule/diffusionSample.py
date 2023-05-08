import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.utils


def extract_(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusion(nn.Module):
    def __init__(self, model, T, schedule):
        super().__init__()
        self.visual = False
        if self.visual:
            self.num = 0
        self.model = model
        self.T = T
        self.schedule = schedule
        betas = self.schedule.get_betas()
        self.register_buffer('betas', betas.float())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        gammas = alphas_bar

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        # calculation for q(y_t|y_{t-1})
        self.register_buffer('gammas', gammas)
        self.register_buffer('sqrt_one_minus_gammas', np.sqrt(1 - gammas))
        self.register_buffer('sqrt_gammas', np.sqrt(gammas))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, cond_, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        #var = self.betas
        var = extract(var, t, x_t.shape)
        eps = self.model(torch.cat((x_t, cond_), dim=1), t)
        # nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        # eps = (1. + self.w) * eps - self.w * nonEps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def noisy_image(self, t, y):
        """ Compute y_noisy according to (6) p15 of [2]"""
        noise = torch.randn_like(y)
        y_noisy = extract_(self.sqrt_gammas, t, y.shape) * y + extract_(self.sqrt_one_minus_gammas, t, noise.shape) * noise
        return y_noisy, noise

    def forward(self, x_T, cond, pre_ori='False'):
        """
        Algorithm 2.
        """
        x_t = x_T
        cond_ = cond
        for time_step in reversed(range(self.T)):
            print("time_step: ", time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            if pre_ori == 'False':
                mean, var = self.p_mean_variance(x_t=x_t, t=t, cond_=cond_)
                if time_step > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                x_t = mean + torch.sqrt(var) * noise
                assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
            else:
                if time_step > 0:
                    ori = self.model(torch.cat((x_t, cond_), dim=1), t)
                    eps = x_t - extract_(self.sqrt_gammas, t, ori.shape) * ori
                    eps = eps / extract_(self.sqrt_one_minus_gammas, t, eps.shape)
                    x_t = extract_(self.sqrt_gammas, t - 1, ori.shape) * ori + extract_(self.sqrt_one_minus_gammas, t - 1, eps.shape) * eps
                else:
                    x_t = self.model(torch.cat((x_t, cond_), dim=1), t)

        x_0 = x_t
        return x_0


if __name__ == '__main__':
    from schedule import Schedule
    test = GaussianDiffusion(None, 100, Schedule('linear', 100))
    print(test.gammas)
