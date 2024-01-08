import torch
from torch.optim import Optimizer

import torch
import torch.optim as optim

from torch.optim.optimizer import required
from collections import defaultdict





class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        default = dict(lr=lr, mu=mu)
        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params, device):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                g = g.to(device)
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])

# class DFWP(Optimizer):
#     def __init__(self, params, global_params, lr=required, momentum=0.9, weight_decay=0, eps=1e-5, mu=0.0):
#         if lr is not required and lr <= 0.0:
#             raise ValueError("Invalid eta: {}".format(lr))
#         if momentum < 0.0:
#             raise ValueError("Invalid momentum value: {}".format(momentum))
#         if weight_decay < 0.0:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

#         defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, mu=mu)
#         super(DFWP, self).__init__(params, defaults)
#         self.eps = eps
#         self.global_params = global_params

#         for group in self.param_groups:
#             if group['momentum']:
#                 for p in group['params']:
#                     self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

#     @torch.autograd.no_grad()
#     def step(self, closure=None):
#         loss = float(closure())

#         w_dict = defaultdict(dict)
#         for group in self.param_groups:
#             wd = group['weight_decay']
#             for param in group['params']:
#                 if param.grad is None:
#                     continue
#                 w_dict[param]['delta_t'] = param.grad.data
#                 w_dict[param]['r_t'] = wd * param.data

#         self._line_search(loss, w_dict)

#         for group in self.param_groups:
#             lr = group['lr']
#             mu = group['mu']
#             for param, global_param in zip(group['params'], self.global_params):
#                 if param.grad is None:
#                     continue
#                 state = self.state[param]
#                 delta_t = param.grad.data
#                 regularization_term = mu * (param.data - global_param.data)

#                 param.data -= lr * (delta_t + regularization_term)

#                 if group['momentum']:
#                     z_t = state['momentum_buffer']
#                     z_t *= mu
#                     z_t -= lr * (delta_t + regularization_term)
#                     param.data += mu * z_t

#     @torch.autograd.no_grad()
#     def _line_search(self, loss, w_dict):
#         """
#         Computes the line search in closed form.
#         """
#         num = loss
#         denom = 0

#         for group in self.param_groups:
#             lr = group['lr']
#             for param in group['params']:
#                 if param.grad is None:
#                     continue
#                 delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']
#                 num -= lr * torch.sum(delta_t * r_t)
#                 denom += lr * delta_t.norm() ** 2

#         self.gamma = float((num / (denom + self.eps)).clamp(min=0, max=1))



                    
  
class DFWP(Optimizer):
    def __init__(self, params, global_params, lr=required, momentum=0.9, weight_decay=0, eps=1e-5, mu=0.0):
        if lr is not required and lr <= 0.0:
            raise ValueError("Invalid eta: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, mu=mu)
        super(DFWP, self).__init__(params, defaults)
        self.eps = eps
        self.global_params = global_params

        for group in self.param_groups:
            if group['momentum']:
                for p in group['params']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)
    @torch.autograd.no_grad()
    def step(self, global_params, device, closure=None):
        loss = float(closure())

        w_dict = defaultdict(dict)
        for group in self.param_groups:
            wd = group['weight_decay']
            mu = group['mu']
            momentum = group['momentum']
            for param in group['params']:
                if param.grad is None:
                    continue
                w_dict[param]['delta_t'] = param.grad.data
                w_dict[param]['r_t'] = wd * param.data

        self._line_search(loss, w_dict)

        for group in self.param_groups:
            lr = group['lr']
            mu = group['mu']
            momentum = group['momentum']
            for param, global_param in zip(group['params'], global_params):
                if param.grad is None:
                    continue
                state = self.state[param]
                delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']

                # Regularization term
                regularization_term = mu * (param.data - global_param.data)
                param.data -= lr * (r_t + self.gamma * delta_t + regularization_term)

                # Momentum
                if momentum:
                    z_t = state['momentum_buffer']
                    z_t *= momentum
                    z_t -= lr * self.gamma * (delta_t + r_t + regularization_term)
                    param.data += momentum * z_t


    @torch.autograd.no_grad()
    def _line_search(self, loss, w_dict):
        """
        Computes the line search in closed form.
        """
        num = loss
        denom = 0

        for group in self.param_groups:
            lr = group['lr']
            for param in group['params']:
                if param.grad is None:
                    continue
                delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']
                num -= lr * torch.sum(delta_t * r_t)
                denom += lr * delta_t.norm() ** 2

        self.gamma = float((num / (denom + self.eps)).clamp(min=0, max=1))



