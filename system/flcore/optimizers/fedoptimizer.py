import torch
from torch.optim import Optimizer

import torch
import torch.optim as optim

from torch.optim.optimizer import required
from collections import defaultdict


# class DFW(optim.Optimizer):
#     def __init__(self, params, lr=required, momentum=0, weight_decay=0, eps=1e-5):
#         if lr is not required and lr <= 0.0:
#             raise ValueError("Invalid eta: {}".format(lr))
#         if momentum < 0.0:
#             raise ValueError("Invalid momentum value: {}".format(momentum))
#         if weight_decay < 0.0:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

#         defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
#         super(DFW, self).__init__(params, defaults)
#         self.eps = eps

#         for group in self.param_groups:
#             if group['momentum']:
#                 for p in group['params']:
#                     self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

#     @torch.autograd.no_grad()
#     def step(self, closure):
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
#             mu = group['momentum']
#             for param in group['params']:
#                 if param.grad is None:
#                     continue
#                 state = self.state[param]
#                 delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']

#                 param.data -= lr * (r_t + self.gamma * delta_t)

#                 if mu:
#                     z_t = state['momentum_buffer']
#                     z_t *= mu
#                     z_t -= lr * self.gamma * (delta_t + r_t)
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
        
class PerAvgOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(PerAvgOptimizer, self).__init__(params, defaults)

    def step(self, beta=0):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(beta != 0):
                    p.data.add_(other=d_p, alpha=-beta)
                else:
                    p.data.add_(other=d_p, alpha=-group['lr'])


class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)

    def step(self, server_cs, client_cs):
        for group in self.param_groups:
            for p, sc, cc in zip(group['params'], server_cs, client_cs):
                p.data.add_(other=(p.grad.data + sc - cc), alpha=-group['lr'])


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_model, device):
        group = None
        weight_update = local_model.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                localweight = localweight.to(device)
                # approximate local model
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu'] * p.data)

        return group['params']



class APFLOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(APFLOptimizer, self).__init__(params, defaults)

    def step(self, beta=1, n_k=1):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = beta * n_k * p.grad.data
                p.data.add_(-group['lr'], d_p)


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




# class DFW(optim.Optimizer):
#     def __init__(self, params, model,lr=required, momentum=0, weight_decay=0, eps=1e-5, mu=0.0):
#         if lr is not required and lr <= 0.0:
#             raise ValueError("Invalid eta: {}".format(lr))
#         if momentum < 0.0:
#             raise ValueError("Invalid momentum value: {}".format(momentum))
#         if weight_decay < 0.0:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

#         defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, mu=mu)
#         super(DFW, self).__init__(params, defaults)
#         self.eps = eps
#         self.model = model

#         for group in self.param_groups:
#             if group['mu']:
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
#             for param in group['params']:
#                 if param.grad is None:
#                     continue
#                 state = self.state[param]
#                 delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']

#                 param.data -= lr * (r_t + self.gamma * delta_t)

#                 if mu:
#                     z_t = state['momentum_buffer']
#                     z_t *= mu
#                     z_t -= lr * self.gamma * (delta_t + r_t)
#                     param.data += mu * z_t

#                 # Additional Proximal Term
#                 if mu:
#                     d_p = param.grad.data + mu * (param.data - self.model.state_dict()[param_name])  
#                     # Use self.model as the reference
#                     param.data.add_(d_p, alpha=-lr)

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

class DFW(optim.Optimizer):
    def __init__(self, params, global_params, lr=required, momentum=0, weight_decay=0, eps=1e-5, mu=0.0):
        if lr is not required and lr <= 0.0:
            raise ValueError("Invalid eta: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.global_params = global_params  # Include global parameters
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps, mu=mu)
        super(DFW, self).__init__(params, defaults)
        self.eps = eps

        for group in self.param_groups:
            if group['momentum']:
                for p in group['params']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

    @torch.autograd.no_grad()
    def step(self, closure=None):
        loss = float(closure())

        w_dict = defaultdict(dict)
        for group in self.param_groups:
            wd = group['weight_decay']
            for param in group['params']:
                if param.grad is None:
                    continue
                w_dict[param]['delta_t'] = param.grad.data
                w_dict[param]['r_t'] = wd * param.data

        self._line_search(loss, w_dict)

        for group in self.param_groups:
            lr = group['lr']
            mu = group['mu']
            for param in group['params']:
                if param.grad is None:
                    continue
                state = self.state[param]
                delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']

                param.data -= lr * (r_t + self.gamma * delta_t)

                if mu:
                    z_t = state['momentum_buffer']
                    z_t *= mu
                    z_t -= lr * self.gamma * (delta_t + r_t)
                    param.data += mu * z_t

                # Additional Proximal Term with global_params
                if mu:
                    d_p = param.grad.data + mu * (param.data - self.global_params)  # Use global_params
                    param.data.add_(d_p, alpha=-lr)

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

        self.gamma = float((num / (denom + self.defaults['eps'])).clamp(min=0, max=1))

