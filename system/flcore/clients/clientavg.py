import torch
import torch.nn as nn
import numpy as np
import copy
import time
from flcore.clients.clientbase import Client
from utils.privacy import *
from flcore.optimizers.fedoptimizer import DFW
from flcore.optimizers.dfw import DFW1



class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.round_counter = 0
        self.loss = nn.CrossEntropyLoss()
        self.mu = args.mu
        self.global_params = copy.deepcopy(list(self.model.parameters()))
        #self.loss = HingeLoss()
        #self.loss = nn.MultiMarginLoss()
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = DFW1(self.model.parameters(), lr=0.1)
        #self.optimizer = DFW(self.model.parameters(),self.global_params, lr=0.1, momentum=0, mu = self.mu)

    def train(self):
        trainloader = self.load_train_data()        
        start_time = time.time()
        self.model.train()
        max_local_steps = 1
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))                    
                self.round_counter += 1
                if self.round_counter % self.local_steps == 0:
                    self.optimizer.zero_grad()
                    output = self.model(x)
                    loss = self.loss(output, y)
                    loss.backward()
                    if self.privacy:
                        dp_step(self.optimizer, i, len(trainloader))
                    else:
                        #self.optimizer.step()
                        self.optimizer.step(lambda: float(loss))
                        #self.optimizer.step(self.global_params, self.device, lambda: float(loss))

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            res, DELTA = get_dp_params(self.optimizer)
            print(f"Client {self.id}", f"(ε = {res[0]:.2f}, δ = {DELTA}) for α = {res[1]}")
