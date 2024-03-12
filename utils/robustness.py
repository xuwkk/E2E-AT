"""
do adversarial attack
"""

from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from torch.nn.functional import mse_loss
from time import time
import json

class PGD:
    
    """
    attack on the [input, parameter] of the task aware end to end learning
    """
    
    def __init__(self, operator, is_spo, nn, attack_method, no_iter, **kwargs):
        
        with open('config.json') as f:
            config = json.load(f)
        
        self.fix_first_b = config['nn']['fix_first_b']
        feature_size = config['nn']['feature_size']
        
        self.b_default = torch.from_numpy(operator.b).float() 
        self.no_iter = no_iter
        self.attack_method = attack_method
        self.is_spo = is_spo
        
        # check model type
        if self.is_spo:
            assert nn.name == 'NN_SPO'
        else:
            assert nn.name == 'NN'
        
        self.nn = nn
        self.nn.eval()
        
        # input feature attack
        if self.attack_method == 'input' or self.attack_method == 'both':
            self.max_eps_input = kwargs['max_eps_input'] # ! absolute value
            assert self.max_eps_input <= 1
            self.step_size_input = self.max_eps_input / no_iter * 2 # covering two times of the max_eps_input
        
        # parameter attack
        if self.attack_method == 'parameter' or self.attack_method == 'both':
            self.max_eps_parameter = kwargs['max_eps_parameter'] # ! percentage value
            assert self.max_eps_parameter <= 1
            # b_max and b_min are the same for all batches
            self.b_min = self.b_default * (1 - self.max_eps_parameter)
            self.b_max = self.b_default * (1 + self.max_eps_parameter)
            self.step_size_parameter = (self.b_max - self.b_min) / no_iter * 2 # covering two times of the max_eps_parameter
        
        if self.attack_method == 'parameter' or self.attack_method == 'both':
            assert self.is_spo == True
        
        self.first_coeff = torch.tensor(operator.first_coeff, dtype=torch.float)
        self.load_shed_coeff = torch.tensor(operator.load_shed_coeff, dtype = torch.float)
        self.gen_storage_coeff = torch.tensor(operator.gen_storage_coeff, dtype = torch.float)
        
        if self.attack_method == 'input' or self.attack_method == 'both':
            self.fixed_feature = kwargs['fixed_feature']
            self.flexible_feature = list(set(np.arange(feature_size)) - set(self.fixed_feature))
        
    def input_bound_clamp(self, X):
        
        X_min = deepcopy(X)
        X_max = deepcopy(X)
        
        X_min[:, self.flexible_feature] = (X_min[:, self.flexible_feature] - self.max_eps_input).clamp(0, 1)
        X_max[:, self.flexible_feature] = (X_max[:, self.flexible_feature] + self.max_eps_input).clamp(0, 1)
        
        assert torch.all(X_min[:, self.flexible_feature] >= 0)
        assert torch.all(X_max[:, self.flexible_feature] <= 1)
        assert torch.all(X_min[:, self.fixed_feature] == X[:, self.fixed_feature])
        assert torch.all(X_max[:, self.fixed_feature] == X[:, self.fixed_feature])
        
        return (X_min, X_max)    

    def loss_mse(self, forecast_load, target):
        # mse loss
        loss = mse_loss(forecast_load, target)
        return loss
    
    def loss_spo(self, pg, ls, gs):
        # spo loss
        loss = pg @ self.first_coeff + ls @ self.load_shed_coeff + gs @ self.gen_storage_coeff
        return loss.mean()
    
    def attack(self, feature, target):
        
        if self.attack_method == 'input' or self.attack_method == 'both':
            X_min, X_max = self.input_bound_clamp(feature)
            X_att = torch.rand_like(X_min) * (X_max - X_min) + X_min # initialize the input
            X_att.requires_grad = True
        
        if self.attack_method == 'parameter' or self.attack_method == 'both':
            # start from the default susceptance
            # extend the dimension to batch_size
            # b_att = deepcopy(self.b_default).repeat(target.shape[0], 1) 
            b_att = torch.rand_like(self.b_default).repeat(target.shape[0], 1) * (self.b_max - self.b_min) + self.b_min
            b_att.requires_grad = True
        
        if self.attack_method == 'input':
            # attack on the input
            for t in range(self.no_iter):
                with torch.enable_grad():

                    if self.is_spo:
                        forecast_load, pg, ls, gs = self.nn(X_att, target, self.b_default.repeat(target.shape[0], 1))
                        loss = self.loss_spo(pg, ls, gs)
                    else:
                        forecast_load = self.nn(X_att)
                        loss = self.loss_mse(forecast_load, target)
                    
                    loss.backward()
                    # update attack using gradient ascent
                    X_att.data = X_att.data + self.step_size_input * X_att.grad.data.sign()
                    X_att.data.clamp_(min = X_min, max = X_max)
                    X_att.grad.zero_()
            
            assert torch.all(X_att[:, self.fixed_feature] == feature[:, self.fixed_feature])
            assert torch.all(X_att[:, self.flexible_feature] >= X_min[:, self.flexible_feature])
            assert torch.all(X_att[:, self.flexible_feature] <= X_max[:, self.flexible_feature])
            
            X_att.detach()
            return X_att
        
        elif self.attack_method == 'parameter':
            # attack on the design parameter of the optimization
            for t in range(self.no_iter):
                with torch.enable_grad():
                    forecast_load, pg, ls, gs = self.nn(feature, target, b_att)
                    loss = self.loss_spo(pg, ls, gs)
                    loss.backward()
                    # each susceptance is updated according to their attack range
                    att_vector = self.step_size_parameter.repeat(target.shape[0], 1) * b_att.grad.data.sign()
                    b_att.data = b_att.data + att_vector # size: (batch_size, no_branch) gradient ascent
                    
                    # projection
                    b_att.data.clamp_(self.b_min, self.b_max)
                    b_att.grad.zero_()
            
            b_att.detach()
            assert torch.all(b_att >= self.b_min)
            assert torch.all(b_att <= self.b_max)
            
            return b_att
        
        elif self.attack_method == 'both':
            # attack on both input and parameter
            for t in range(self.no_iter):
                with torch.enable_grad():
                    
                    forecast_load, pg, ls, gs = self.nn(X_att, target, b_att)
                    loss = self.loss_spo(pg, ls, gs)
                    loss.backward()
                    # update input feature
                    X_att.data = X_att.data + self.step_size_input * X_att.grad.data.sign()
                    X_att.data.clamp_(min = X_min, max = X_max)
                    X_att.grad.zero_()
                    
                    # update parameter
                    att_vector = self.step_size_parameter.repeat(target.shape[0], 1) * b_att.grad.data.sign()
                    b_att.data = b_att.data + att_vector # size: (batch_size, no_branch) gradient ascent
                    b_att.data.clamp_(self.b_min, self.b_max)
                    b_att.grad.zero_()
                    
            # check
            X_att.detach()
            assert torch.all(X_att[:, self.fixed_feature] == feature[:, self.fixed_feature])
            assert torch.all(X_att[:, self.flexible_feature] >= X_min[:, self.flexible_feature])
            assert torch.all(X_att[:, self.flexible_feature] <= X_max[:, self.flexible_feature])
            
            b_att.detach()
            assert torch.all(b_att >= self.b_min)
            assert torch.all(b_att <= self.b_max)
            
            return X_att, b_att
            
