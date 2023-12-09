"""
adversarial train the neural network of mse loss
"""

from utils.dataset import MyDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.nn.functional import mse_loss
import random
import os
import json
from copy import deepcopy

class Trainer:
    
    def __init__(self, net, optimizer, train_loader, test_loader, max_eps):
        
        self.net = net
        self.optimizer = optimizer
        self.trainloader = train_loader
        self.testloader = test_loader
        
        with open("config.json") as f:
            config = json.load(f)
        
        self.fixed_feature = config['attack']['fixed_feature']
        self.feature_size = config['nn']['feature_size']
        self.flexible_feature = list(set(np.arange(self.feature_size)) - set(self.fixed_feature))
        self.max_eps_input = max_eps
        self.no_iter = config['attack']['no_iter']
        self.step_size_input = self.max_eps_input / self.no_iter * 2
        print('pgd step size input: {}'.format(self.step_size_input))
        
    def input_bound_clamp(self, feature):
        feature_min = deepcopy(feature)
        feature_max = deepcopy(feature)
        
        feature_min[:, self.flexible_feature] = (feature_min[:, self.flexible_feature] - self.max_eps_input).clamp(0,1)
        feature_max[:, self.flexible_feature] = (feature_max[:, self.flexible_feature] + self.max_eps_input).clamp(0,1)
        
        return feature_min, feature_max
    
    def train(self):
        """
        conventional adversarial training
        """
        assert self.net.name == 'NN' # ! use NN model (a feedforward neural network)
        self.net.train()
        loss_sum = 0.
        
        for feature, target in self.trainloader:
            
            # generate attack
            self.net.eval()
            
            feature_min, feature_max = self.input_bound_clamp(feature)
            feature_att = torch.rand_like(feature_min) * (feature_max - feature_min) + feature_min
            # verify the attack strength
            assert torch.all(feature_att[:, self.fixed_feature] == feature[:, self.fixed_feature]), "fixed feature is not the same"
            assert torch.all(feature_att[:, self.flexible_feature] >= 0), "feature_att < 0"
            assert torch.all(feature_att[:, self.flexible_feature] <= 1), "feature_att > 1"
            att_eps = (feature_att[:, self.flexible_feature] - feature[:, self.flexible_feature]).abs()
            assert torch.all(att_eps <= self.max_eps_input + 1e-5), "att_eps > max_eps_input"
            
            feature_att.requires_grad_()
            
            # generate adversarial example
            for _ in range(self.no_iter):
                forecast_load = self.net(feature_att)
                loss = mse_loss(forecast_load, target)
                loss.backward()
                assert torch.norm(feature_att.grad.data) != 0
                feature_att.data = feature_att.data + self.step_size_input * feature_att.grad.data.sign()
                # clamp
                feature_att.data.clamp_(min = feature_min, max = feature_max)
                assert torch.all(feature_att[:, self.fixed_feature] == feature[:, self.fixed_feature])
                feature_att.grad.zero_()
            
            # update network
            self.net.train()
            forecast_load = self.net(feature_att)
            loss = mse_loss(forecast_load, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            loss_sum += loss.item() * len(target)
        
        return loss_sum / len(self.trainloader.dataset)

            
if __name__ == "__main__":
    
    import json
    import argparse
    from utils.dataset import case_modifier
    from utils.optimization import Operator
    from helper import return_nn_model
    import time
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--case_name', type = str, default = 'case14')
    parser.add_argument('-e', '--max_eps', type = float)
    args = parser.parse_args()
    
    print(f"train on {args.case_name}; max eps: {args.max_eps}")
    with open("config.json") as f:
        config = json.load(f)
    
    random_seed = config['random_seed']
    batch_size = config['nn']['batch_size']
    batch_size_eval = config['nn']['batch_size_eval']
    lr = config['nn'][f'lr_mse']
    epoch = config['nn'][f'epoch_mse']
    model_dir = config['nn']['model_dir']
    watch = config['nn']['watch_mse']
    T_max = config['nn']['T_max']
    min_lr_ratio = config['nn']['min_lr_ratio']
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # data
    train_dataset = MyDataset(case_name = args.case_name, mode = "train")
    test_dataset = MyDataset(case_name = args.case_name, mode = "test")
    trainloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    testloader = DataLoader(test_dataset, batch_size = batch_size_eval, shuffle = False)
    
    print("Training on {} with MSE loss under attack eps {}".format(args.case_name, args.max_eps))
    print("Size of train dataset: {}".format(len(train_dataset)))
    print("Shape of feature: {}".format(train_dataset[0][0].shape))
    
    is_small_size = config['is_small_size']
    if is_small_size:
        sample_size = len(train_dataset)
    else:
        sample_size = 'full'
    
    # net
    no_load = train_dataset.no_load
    case = case_modifier(case_name = args.case_name)
    operator = Operator(case)
    
    net = return_nn_model(case_name = args.case_name, is_load = False)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    trainer = Trainer(net = net, optimizer = optimizer, train_loader = trainloader, test_loader = testloader, max_eps=args.max_eps)
    
    save_path = f'{model_dir}/{sample_size}/mse_robust-{args.max_eps}.pth'
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max = T_max, eta_min = min_lr_ratio * lr)
    best_loss = 1e5
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    for i in range(1, epoch+1):
        start_time = time.time()
        train_loss = trainer.train()
        
        print("Epoch {}: train loss-{:.6f}".format(i, train_loss))
        print("Time: {:.2f}s".format(time.time() - start_time))
        lr_scheduler.step()
        
        for param_group in trainer.optimizer.param_groups:
            print("LR: {:.6f}".format(param_group['lr']))
        
        if watch == 'train' and train_loss < best_loss:
            best_loss = train_loss
            torch.save(trainer.net.state_dict(), save_path)
            print("Best model saved by train!")