"""
clean train nn with spo loss
"""

from utils.dataset import MyDataset
from torch.utils.data import DataLoader
from utils.net import NN_SPO
import torch
import numpy as np
import random
import os
from tqdm import tqdm

class Trainer_SPO:
    
    def __init__(self, net, optimizer, train_loader, test_loader, operator, clip_norm):
        """
        b_default: default susceptance for the power grid
        """
        self.net = net
        assert self.net.name == 'NN_SPO'
        self.optimizer = optimizer
        self.trainloader = train_loader
        self.testloader = test_loader
        self.first_coeff = torch.tensor(operator.first_coeff, dtype=torch.float)
        self.load_shed_coeff = torch.tensor(operator.load_shed_coeff, dtype = torch.float)
        self.gen_storage_coeff = torch.tensor(operator.gen_storage_coeff, dtype = torch.float)
        self.b_default = torch.from_numpy(operator.b).float()
        self.clip_norm = clip_norm
        
    def loss(self, pg, ls, gs):
        loss = pg @ self.first_coeff + ls @ self.load_shed_coeff + gs @ self.gen_storage_coeff
        
        return loss
    
    def train(self):
        self.net.train()
        loss_sum = 0.

        for feature, target in tqdm(self.trainloader, total = len(self.trainloader)):
            self.optimizer.zero_grad()
            forecast_load, pg, ls, gs = self.net(feature, target, self.b_default.repeat(len(target), 1))
            loss = self.loss(pg, ls, gs)
            loss = loss.mean()
            loss.backward()
            # gradient clip
            if self.clip_norm != 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), norm_type = 1, max_norm = self.clip_norm)
            
            self.optimizer.step()
            loss_sum += loss.item() * len(target)
        
        return loss_sum / len(self.trainloader.dataset)

    def eval(self):
        self.net.eval()
        loss_sum = 0.
        with torch.no_grad():
            for feature, target in self.testloader:
                forecast_load, pg, ls, gs = self.net(feature, target, self.b_default.repeat(len(target), 1))
                loss = self.loss(pg, ls, gs)
                loss = loss.mean()
                loss_sum += loss.item() * len(target)
            
        return loss_sum / len(self.testloader.dataset)

if __name__ == '__main__':
    
    import json
    import argparse
    from helper import return_nn_model, return_operator
    import time
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--case_name', type = str, default = 'case14')
    parser.add_argument('-p', '--pre_train', default = False, action = 'store_true')
    args = parser.parse_args()
    
    with open("config.json") as f:
        config = json.load(f)
    
    random_seed = config['random_seed']
    batch_size = config['nn']['batch_size_spo']
    batch_size_eval = config['nn']['batch_size_spo']
    lr = config['nn']['lr_spo']
    epoch = config['nn']['epoch_spo']
    model_dir = config['nn']['model_dir']
    watch = config['nn']['watch_spo']
    fix_first_b = config['fix_first_b']
    is_scale = config['is_scale']
    gradient_clip_norm = config['nn']['gradient_clip_norm']
    T_max = config['nn']['T_max']
    min_lr_ratio = config['nn']['min_lr_ratio']
    train_with_test = config['nn']['train_with_test_spo']
    solver_args = config['nn']['solver_args']
    
    if watch == 'test':
        assert train_with_test == True 
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # data
    train_dataset = MyDataset(case_name = args.case_name, mode = "train")
    test_dataset = MyDataset(case_name = args.case_name, mode = "test")
    trainloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    testloader = DataLoader(test_dataset, batch_size = batch_size_eval, shuffle = False)
    
    print("Training on {} with SPO loss".format(args.case_name))
    print("Size of train dataset: {}".format(len(train_dataset)))
    print("Shape of feature: {}".format(train_dataset[0][0].shape))
    print("epoch: ", epoch)
    
    # net
    operator = return_operator(args.case_name)
    # pack the optimization layers
    net = return_nn_model(case_name = args.case_name, is_load = args.pre_train, train_method = f'mse_warm')
    assert net.name == 'NN'
    
    if is_scale:
        mean = train_dataset.target_mean
        std = train_dataset.target_std
    else:
        mean = 0
        std = 1
    net = NN_SPO(model = net, operator=operator, mean = mean, std = std, 
                fix_first_b = fix_first_b,
                solver_args=solver_args) # construct the spo model
    print('nn structure')
    print(net)
    assert net.name == 'NN_SPO'
    
    is_small_size = config['is_small_size']
    if is_small_size:
        sample_size = len(train_dataset)
    else:
        sample_size = 'full'
    
    # optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    optimizer = torch.optim.SGD(net.parameters(), lr = lr)
    trainer = Trainer_SPO(net, optimizer, trainloader, testloader, operator, gradient_clip_norm)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
    save_path = f'{model_dir}/{sample_size}/spo_clean.pth'
        
    best_loss = 1e5
    for i in range(1, epoch+1):
        start_time = time.time()
        train_loss = trainer.train()
        
        if train_with_test:
            test_loss = trainer.eval()
            print("Epoch {}: train loss-{:.4f}, test loss-{:.4f}".format(i, train_loss, test_loss))
        else:
            print("Epoch {}: train loss-{:.4f}".format(i, train_loss))
        
        print("Time: {:.2f}s".format(time.time() - start_time))
        
        for param_group in trainer.optimizer.param_groups:
            print("LR: {:.6f}".format(param_group['lr']))
        
        # reduce the learning rate
        # if i == int((epoch+1)/2):
        #     for param_group in trainer.optimizer.param_groups:
        #         param_group['lr'] *= 0.2
        
        if watch == 'train' and train_loss < best_loss:
            best_loss = train_loss
            torch.save(trainer.net.state_dict(), save_path)
            print("Best model saved by train!")
            
        if watch == 'test' and test_loss < best_loss:
            best_loss = test_loss
            torch.save(trainer.net.state_dict(), save_path)
            print("Best model saved by test!")
        
        print("==============================================")