"""
clean training using mse loss
"""

from utils.dataset import MyDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.nn.functional import mse_loss
import random
import os

class Trainer_STAT:
    
    def __init__(self, net, optimizer, train_loader, test_loader):
        
        self.net = net
        self.optimizer = optimizer
        self.trainloader = train_loader
        self.testloader = test_loader
        
    def train(self):
        self.net.train()
        loss_sum = 0.
        for feature, target in self.trainloader:
            self.optimizer.zero_grad()
            output = self.net(feature)
            loss = mse_loss(output, target)
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item() * len(target)
            
        return loss_sum / len(self.trainloader.dataset)
    
    def eval(self):
        self.net.eval()
        loss_sum = 0.
        with torch.no_grad():
            for feature, target in self.testloader:
                output = self.net(feature)
                loss = mse_loss(output, target)
                loss_sum += loss.item() * len(target)
        
        return loss_sum / len(self.testloader.dataset)
    
    
if __name__ == '__main__':
    
    import json
    import argparse
    from utils.dataset import case_modifier
    from helper import return_nn_model
    import time
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--case_name', type = str, default = 'case14')
    args = parser.parse_args()
    
    with open("config.json") as f:
        config = json.load(f)
    
    random_seed = config['random_seed']
    batch_size = config['nn']['batch_size']
    batch_size_eval = config['nn']['batch_size_eval']
    lr = config['nn'][f'lr_mse']
    epoch = config['nn'][f'epoch_mse']
    epoch_save = config['nn']['epoch_mse_warm']
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
    
    is_small_size = config['is_small_size']
    if is_small_size:
        sample_size = len(train_dataset)
    else:
        sample_size = 'full'
    
    print("==============================================")
    print("Clean training on {} with MSE loss".format(args.case_name))
    print("Size of train dataset: {} {}".format(sample_size, len(train_dataset)))
    print("Size of test dataset: {}".format(len(test_dataset)))
    print("Shape of feature: {}".format(train_dataset[0][0].shape))
    
    
    # net
    net = return_nn_model(is_load = False)

    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print("Number of parameters: {}".format(num_params))
    print("==============================================")
    
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    trainer = Trainer_STAT(net, optimizer, trainloader, testloader)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max = T_max, eta_min = min_lr_ratio * lr)
    
    best_loss = 1e5
    
    save_path = f'{model_dir}/{sample_size}/mse.pth'
    save_path_warm = f'{model_dir}/{sample_size}/mse_warm.pth'

    if not os.path.exists(f'{model_dir}/{sample_size}'):
        os.makedirs(f'{model_dir}/{sample_size}')
        
    for i in range(1, epoch+1):
        start_time = time.time()
        train_loss = trainer.train()
        test_loss = trainer.eval()
        
        print("Epoch {}: train loss-{:.6f}, test loss-{:.6f}".format(i, train_loss, test_loss))
        #  print("Epoch {}: train loss-{:.6f}".format(i, train_loss))
        print("Time: {:.2f}s".format(time.time() - start_time))
        lr_scheduler.step()
        
        for param_group in trainer.optimizer.param_groups:
            print("LR: {:.6f}".format(param_group['lr']))
        
        if watch == 'train' and train_loss < best_loss:
            best_loss = train_loss
            torch.save(trainer.net.state_dict(), save_path)
            print("Best model saved by train!")
            
        if watch == 'test' and test_loss < best_loss:
            best_loss = test_loss
            torch.save(trainer.net.state_dict(), save_path)
            print("Best model saved by test!")
        
        if i == epoch_save:
            torch.save(trainer.net.state_dict(), save_path_warm)
            print("Warm Model saved!")
        
        print("==============================================")
    
    

    
    
    
    
    