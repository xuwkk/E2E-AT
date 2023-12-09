"""
adversarial training for spo
1. input
2. parameter
3. both
"""

from utils.dataset import MyDataset
from torch.utils.data import DataLoader
from utils.net import NN_SPO
import torch
import numpy as np
import random
import json
from utils.robustness import PGD
from copy import deepcopy
from tqdm import tqdm


class Trainer:
    
    def __init__(self, operator, net, optimizer, train_loader, test_loader, max_eps_input, max_eps_parameter, init_method, attack_method, train_method, grad_clip, alpha):
        """
        contains all information to train robust model
        """
        self.net = net
        assert self.net.name == 'NN_SPO'
        self.optimizer = optimizer
        self.trainloader = train_loader
        self.testloader = test_loader
        
        with open("config.json") as f:
            config = json.load(f)
        
        self.fix_first_b = config['fix_first_b']
        self.no_iter = config['attack']['no_iter']
        
        # cost parameters
        self.first_coeff = torch.tensor(operator.first_coeff, dtype=torch.float)
        self.load_shed_coeff = torch.tensor(operator.load_shed_coeff, dtype = torch.float)
        self.gen_storage_coeff = torch.tensor(operator.gen_storage_coeff, dtype = torch.float)
        
        # input feature attack settings
        self.fixed_feature = config['attack']['fixed_feature']
        self.feature_size = config['nn']['feature_size']
        self.flexible_feature = list(set(np.arange(self.feature_size)) - set(self.fixed_feature))
        self.max_eps_input = max_eps_input
        self.step_size_input = self.max_eps_input / no_iter * 2
        
        # parameter attack settings
        self.b_default = torch.from_numpy(operator.b).float()
        self.max_eps_parameter = max_eps_parameter
        self.b_min = self.b_default * (1 - self.max_eps_parameter)
        self.b_max = self.b_default * (1 + self.max_eps_parameter)
        self.step_size_parameter = (self.b_max - self.b_min) / no_iter * 2
        
        self.init_method = init_method # random or previous or default
        self.attack_method = attack_method # input or parameter or both
        self.train_method = train_method # normal or free
        self.grad_clip = grad_clip
        self.alpha = alpha  # balance between clean loss and adversarial loss
        assert alpha > 0 and alpha <= 1

        print('step size input: ', self.step_size_input, '\nstep size parameter: ', self.step_size_parameter)
        
    def loss(self, pg, ls, gs):
        loss = pg @ self.first_coeff + ls @ self.load_shed_coeff + gs @ self.gen_storage_coeff
        return loss.mean() 
    
    def nn_loss(self, feature, target, b):
        _, pg, ls, gs = self.net(feature, target, b)
        loss = self.loss(pg, ls, gs)
        return loss
    
    def input_bound_clamp(self, feature):
        feature_min = deepcopy(feature)
        feature_max = deepcopy(feature)
        feature_min[:, self.flexible_feature] = (feature_min[:, self.flexible_feature] - self.max_eps_input).clamp(0,1)
        feature_max[:, self.flexible_feature] = (feature_max[:, self.flexible_feature] + self.max_eps_input).clamp(0,1)
        return feature_min, feature_max
    
    def train(self):
        """
        attack_method:
            input: only attack the input feature
            parameter: only attack the parameter
            both: attack both the input feature and the parameter
        train_method:
            free: adversarial training for free, for a minibatch, find the gradient on both attack and the neural network parameter simultaneously
            normal: the normal adversarial training, finding the attack first and then update the neural network parameter
        init_method:
            random: random initialization
            previous: use the previous adversarial input as the initialization, disable shuffle in the dataloader
            default: use the default input as the initialization
        """
        self.net.train()
        loss_sum = 0. # sum over the entire dataset
        
        global feature_att_global # record the adversarial input if the init method is previous
        global parameter_att_global
        
        for batch_index, (feature, target) in tqdm(enumerate(self.trainloader), total = len(self.trainloader)):
            self.optimizer.zero_grad()
            
            """
            attack the input
            """
            if self.attack_method == 'input':
                feature_min, feature_max = self.input_bound_clamp(feature)
                if self.init_method == 'random':
                    feature_att = torch.rand_like(feature_min) * (feature_max - feature_min) + feature_min
                elif self.init_method == 'previous':
                    feature_att = deepcopy(feature_att_global[batch_index])
                elif self.init_method == 'default':
                    feature_att = deepcopy(feature)
                feature_att.requires_grad = True
                # assert torch.all(feature_att[:, self.fixed_feature] == feature[:, self.fixed_feature]), "fixed feature is not the same"
                
                if self.train_method == "normal":
                    self.net.eval()
                    # attack
                    for pgd_no in range(self.no_iter):
                        loss = self.nn_loss(feature_att, target, self.b_default.repeat(feature_att.shape[0], 1))
                        loss.backward()
                        assert torch.norm(feature_att.grad.data) != 0
                        feature_att.data = feature_att.data + self.step_size_input * feature_att.grad.data.sign()
                        feature_att.data.clamp_(min = feature_min, max = feature_max)
                        # assert torch.all(feature_att[:, self.fixed_feature] == feature[:, self.fixed_feature]), "fixed feature is not the same"
                        feature_att.grad.zero_()
                    # nn parameter
                    feature_att.requires_grad = False
                    self.net.train()
                    loss = self.nn_loss(feature_att, target, self.b_default.repeat(feature_att.shape[0], 1))
                    if self.alpha == 1:
                        pass
                    else:
                        # clean loss
                        loss_clean = self.nn_loss(feature, target, self.b_default.repeat(feature.shape[0], 1))
                        loss = self.alpha * loss + (1 - self.alpha) * loss_clean
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), norm_type=1, max_norm = self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                elif self.train_method == "free":
                    # adversarial training for free
                    self.net.train()
                    for pgd_no in range(self.no_iter):
                        loss = self.nn_loss(feature_att, target, self.b_default.repeat(feature_att.shape[0], 1))
                        if self.alpha == 1:
                            pass
                        else:
                            loss_clean = self.nn_loss(feature, target, self.b_default.repeat(feature.shape[0], 1))
                            loss = self.alpha * loss + (1 - self.alpha) * loss_clean
                        loss.backward()
                        # feature attack
                        # assert torch.norm(feature_att.grad.data) != 0
                        feature_att.data = feature_att.data + self.step_size_input * feature_att.grad.data.sign()
                        feature_att.data.clamp_(min = feature_min, max = feature_max)
                        # assert torch.all(feature_att[:, self.fixed_feature] == feature[:, self.fixed_feature])
                        feature_att.grad.zero_()
                        # nn parameter
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), norm_type=1, max_norm = self.grad_clip)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                
                # update global variable
                feature_att_global[batch_index] = feature_att.detach()
                
            elif self.attack_method == 'parameter':
                # attack initialization
                if self.init_method == 'random':
                    b_att = torch.rand_like(self.b_default).repeat(target.shape[0], 1) * (self.b_max - self.b_min) + self.b_min
                elif self.init_method == 'previous':
                    b_att = deepcopy(parameter_att_global[batch_index])
                elif self.init_method == 'default':
                    b_att = deepcopy(self.b_default).repeat(target.shape[0], 1)
                b_att.requires_grad = True
                
                if self.train_method == "normal":
                    # parameter attack
                    self.net.eval()
                    for pgd_no in range(self.no_iter):
                        loss = self.nn_loss(feature, target, b_att)
                        loss.backward()
                        # assert torch.norm(b_att.grad.data) != 0
                        b_att.data = b_att.data + self.step_size_parameter.repeat(target.shape[0], 1) * b_att.grad.data.sign() # different b has different step size
                        b_att.data.clamp_(self.b_min, self.b_max)
                        b_att.grad.zero_()
                    # nn parameter
                    b_att.requires_grad = False
                    self.net.train()
                    loss = self.nn_loss(feature, target, b_att)
                    # loss
                    if self.alpha == 1:
                        pass
                    else:
                        # clean loss
                        loaa_clean = self.nn_loss(feature, target, self.b_default.repeat(feature.shape[0], 1))
                        loss = self.alpha * loss + (1 - self.alpha) * loss_clean
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), norm_type=1, max_norm = self.grad_clip)
                    self.optimizer.step() # update the nn parameter
                    self.optimizer.zero_grad()
                
                elif self.train_method == "free":
                    self.net.train()
                    for pgd_no in range(self.no_iter):
                        loss = self.nn_loss(feature, target, b_att)
                        if self.alpha == 1:
                            pass
                        else:
                            loss_clean = self.nn_loss(feature, target, self.b_default.repeat(feature.shape[0], 1))
                            loss = self.alpha * loss + (1 - self.alpha) * loss_clean
                        loss.backward()
                        # parameter attack
                        # assert torch.norm(b_att.grad.data) != 0
                        b_att.data = b_att.data + self.step_size_parameter.repeat(target.shape[0], 1) * b_att.grad.data.sign()
                        b_att.data.clamp_(self.b_min, self.b_max)
                        b_att.grad.zero_()
                        # nn parameter
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), norm_type=1, max_norm = self.grad_clip)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                
                # update the global
                parameter_att_global[batch_index] = b_att.detach()
            
            elif self.attack_method == 'both':
                
                # attack initialization
                feature_min, feature_max = self.input_bound_clamp(feature)
                if self.init_method == 'random':
                    feature_att = torch.rand_like(feature_min) * (feature_max - feature_min) + feature_min
                    b_att = torch.rand_like(self.b_default).repeat(target.shape[0], 1) * (self.b_max - self.b_min) + self.b_min
                elif self.init_method == 'previous':
                    feature_att = deepcopy(feature_att_global[batch_index])
                    b_att = deepcopy(parameter_att_global[batch_index])
                elif self.init_method == 'default':
                    feature_att = deepcopy(feature)
                    b_att = deepcopy(self.b_default).repeat(target.shape[0], 1)
                feature_att.requires_grad = True
                b_att.requires_grad = True
                assert torch.all(feature_att[:, self.fixed_feature] == feature[:, self.fixed_feature])
                
                if self.train_method == "normal":
                    self.net.eval()
                    # attack
                    for pgd_no in range(self.no_iter):
                        loss = self.nn_loss(feature_att, target, b_att)
                        loss.backward()
                        # assert torch.norm(feature_att.grad.data) != 0
                        # assert torch.norm(b_att.grad.data) != 0
                        
                        feature_att.data = feature_att.data + self.step_size_input * feature_att.grad.data.sign()
                        feature_att.data.clamp_(min = feature_min, max = feature_max)
                        b_att.data = b_att.data + self.step_size_parameter.repeat(target.shape[0], 1) * b_att.grad.data.sign()
                        b_att.data.clamp_(self.b_min, self.b_max)
                        feature_att.grad.zero_()
                        b_att.grad.zero_()
                        
                    # update the nn parameter
                    feature_att.requires_grad = False
                    b_att.requires_grad = False
                    self.net.train()
                    loss = self.nn_loss(feature_att, target, b_att)
                    if self.alpha == 1:
                        pass
                    else:
                        # clean loss
                        loss_clean = self.nn_loss(feature, target, self.b_default.repeat(feature.shape[0], 1))
                        loss = self.alpha * loss + (1 - self.alpha) * loss_clean
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), norm_type=1, max_norm = self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                elif self.train_method == "free":
                    self.net.train()
                    for pgd_no in range(self.no_iter):
                        loss = self.nn_loss(feature_att, target, b_att)
                        if self.alpha == 1:
                            pass
                        else:
                            loss_clean = self.nn_loss(feature, target, self.b_default.repeat(feature.shape[0], 1))
                            loss = self.alpha * loss + (1 - self.alpha) * loss_clean
                        loss.backward()
                        # attack
                        # assert torch.norm(feature_att.grad.data) != 0
                        # assert torch.norm(b_att.grad.data) != 0
                        feature_att.data = feature_att.data + self.step_size_input * feature_att.grad.data.sign()
                        feature_att.data.clamp_(min = feature_min, max = feature_max)
                        b_att.data = b_att.data + self.step_size_parameter.repeat(target.shape[0], 1) * b_att.grad.data.sign()
                        b_att.data.clamp_(self.b_min, self.b_max)
                        feature_att.grad.zero_()
                        b_att.grad.zero_()
                        # nn parameter
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), norm_type=1, max_norm = self.grad_clip)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                # update global variable
                feature_att_global[batch_index] = feature_att.detach()
                parameter_att_global[batch_index] = b_att.detach()
                    
            loss_sum += loss.item() * len(target)
        
        return loss_sum / len(self.trainloader.dataset)
    

if __name__ == '__main__':
    
    import json
    import argparse
    from helper import return_nn_model, return_operator
    import time
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--case_name', type = str, default = 'case14')
    parser.add_argument('-p', '--pre_train_method', type = str)
    parser.add_argument('-a', '--attack_method', type = str, choices = ['input', 'parameter', 'both'])
    parser.add_argument('-t', '--train_method', type = str, choices = ['normal', 'free'])
    parser.add_argument('-i', '--initial_method', type = str)
    parser.add_argument('--eps_input', type = float, default = 0.005)
    parser.add_argument('--eps_parameter', type = float, default = 0.1)
    parser.add_argument('--alpha', type = float)
    args = parser.parse_args()
    
    print("==============================================")
    print("Training for robust spo model...")
    print(f"Train on {args.case_name} \n warm start from {args.pre_train_method}\
        \n attack method: {args.attack_method} \n initial method: {args.initial_method} train method: {args.train_method} \
        \n eps_input: {args.eps_input} \n eps_parameter: {args.eps_parameter} \
        \n alpha: {args.alpha}")
    
    with open("config.json") as f:
        config = json.load(f)
    
    random_seed = config['random_seed']
    batch_size = config['nn']['batch_size']
    batch_size_eval = config['nn']['batch_size_eval']
    lr = config['nn'][f'lr_spo_robust_{args.attack_method}']
    epoch = config['nn'][f'epoch_spo_robust']
    model_dir = config['nn']['model_dir']
    watch = config['nn']['watch_spo']
    fix_first_b = config['fix_first_b']
    no_iter = config['attack']['no_iter']
    is_scale = config['is_scale']
    grad_clip = config['nn']['gradient_clip_norm']
    train_with_test = config['nn']['train_with_test_spo']
    
    shuffle = False if args.initial_method == 'previous' else True # disable shuffle if the init method is previous
    
    if watch == 'test':
        assert train_with_test == True 
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # data
    train_dataset = MyDataset(case_name = args.case_name, mode = "train")
    test_dataset = MyDataset(case_name = args.case_name, mode = "test")
    trainloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle)
    testloader = DataLoader(test_dataset, batch_size = batch_size_eval, shuffle = False)
    
    sample_size = len(train_dataset)
    print("Size of train dataset: {}".format(len(train_dataset)))
    print("batch size: {}".format(batch_size))
    print("Shape of feature: {}".format(train_dataset[0][0].shape))
    
    is_small_size = config['is_small_size']
    if is_small_size:
        sample_size = len(train_dataset)
    else:
        sample_size = 'full'
    
    # net
    operator = return_operator(case_name = args.case_name)
    net = return_nn_model(case_name = args.case_name, is_load = True, train_method = args.pre_train_method)
    
    if is_scale:
        mean = train_dataset.target_mean
        std = train_dataset.target_std
    else:
        mean = 0
        std = 1
    net = NN_SPO(model = net, operator=operator, mean = mean, std = std, fix_first_b = fix_first_b) # construct the spo model
    
    # optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    optimizer = torch.optim.SGD(net.parameters(), lr = lr)
    trainer = Trainer(operator = operator, net = net, optimizer = optimizer,
                    train_loader = trainloader, test_loader = testloader,
                    max_eps_input = args.eps_input, max_eps_parameter = args.eps_parameter, init_method = args.initial_method,
                    attack_method = args.attack_method, train_method = args.train_method, 
                    grad_clip = grad_clip, alpha = args.alpha)
    
    save_path = f'{model_dir}/{sample_size}/spo_robust-{args.pre_train_method}-{args.train_method}-{args.initial_method}-{args.alpha}-{args.attack_method}'
    if args.attack_method == "input":
        save_path += f'-{args.eps_input}'
    elif args.attack_method == "parameter":
        save_path += f'-{args.eps_parameter}'
    elif args.attack_method == "both":
        save_path += f'-{args.eps_input}-{args.eps_parameter}'
    
    save_path_best = save_path + '.pth'
    # save_path_last = save_path + '_last.pth'
    
    # random initialize global variable
    global feature_att_global
    global parameter_att_global
    feature_att_global = []
    parameter_att_global = []
    for feature, target in trainloader:
        if args.attack_method == 'input' or args.attack_method == "both":
            feature_min, feature_max = trainer.input_bound_clamp(feature)
            feature_att = torch.rand_like(feature_min) * (feature_max - feature_min) + feature_min
            feature_att_global.append(feature_att)
            
        if args.attack_method == 'parameter' or args.attack_method == "both":
            parameter_att_global.append(deepcopy(trainer.b_default).repeat(target.shape[0], 1))
        
    if args.train_method == "normal":
        pass
    elif args.train_method == "free":
        # reduce the number of iteration
        epoch = int(epoch / no_iter)
    
    best_loss = 1e5
    
    for i in range(1, epoch + 1):
        start_time = time.time()
        train_loss = trainer.train()
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
            torch.save(trainer.net.state_dict(), save_path_best)
            print("Best model saved by train!")
        
        # torch.save(trainer.net.state_dict(), save_path_last)
        # print("last epoch model saved!")
        
        print("==============================================")
        
    
    