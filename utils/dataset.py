import torch
from torch.utils.data import Dataset, DataLoader
from os import listdir
import pandas as pd
import numpy as np
from pypower.api import case14, case9
import json
from pypower.idx_bus import PD
from pypower.idx_brch import RATE_A, RATE_B, RATE_C, BR_X
import json
import random

def case_modifier(case_name):
    """modify the default system settings in the pypower case file"""
    
    # generate the case
    assert case_name in ["case14", "case9"], "case name not found"
    if case_name == "case14":
        case = case14()
    elif case_name == "case9":
        case = case9()
    
    # configuration
    with open("config.json", "r") as f:
        config = json.load(f)
    case_config = config[case_name]

    # rescale the load
    pd = case['bus'][:, PD] * case_config["load_scale"]
    # add load to the zero-load bus
    average_pd = np.mean(pd)
    max_pd = np.max(pd)
    for i in range(len(pd)):
        if pd[i] == 0:
            pd[i] = np.min([average_pd * np.sqrt(i + 1), max_pd * 0.8]) # no randomness here
    
    # modify the case file
    case['bus'][:,PD] = pd
    case['branch'][:,RATE_A] = np.array(case_config["RATE_A"]) if case_config["RATE_A"] != None else case['branch'][:,RATE_A]
    case['branch'][:,RATE_B] = case_config["RATE_A"] if case_config["RATE_A"] != None else case['branch'][:,RATE_A]
    case['branch'][:,RATE_C] = case_config["RATE_A"] if case_config["RATE_A"] != None else case['branch'][:,RATE_A]

    # load shedding > over-generation > first order
    case["first_coeff"] = case_config["first_coeff"]  # cost of first order in $/MW
    max_first_coeff = np.max(case["first_coeff"])
    case['load_shed_coeff'] = max_first_coeff * case_config["load_shed_coeff_scale"] # cost of load shedding in $/MW  (very large)
    case['gen_storage_coeff'] =  max_first_coeff * case_config["gen_storage_coeff_scale"] # cost of over-generation
    
    if case_config["shunt_tap_linecharge"] == False:
        case['bus'][:, 4:6] = 0
        case['branch'][:, 4] = 0
        case['branch'][:, 8:10] = 0
    
    return case

class MyDataset(Dataset):
    
    def __init__(self, case_name = "case14", mode = 'train'):
        
        """
        param:
            case_name: the name of the case file, note that we need to rescale the load so that it is suitable for the power flow
            mode: train or test
        """
        
        with open("./config.json") as f:
            config = json.load(f)
        
        self.is_scale = config['is_scale'] # scale the target or not
        is_small_size = config['is_small_size']        # use small size data for debugging
        small_size = config['small_size']              # the size of the small size data
        random_seed = config['random_seed']      # random seed
        is_flatten = config['is_flatten']        # flatten the feature or not
        exp_dim = config['exp_dim']              # expand the dimension of the feature or not
        train_prop = config['train_prop']        # the proportion of the training data
        
        if is_flatten:
            assert exp_dim == False, "flatten and exp_dim cannot be both true"
        
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        case = case_modifier(case_name)
        load_standard = case["bus"][:,PD]   
        load_standard = load_standard[load_standard > 0]
        self.no_load = len(load_standard)
        
        DATA_DIR = f'data/data_{case_name}/'
        FILE_NAME = listdir(DATA_DIR)
        FILE_NAME = sorted(FILE_NAME, key = lambda x: int(x.split('_')[1].split('.')[0]))[:self.no_load]
        
        TRAIN_DATA = []
        for file_name in FILE_NAME:
            TRAIN_DATA.append(pd.read_csv(DATA_DIR + file_name).values)
        
        # target (ground truth load)
        load = []
        for i in range(len(FILE_NAME)):
            load.append(TRAIN_DATA[i][:, -1])
        
        target = np.array(load).T  # (sample, no_bus)
        # rescale the load by the standard load, so that the max load equals to standard load, power flow is feasible
        target = target / target.max(axis = 0)[None, :] * load_standard[None, :]
        assert np.allclose(target.max(axis = 0), load_standard, atol = 1e-2), "the target is not rescaled correctly"
        target = target / case["baseMVA"] # in p.u.
        
        # feature
        NO_SAMPLE = len(target)
        NO_FEATURE = len(TRAIN_DATA[0][0]) - 1

        feature = np.zeros((NO_SAMPLE * len(FILE_NAME), NO_FEATURE))

        for i in range(len(FILE_NAME)):
            position_index = np.arange(i, NO_SAMPLE * len(FILE_NAME), len(FILE_NAME))
            feature[position_index] = TRAIN_DATA[i][:, :-1]
        
        feature = feature.reshape(NO_SAMPLE, len(FILE_NAME), NO_FEATURE)  # (sample, no_bus, no_feature)
        
        if is_flatten:
            # flatten the feature for feedforward network
            # only add calendric feature once on the flatten feature
            calendric_feature = torch.tensor(feature)[:, 0, :4]
            feature = torch.tensor(feature)[:, :, 4:].flatten(start_dim = 1)
            self.feature = torch.cat((calendric_feature, feature), dim = 1).float()
        else:
            self.feature = torch.tensor(feature).float()
        
        self.target = torch.tensor(target).float()
        
        if exp_dim:
            self.feature = self.feature[:,None,:,:] # add None to make it (batch_size, 1, no_bus, no_feature)
        
        # scale
        # the feature has already been scaled
        self.target_mean = self.target.mean(axis = 0).float()
        self.target_std = self.target.std(axis = 0).float()
        if self.is_scale:
            self.target = (self.target - self.target_mean) / self.target_std
        
        # random train test split
        train_size = int(train_prop * len(target))
        test_size = len(target) - train_size
        # random train test split
        train_index =  np.random.choice(len(target), train_size, replace = False)
        test_index = np.array(list(set(np.arange(len(target))) - set(train_index)))
        
        # print(len(train_index), len(test_index))
        
        if mode == 'train':
            self.target = self.target[train_index]
            self.feature = self.feature[train_index]
        elif mode == 'test':
            self.target = self.target[test_index]
            self.feature = self.feature[test_index]
        else:
            pass
    
        if is_small_size and mode == 'train':
            no = config['small_size']
            random_index = np.random.choice(len(self.target), no, replace = False)
            self.feature = self.feature[random_index]
            self.target = self.target[random_index]
                
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return self.feature[idx], self.target[idx] 
    