import json
import os
import torch
import numpy as np
import cvxpy as cp
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.net import NN, NN_SPO, NN_WITH_RELU, SPO
from utils.dataset import case_modifier, MyDataset
from utils.optimization import Operator
from utils.robustness import PGD

def return_operator(case_name = "case14"):
    """
    return the operator for the given case
    """
    case = case_modifier(case_name = case_name)
    operator = Operator(case)
    return operator

def return_nn_model(is_load, train_method = None, **kwargs):
    """
    ! load the trained nn model WITHOUT the optimization layers
    train_method: the training method used
    """
    
    with open("config.json") as f:
        config = json.load(f)   

    feature_size = config['nn']['feature_size']
    output_size = config['nn']['output_size']
    hidden_size = config['nn']['hidden_size']
    is_small_size = config['is_small_size']
    if is_small_size:
        sample_size = config['small_size']
    else:
        sample_size = 'full'
    
    if 'with_relu' in kwargs.keys():
        net = NN_WITH_RELU(feature_size=feature_size, output_size=output_size, hidden_size=hidden_size)
    else:
        net = NN(feature_size=feature_size, output_size=output_size, hidden_size=hidden_size)
    
    if is_load:
        save_dir = f"{config['nn']['model_dir']}/{sample_size}/{train_method}.pth"
        assert os.path.exists(save_dir), f'the nn model file {save_dir} does not exist, please train the model first.'
        print(f'Loading model from {save_dir}')    
        if "spo" in train_method:
            # change the name of the saved model state
            state = torch.load(save_dir)
            assert len([param for param in net.parameters()]) == len(state.keys()), "number of parameters does not match"
            # change the name of the parameters
            for name, param in net.named_parameters():
                param.data = state["nn_model." + name]
        else:
            net.load_state_dict(torch.load(save_dir))
    
    return net

def generator_loss(pg, ls, gs, operator):
    first_coeff = torch.tensor(operator.first_coeff, dtype=torch.float)
    load_shed_coeff = torch.tensor(operator.load_shed_coeff, dtype = torch.float)
    gen_storage_coeff = torch.tensor(operator.gen_storage_coeff, dtype = torch.float)
    loss = pg @ first_coeff + ls @ load_shed_coeff + gs @ gen_storage_coeff
        
    return loss

def choose_best_attack(cost_att_all, quantity_att_all):
        # choose the best attack during the multi-runs
        cost_att_all = np.array(cost_att_all)
        att_index = np.argmax(cost_att_all, axis = 0) # the index of the best attack for each samlpe
        quantity_att_all = torch.stack(quantity_att_all, axis = 1)
        quantity_att = []
        for i in range(len(att_index)):
            quantity_att.append(quantity_att_all[i][att_index[i]])
        
        return torch.stack(quantity_att,axis=0)

class EVALUATOR_SPO:
    """
    given load and b, evaluate the performance of the convex layer
    """
    def __init__(self, case_name):
        
        with open("config.json") as f:
            config = json.load(f)
        
        batch_size = config['nn']['batch_size_eval']
        fix_first_b = config['fix_first_b']
        feature_size = config['nn']['feature_size']
        operator = return_operator(case_name = case_name)
        
        self.train_dataset = MyDataset(case_name = case_name, mode = "train")
        self.test_dataset = MyDataset(case_name = case_name, mode = "test")
        self.train_loader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = True)
        self.test_loader = DataLoader(self.test_dataset, batch_size = batch_size, shuffle = True)
        
        self.net_spo = SPO(operator = operator, fix_first_b = fix_first_b)
        self.net_spo.eval()
        
        self.no_iter = config['attack']['no_iter_eval']    # number of pgd step
        self.multirun_no = config['attack']['multirun_no'] # number of multiruns to generate the worst attack
    
    def spo_loss(self, pg, ls, gs):
        return (pg @ self.first_coeff + ls @ self.load_shed_coeff + gs @ self.gen_storage_coeff).mean()

class EVALUATOR:
    """
    evaluate the performance on a given trained network
    """
    
    def __init__(self, case_name, train_method):
        """
        train_method: the training method used
        """
        
        with open("config.json") as f:
            config = json.load(f)
        
        batch_size = config['nn']['batch_size_eval']
        fix_first_b = config['fix_first_b']
        feature_size = config['nn']['feature_size']
        operator = return_operator(case_name = case_name)
        
        self.train_dataset = MyDataset(case_name = case_name, mode = "train")
        self.test_dataset = MyDataset(case_name = case_name, mode = "test")
        self.train_loader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = True)
        self.test_loader = DataLoader(self.test_dataset, batch_size = batch_size, shuffle = True)
        
        # trained nn
        self.net = return_nn_model(case_name = case_name, is_load = True, train_method = train_method)
        self.net.eval()
        
        # concate the optimization layer
        is_scale = config['is_scale']
        if is_scale:
            mean = self.train_dataset.target_mean
            std = self.train_dataset.target_std
        else:
            mean = 0
            std = 1
        self.net_spo = NN_SPO(model = self.net, operator = operator, mean = mean, std = std, fix_first_b = fix_first_b)
        self.net_spo.eval()
        
        self.operator = operator
        self.b_default = torch.from_numpy(operator.b).float()
        self.first_coeff = torch.tensor(operator.first_coeff, dtype=torch.float)
        self.load_shed_coeff = torch.tensor(operator.load_shed_coeff, dtype = torch.float)
        self.gen_storage_coeff = torch.tensor(operator.gen_storage_coeff, dtype = torch.float)
        
        self.is_scale = self.train_dataset.is_scale
        
        self.no_iter = config['attack']['no_iter_eval']
        self.fixed_feature = config['attack']['fixed_feature']
        self.flexible_feature = list(set(np.arange(feature_size)) - set(self.fixed_feature))
        self.multirun_no = config['attack']['multirun_no']
    
    def spo_loss(self, pg, ls, gs):
        return (pg @ self.first_coeff + ls @ self.load_shed_coeff + gs @ self.gen_storage_coeff).mean()
    
    def spo_loss_individual(self, pg, ls, gs):
        return (pg @ self.first_coeff + ls @ self.load_shed_coeff + gs @ self.gen_storage_coeff)

    def clean_mse(self, is_test = False):
        
        if not is_test:
            dataloader = self.train_loader
        else:
            dataloader = self.test_loader
        
        mse = 0
        for feature, target in tqdm(dataloader, total=len(dataloader), desc = 'clean mse'):
            with torch.no_grad():
                pred = self.net(feature)
                if self.is_scale:
                    mean = self.train_dataset.target_mean
                    std = self.train_dataset.target_std
                    target = target * std + mean
                    pred = pred * std + mean
                
                mse += torch.mean((pred - target)**2) * feature.shape[0]
        
        return mse.item() / len(dataloader.dataset)
    
    def clean_cost(self, is_test = False):
        
        if not is_test:
            dataloader = self.train_loader
        else:
            dataloader = self.test_loader
        
        cost = 0
        with torch.no_grad():
            for feature, target in tqdm(dataloader, total=len(dataloader), desc = 'clean cost'):
                pred = self.net_spo(feature, target, self.b_default.repeat(feature.shape[0], 1))
                cost += self.spo_loss(pred[1], pred[2], pred[3]).item() * feature.shape[0]
        
        return cost / len(dataloader.dataset)
    
    def clean_cost_individual(self, is_test = False):
        
        if not is_test:
            dataloader = self.train_loader
        else:
            dataloader = self.test_loader
        
        cost = []
        forecast_error = []
        with torch.no_grad():
            for feature, target in tqdm(dataloader, total=len(dataloader), desc = 'clean cost'):
                pred = self.net_spo(feature, target, self.b_default.repeat(feature.shape[0], 1))
                forecast_error.append(np.sum(pred[0].numpy(),1) - np.sum(target.numpy(),1))
                cost.append(self.spo_loss_individual(pred[1], pred[2], pred[3]).numpy().flatten())
        
        cost = np.concatenate(cost, axis = 0)
        forecast_error = np.concatenate(forecast_error, axis = 0)
        
        return cost, forecast_error
    
    def clean_cost_cvxpy(self, is_test = False):
        """
        evaluate the cost on the clean dataset using cvxpy to check the result
        """
        if not is_test:
            dataset = self.train_dataset
        else:
            dataset = self.test_dataset
        
        feature = dataset.feature.numpy()
        target = dataset.target.numpy()
        
        with torch.no_grad():
            forecast_load = self.net(torch.tensor(feature)).numpy()
        
        if dataset.is_scale:
            mean = dataset.target_mean.numpy()
            std = dataset.target_std.numpy()
            forecast_load = forecast_load * std + mean
            target = target * std + mean
        
        pg, _, _, cost1, _ = self.operator.solve_one(forecast_load)
        _, _, _, cost2, _ = self.operator.solve_two(target, pg)
        
        return np.mean(cost1 + cost2)

    
    def adv_input_mse(self, max_eps, is_test = False):
        """
        evaluate the adversarial attack on the input space targetting on the mse loss
        """
        attacker = PGD(operator = self.operator, is_spo = False, nn = self.net, attack_method='input', no_iter = self.no_iter, 
                    flexible_feature=self.flexible_feature, fixed_feature=self.fixed_feature, max_eps_input = max_eps)
        
        if not is_test:
            dataloader = self.train_loader
        else:
            dataloader = self.test_loader
        
        mse = 0
        for feature, target in tqdm(dataloader, total=len(dataloader), desc = 'attack input mse'):
            feature_att = attacker.attack(feature, target)
            
            with torch.no_grad():
                pred_att = self.net(feature_att)
                if self.is_scale:
                    mean = self.train_dataset.target_mean
                    std = self.train_dataset.target_std
                    target = target * std + mean
                    pred_att = pred_att * std + mean
                loss_att_ = torch.mean((pred_att - target)**2)
                
            mse += loss_att_ * feature.shape[0]
        
        return mse.item() / len(dataloader.dataset)
    
    def adv_input_cost(self, max_eps, is_test = False):
        """
        evaluate the adversarial attack on the input space targetting on the spo loss
        """
        attacker = PGD(operator = self.operator, is_spo = True, nn = self.net_spo, attack_method='input', no_iter = self.no_iter,
                    flexible_feature=self.flexible_feature, fixed_feature=self.fixed_feature, max_eps_input = max_eps)
        
        if not is_test:
            dataloader = self.train_loader
        else:
            dataloader = self.test_loader
        
        cost = 0
        for feature, target in tqdm(dataloader, total=len(dataloader), desc = 'attack input spo'):
            
            cost_att_all = []
            feature_att_all = []
            """
            multi run: for each instance, find the worst attack across multiple runs
            """
            for i in range(self.multirun_no):
                feature_att = attacker.attack(feature, target)
                with torch.no_grad():
                    pred_att = self.net_spo(feature_att, target, self.b_default.repeat(feature.shape[0], 1))
                    cost_att = self.spo_loss_individual(pred_att[1], pred_att[2], pred_att[3])
                    cost_att_all.append(cost_att.numpy())
                    feature_att_all.append(feature_att)
            
            feature_att = self.choose_best_attack(cost_att_all, feature_att_all)
            pred_att = self.net_spo(feature_att, target, self.b_default.repeat(feature.shape[0], 1))
            cost_att = self.spo_loss(pred_att[1], pred_att[2], pred_att[3])
            cost += cost_att.item() * feature.shape[0]
            
        return cost / len(dataloader.dataset)
    
    def adv_parameter_cost(self, max_eps, is_test = False):
        """
        evaluate the adversarial attack on the parameter space targetting on the spo loss
        """
        attacker = PGD(operator = self.operator, is_spo = True, nn = self.net_spo, attack_method='parameter', no_iter = self.no_iter, max_eps_parameter = max_eps)
        if not is_test:
            dataloader = self.train_loader
        else:
            dataloader = self.test_loader
        cost = 0
        for feature, target in tqdm(dataloader, total=len(dataloader), desc = 'attack parameter spo'):
            cost_att_all = []
            b_att_all = []
            for i in range(self.multirun_no):
                b_att = attacker.attack(feature, target)
                with torch.no_grad():
                    pred_att = self.net_spo(feature, target, b_att)
                    cost_att_infividual = self.spo_loss_individual(pred_att[1], pred_att[2], pred_att[3])
                    cost_att_all.append(cost_att_infividual.numpy())
                    b_att_all.append(b_att)
            # choose the best attack
            b_att = self.choose_best_attack(cost_att_all, b_att_all)
            # evaluate the cost
            pred_att = self.net_spo(feature, target, b_att)
            cost_att = self.spo_loss(pred_att[1], pred_att[2], pred_att[3])
            cost += cost_att.item() * feature.shape[0]
        
        return cost / len(dataloader.dataset)
    
    def adv_pgd_both_cost(self, max_eps_input, max_eps_parameter, is_test = False):
        """
        evaluate the adversarial attack on both the input space and the parameter space targetting on the spo cost
        """
        attacker = PGD(operator = self.operator, is_spo = True, nn = self.net_spo, attack_method='both', no_iter = self.no_iter,
                    flexible_feature=self.flexible_feature, fixed_feature=self.fixed_feature, max_eps_input = max_eps_input, max_eps_parameter = max_eps_parameter)
        
        if not is_test:
            dataloader = self.train_loader
        else:
            dataloader = self.test_loader
        
        cost = 0
        for feature, target in tqdm(dataloader, total=len(dataloader), desc = 'attack both spo'):
            
            cost_att_all = []
            quantity_att_all = []
            
            for i in range(self.multirun_no):
                feature_att, b_att = attacker.attack(feature, target)
            
                with torch.no_grad():
                    pred_att = self.net_spo(feature_att, target, b_att)
                    cost_att = self.spo_loss_individual(pred_att[1], pred_att[2], pred_att[3])
                    cost_att_all.append(cost_att.numpy())
                    quantity_att_all.append(torch.cat([feature_att, b_att], dim = 1))
            
            quantity_att = self.choose_best_attack(cost_att_all, quantity_att_all)
            feature_att = quantity_att[:, :feature.shape[1]]
            b_att = quantity_att[:, feature.shape[1]:]
            pred_att = self.net_spo(feature_att, target, b_att)
            cost_att = self.spo_loss(pred_att[1], pred_att[2], pred_att[3])    
            cost += cost_att.item() * feature.shape[0]
        
        return cost / len(dataloader.dataset)




# def evaluate(case_name, model, dataset, is_average = True):    
    
#     case = case_modifier(case_name = case_name)
#     operator = Operator(case=case) # the operator is used to solve the power system operation problem
    
#     feature = dataset.feature.numpy()
#     target = dataset.target.numpy()
    
#     model.eval()
#     with torch.no_grad():
#         pred = model(torch.tensor(feature)).numpy()
        
#     if dataset.is_scale:
#         mean = dataset.target_mean.numpy()
#         std = dataset.target_std.numpy()
#         target = target * std + mean
#         pred = pred * std + mean

#     metric_dict = {}
    
#     pg, _, _, cost1, _ = operator.solve_one(pred)
#     _, _, _, cost2, _ = operator.solve_two(target, pg)

#     if is_average:
#         mse = np.mean((pred - target)**2)
#         cost = np.mean(cost1 + cost2)
#     else:
#         mse = np.mean((pred - target)**2, axis = 1)
#         cost = cost1 + cost2
    
#     metric_dict['mse'] = mse
#     metric_dict['cost'] = cost
#     metric_dict['gen_mismatch'] = pg.sum(axis = 1) - target.sum(axis = 1)  
    
#     return metric_dict

# def evaluate_spo(case_name, model, dataset, is_average = True):    
    
#     case = case_modifier(case_name = case_name)
#     operator = Operator(case=case) # the operator is used to solve the power system operation problem
#     b_default = torch.from_numpy(operator.b).float()
    
#     feature = dataset.feature
#     target = dataset.target
#     model.eval()
#     with torch.no_grad():
#         pred = model(feature, target, b_default.repeat(feature.shape[0], 1))
    
#     target = target.numpy()
    
#     if dataset.is_scale:
#         target = target * dataset.target_std.numpy() + dataset.target_mean.numpy()
    
#     forecast_load = pred[0].numpy()
#     pg = pred[1].numpy()
#     ls = pred[2].numpy()
#     gs = pred[3].numpy()
#     cost = pg @ np.array(operator.first_coeff)[:, None] + ls @ operator.load_shed_coeff + gs @ operator.gen_storage_coeff
    
#     metric_dict = {}
    
#     cost = pg @ operator.first_coeff + ls @ operator.load_shed_coeff + gs @ operator.gen_storage_coeff
    
#     if is_average:
#         mse = np.mean((forecast_load - target)**2)
#         cost = np.mean(cost)
#     else:
#         mse = np.mean((forecast_load - target)**2, axis = 1)
#         cost = cost
    
#     metric_dict['mse'] = mse
#     metric_dict['cost'] = cost
#     metric_dict['gen_mismatch'] = pg.sum(axis = 1) - target.sum(axis = 1)
    
#     return metric_dict


    

# def evaluate_input_mse():
#     """
#     evaluate the input space attack targetting on mse
#     """
#     pass

# def evaluate_input_spo():
#     """
#     evaluate the input space attack targetting on
#     """
#     pass


# def kkt(prob, ineq_index, eq_index):
#     data, chain, inverse_data = prob.get_problem_data(solver = cp.GUROBI)
#     Q = data['P'].todense()
#     q = data['q']
#     A = data['F'].todense()
#     b = data['G']
#     G = data['A'].todense()
#     h = data['b']
    
#     prob.solve(solver = cp.GUROBI, verbose = False)
#     x = []
#     for i in range(len(prob.variables())):
#         x += prob.variables()[i].value.tolist()

#     ineq_multiplier = []
#     for i in ineq_index:
#         ineq_multiplier += prob.constraints[i].dual_value.tolist()
#     eq_multiplier = []
#     for i in eq_index:
#         eq_multiplier += prob.constraints[i].dual_value.tolist()

#     ineq_multiplier = np.array(ineq_multiplier)
#     eq_multiplier = np.array(eq_multiplier)
    
#     # inequality constraints
#     assert np.all(A @ x - b <= 0)
#     # equality constraints
#     assert np.allclose(G @ x - h, 0)
#     # test the complementarity condition
#     assert np.all(np.diag(ineq_multiplier) @ (A @ x - b).T == 0)
#     # lambda
#     assert np.all(np.array(ineq_multiplier) >= 0)
#     # stationary
#     assert np.allclose(Q @ x + q + A.T @ ineq_multiplier + G.T @ eq_multiplier, 0)
    
# def solve_kkt(prob, ineq_index, eq_index):
    
#     # extract the matrices in standard form
#     data, chain, inverse_data = prob.get_problem_data(solver = cp.GUROBI)
#     Q = data['P'].todense()
#     q = data['q']
#     A = data['F'].todense()
#     b = data['G']
#     G = data['A'].todense()
#     h = data['b']
    
#     M = 1e6 # big M method
    
#     phi = cp.Variable(A.shape[0], integer = True)
#     x = cp.Variable(Q.shape[0])
#     ineq_multiplier = cp.Variable(A.shape[0], nonneg = True)
#     eq_multiplier = cp.Variable(G.shape[0])
    
#     constraints = []
#     # stationarity
#     constraints += [Q @ x + q + A.T @ ineq_multiplier + G.T @ eq_multiplier == 0]
#     # equality
#     constraints += [G @ x - h == 0]
#     # big-M reformualation of the complementarity condition
#     constraints += [A @ x - b <= 0, 
#                     ineq_multiplier <= M * phi,
#                     A @ x - b >= (phi - 1) * M]
    
#     prob = cp.Problem(cp.Minimize(0), constraints)
    
#     return prob
    
    

# def return_opt_data(prob):
    
#     data, chain, inverse_data = prob.get_problem_data(solver = cp.GUROBI)
    
#     Q = data['P'].todense()
#     q = data['q']
#     A = data['A'].todense()
#     b = data['b']
#     G = data['F'].todense()
#     h = data['G']
    
#     return Q, q, A, b, G, h