"""
cvxpy fomulation for generator scheduling problem
"""
import numpy as np
from pypower.api import ext2int, makeBdc, bustypes
from pypower.idx_brch import RATE_A, RATE_B, RATE_C, BR_X
from pypower.idx_bus import PD
from pypower.idx_gen import PMAX, PMIN
import cvxpy as cp
import torch

class Operator:
    
    def __init__(self, case, theta_slack = 0, tight_constraint = 0):
        
        self.baseMVA = case["baseMVA"]
        self.no_bus = case['bus'].shape[0]
        self.no_gen = case['gen'].shape[0]  
        self.no_branch = case['branch'].shape[0]
        self.no_load = np.sum(case['bus'][:, PD] > 0)
        
        self.no_brh = case['branch'].shape[0]
        self.first_coeff = np.array(case["first_coeff"])
        self.load_shed_coeff = case["load_shed_coeff"]
        self.gen_storage_coeff = case["gen_storage_coeff"]
        
        self.tight_constraint = tight_constraint
        
        # ! change the load and storage coeff into vector
        np.random.seed(0)
        # self.load_shed_coeff = (0.9 + 0.2 * np.random.rand(self.no_load)) * self.load_shed_coeff
        # self.gen_storage_coeff = (0.9 + 0.2 * np.random.rand(self.no_gen)) * self.gen_storage_coeff
        
        self.load_shed_coeff = np.ones(self.no_load) * self.load_shed_coeff
        self.gen_storage_coeff = np.ones(self.no_gen) * self.gen_storage_coeff
        
        self.default_load = case['bus'][:, PD] / self.baseMVA
        self.pg_max = case["gen"][:, PMAX] / self.baseMVA
        self.pf_max = case["branch"][:, RATE_A] / self.baseMVA
        
        # branch susceptance
        self.b = 1/np.array(case['branch'][:, BR_X])
        
        # matrices
        case_int = ext2int(case)
        self.ref_index, _, _ = bustypes(case_int['bus'], case_int['gen'])
        
        # generator incidence matrix
        self.Cg = np.zeros((self.no_bus,self.no_gen))
        for i in range(self.no_gen):
            self.Cg[int(case_int['gen'][i,0]),i] = 1
        
        # load incidence matrix
        self.Cl = np.zeros((self.no_bus,self.no_load))
        load_bus_index = np.where(case_int['bus'][:, PD] > 0)[0]
        for i in range(self.no_load):
            self.Cl[load_bus_index[i],i] = 1
        
        self.Bbus, self.Bf, self.Pbusinj, self.Pfinj = makeBdc(case_int['baseMVA'], case_int['bus'], case_int['branch'])
        if np.all(case['bus'][:, 4:6] == 0) and np.all(case['branch'][:, 4] == 0) and np.all(case['branch'][:, 8:10] == 0):
            assert np.all(self.Pbusinj == 0)
            assert np.all(self.Pfinj == 0)
        
        self.Bbus = self.Bbus.todense()
        self.Bf = self.Bf.todense()
        # convert matrix to array
        self.Bbus = np.array(self.Bbus)
        self.Bf = np.array(self.Bf)
        
        # Branch Incidence Matrix
        f_bus = case_int['branch'][:, 0].astype('int')        # list of "from" buses
        t_bus = case_int['branch'][:, 1].astype('int')        # list of "to" buses
        self.Cf = np.zeros((self.no_brh,self.no_bus))         # "from" bus incidence matrix
        self.Ct = np.zeros((self.no_brh,self.no_bus))         # "to" bus incidence matrix
        for i in range(self.no_brh):
            self.Cf[i,f_bus[i]] = 1
            self.Ct[i,t_bus[i]] = 1
        
        self.A = self.Cf-self.Ct
        
        # check the relationship
        assert np.all(self.A.T@self.Bf == self.Bbus), "the relationship between Bbus and Bf is not correct"
        assert np.all(self.Bf == np.diag(self.b) @ self.A), "the relationship on the Bf"

        self.theta_slack = theta_slack

        # define problems
        
        self.prob_1 = self.stage_one()
        self.prob_2 = self.stage_two()
        assert self.prob_1.is_dpp()  # dpp helps for repeat solving the same problem with varying parameters
        assert self.prob_2.is_dpp()
        
    def stage_one(self):
        """
        construct the one-stage scheduling problem, the operator dispatches the generator based on the forecasted load
        param:
            d: the forecasted load (unscaled in MW)
            theta_slack: the default slack bus angle
        return:
            a cvxpy problem
        """
        
        # parameters: the forecasted load is a parameter
        # the branch susceptance is a parameter to cover the uncertainty
        # other parameters are fixed (constant)
        d = cp.Parameter(self.no_load, name = 'load')         # load forecast
        b = cp.Parameter(self.no_branch, name = 'susceptance')       # branch susceptance
        
        ls = cp.Variable(self.no_load)          # load shedding (slack variable for feasibility)
        pg = cp.Variable(self.no_gen)           # generation
        theta = cp.Variable(self.no_bus)       # voltage angle
        
        # constraints
        constraints = []
        constraints += [pg <= self.pg_max, pg >= 0] # generation limit
        constraints += [ cp.diag(b) @ self.A @ theta <= self.pf_max, cp.diag(b) @ self.A @ theta >= -self.pf_max] # branch flow limit
        constraints += [self.A.T @ (cp.diag(b) @ self.A) @ theta == self.Cg @ pg - self.Cl@(d - ls)] # power balance
        constraints += [ls >= 0]
        constraints += [theta[self.ref_index] == self.theta_slack] # slack bus angle            
        objective = cp.Minimize(
                    # first order + load shedding
                    # cp.scalar_product(self.first_coeff, pg) + self.load_shed_coeff * cp.sum(ls)
                    cp.scalar_product(self.first_coeff, pg) + cp.scalar_product(self.load_shed_coeff, ls)
                    )
        prob = cp.Problem(objective, constraints)
        
        return prob
        
    def stage_two(self):
        d = cp.Parameter(self.no_load, name = 'load')                 # ground-truth load
        b = cp.Parameter(self.no_branch, name = 'susceptance')        # branch susceptance
        pg = cp.Parameter(self.no_gen, name = 'generation')           # generation (passed from stage one)
        ls = cp.Variable(self.no_load)          # load shedding
        gs = cp.Variable(self.no_gen)           # generation storage
        theta = cp.Variable(self.no_bus)        # voltage angle
        
        # constraints
        constraints = []
        constraints += [ cp.diag(b) @ self.A @ theta <= self.pf_max, cp.diag(b) @ self.A @ theta >= -self.pf_max] # branch flow limit
        constraints += [self.A.T @ (cp.diag(b) @ self.A) @ theta == self.Cg @ (pg-gs) - self.Cl@(d - ls)] # power balance
        constraints += [ls >= 0]
        constraints += [gs >= 0]
        constraints += [theta[self.ref_index] == self.theta_slack] # slack bus angle
        objective = cp.Minimize(
            # generation storage + load shedding
            # self.load_shed_coeff * cp.sum(ls) + self.gen_storage_coeff * cp.sum(gs)
            cp.scalar_product(self.load_shed_coeff, ls) + cp.scalar_product(self.gen_storage_coeff, gs)
        )
        
        prob = cp.Problem(objective, constraints)
        
        return prob

    def stage_one_decision(self):
        """
        denote the forecast load as decision variables while susceptance by default value
        """
        ls = cp.Variable(self.no_load)                     # load shedding (slack variable for feasibility)
        pg = cp.Variable(self.no_gen, name = "pg")         # generation
        theta = cp.Variable(self.no_bus)                   # voltage angle
        forecast_load = cp.Variable(self.no_load, name = "parameter") # ! mark as parameter
        
        constraints = []
        constraints += [pg <= self.pg_max, pg >= 0] # generation limit
        constraints += [ cp.diag(self.b) @ self.A @ theta <= self.pf_max, cp.diag(self.b) @ self.A @ theta >= -self.pf_max] # branch flow limit
        constraints += [self.A.T @ (cp.diag(self.b) @ self.A) @ theta == self.Cg @ pg - self.Cl@(forecast_load - ls)] # power balance
        constraints += [ls >= 0]
        constraints += [theta[self.ref_index] == self.theta_slack] # slack bus angle            

        objective = cp.Minimize(
                    # first order + load shedding
                    # cp.scalar_product(self.first_coeff, pg) + self.load_shed_coeff * cp.sum(ls)
                    cp.scalar_product(self.first_coeff, pg) + cp.scalar_product(self.load_shed_coeff, ls)
                    )
        prob = cp.Problem(objective, constraints)
        
        # self.ineq_index = [0, 1, 2, 3, 5]
        # self.eq_index = [4, 6, 7]
        
        return prob
    
    def stage_two_decision(self, true_load_value):
        
        ls = cp.Variable(self.no_load, name = "ls")          # load shedding
        gs = cp.Variable(self.no_gen, name = 'gs')           # generation storage
        theta = cp.Variable(self.no_bus)        # voltage angle

        pg = cp.Variable(self.no_gen, name = "parameter") # ! mark as parameter
        
        # constraints
        constraints = []
        constraints += [self.A.T @ (cp.diag(self.b) @ self.A) @ theta == self.Cg @ (pg-gs) - self.Cl@(true_load_value - ls)] # power balance
        constraints += [ cp.diag(self.b) @ self.A @ theta <= self.pf_max, cp.diag(self.b) @ self.A @ theta >= -self.pf_max] # branch flow limit
        constraints += [ls >= 0]
        constraints += [gs >= 0]
        constraints += [theta[self.ref_index] == self.theta_slack] # slack bus angle
        
        # if pg_value is not None:
        #     constraints += [pg == pg_value]
        
        objective = cp.Minimize(
            # generation storage + load shedding
            # self.load_shed_coeff * cp.sum(ls) + self.gen_storage_coeff * cp.sum(gs)
            cp.scalar_product(self.load_shed_coeff, ls) + cp.scalar_product(self.gen_storage_coeff, gs)
        )
        
        prob = cp.Problem(objective, constraints)
        
        # self.ineq_index = [0, 1, 3, 4]
        # self.eq_index = [2, 5]
        
        return prob

    def solve_one(self, load_forecast, b = None):
        """
        for dpp, the problem can be solved repeatedly with different parameters (load_forecast) 
            faster than solving it from scratch each time
        """
        # assert len(load_forecast.shape) == 1, "load_forecast should be a vector"
        load_forecast = load_forecast.numpy() if isinstance(load_forecast, torch.Tensor) else load_forecast
        if b is None:
            b = [self.b] * load_forecast.shape[0]     
        assert np.all(load_forecast >= 0), "negative load detected"
        
        # print(load_forecast)
        # print(self.prob.parameters())
        # load_forecast = load_forecast / self.baseMVA
        
        load_forecast = load_forecast.reshape(-1, self.no_load)

        # allow batch solving
        p,s,theta,cost,status = [],[],[],[],[]
        for load_forecast_, b_ in zip(load_forecast, b):
            # assign the parameters
            for parameter in self.prob_1.parameters():
                if parameter.name() == 'load':
                    parameter.value = load_forecast_
                elif parameter.name() == 'susceptance':
                    parameter.value = b_
                else:
                    raise ValueError("unknown parameter")

            # self.prob_1.parameters()[0].value = load_forecast_
            
            # self.prob.parameters()[0].value = load_forecast_
            # self.prob.parameters()[1].value = b
            try:
                self.prob_1.solve(solver = cp.GUROBI, verbose = False)
            except:
                self.prob_1.solve(verbose = False)
        
            p.append(self.prob_1.variables()[0].value)
            s.append(self.prob_1.variables()[1].value)
            theta.append(self.prob_1.variables()[2].value)
            cost.append(self.prob_1.value)
            status.append(self.prob_1.status)
        
        return np.array(p), np.array(s), np.array(theta), np.array(cost), np.array(status)
    
    def solve_two(self, load_true, pg, b = None):
        load_true = load_true.numpy() if isinstance(load_true, torch.Tensor) else load_true
        pg = pg.numpy() if isinstance(pg, torch.Tensor) else pg
        if b is None:
            b = [self.b] * load_true.shape[0] 

        assert np.all(load_true >= 0), "negative load detected"
        assert np.all(pg >= 0), "negative generation detected"
        
        load_true = load_true.reshape(-1, self.no_load)
        pg = pg.reshape(-1, self.no_gen)
        
        # allow batch solving
        ls, gs, theta, cost, status = [],[],[],[],[]
        for load_true_, pg_, b_ in zip(load_true, pg, b):
            for parameter in self.prob_2.parameters():
                if parameter.name() == 'load':
                    parameter.value = load_true_
                elif parameter.name() == 'generation':
                    parameter.value = pg_
                elif parameter.name() == 'susceptance':
                    parameter.value = b_
                else:
                    raise ValueError("unknown parameter")
            
            try:
                self.prob_2.solve(solver = cp.GUROBI, verbose = False)
            except:
                self.prob_2.solve(verbose = False)
            ls.append(self.prob_2.variables()[0].value)
            gs.append(self.prob_2.variables()[1].value)
            theta.append(self.prob_2.variables()[2].value)
            cost.append(self.prob_2.value)
            status.append(self.prob_2.status)
        
        return np.array(ls), np.array(gs), np.array(theta), np.array(cost), np.array(status)
                    
    def loss(self, pg, ls, gs):
        
        # pg (batch_size, no_gen)
        # loss = generator cost (from stage one) + load shedding cost (from stage two) + generation storage cost (from stage two)
        
        loss = (pg @ np.array(self.first_coeff)[:, None]).flatten() + self.load_shed_coeff * ls.sum(axis=1) + self.gen_storage_coeff * gs.sum(axis=1)
        
        return loss.flatten()