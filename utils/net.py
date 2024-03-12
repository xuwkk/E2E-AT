# construct a simple convnet for load forecast
# input: (batch_size, 1, no_bus, no_feature)
# output: (batch_size, no_bus)

from torch import nn
from cvxpylayers.torch import CvxpyLayer
import torch
from torch.nn.functional import relu
import cvxpy as cp
from copy import deepcopy

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# # a feedforward neural network
def NN(feature_size = 88, output_size = 14, hidden_size = 200):
    model = nn.Sequential(
        nn.Linear(feature_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, int(hidden_size/2)),
        nn.ReLU(),
        nn.Linear(int(hidden_size/2), output_size)
    )
    model.name = "NN"
    
    return model

def NN_WITH_RELU(feature_size = 88, output_size = 14, hidden_size = 200):
    model = nn.Sequential(
        nn.Linear(feature_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, int(hidden_size/2)),
        nn.ReLU(),
        nn.Linear(int(hidden_size/2), output_size),
        nn.ReLU()
    )
    model.name = "NN"
    
    return model

class DISPATCH(nn.Module):
    """
    differentiable convex layer for the dispatch problem
    not trainable
    """
    def __init__(self, operator, fix_first_b, mean, std, solver_args = None):
        super().__init__()
        self.layer = CvxpyLayer(operator.prob_1,
                parameters = operator.prob_1.parameters(),
                variables = operator.prob_1.variables()
                )
        self.b_default = torch.from_numpy(operator.b).float()
        self.fix_first_b = fix_first_b
        self.mean = mean
        self.std = std
        self.solver_args = solver_args
        
    def forward(self, load_forecast, b = None):
        # load: forecast load which should NOT be normalized
        load_forecast = relu(load_forecast * self.std + self.mean)
        if self.fix_first_b:
            pg = self.layer(self.b_default.repeat(load_forecast.shape[0], 1), load_forecast,
                            solver_args = self.solver_args)[0]
        else:
            pg = self.layer(b, load_forecast,
                            solver_args = self.solver_args)[0]
        
        return pg

class REDISPATCH(nn.Module):
    """
    differentiable convex layer for the redispatch problem
    not trainable
    """
    def __init__(self, operator, mean, std, solver_args):
        super().__init__()
        self.layer = CvxpyLayer(operator.prob_2,
                parameters = operator.prob_2.parameters(),
                variables = operator.prob_2.variables()
                )
        self.mean = mean
        self.std = std
        self.solver_args = solver_args
        
    def forward(self, pg, load_true, b):
        # load_shedding, generator_storage
        load_true = load_true * self.std + self.mean
        output = self.layer(b, pg, load_true, solver_args = self.solver_args)
        return output[0], output[1]

class SPO(nn.Module):
    """
    sequential differentiable convex layer for both dispatch and redispatch problems
    not trainable
    this layer is not trainable
    """
    def __init__(self, operator, fix_first_b, mean, std, solver_args):
        super().__init__()
        self.dispatch = DISPATCH(operator, fix_first_b, mean, std, solver_args=solver_args)
        self.redispatch = REDISPATCH(operator, mean, std, solver_args=solver_args)
    
    def forward(self, load_forecast, load_true, b):
        pg = self.dispatch(load_forecast, b)
        ls, gs = self.redispatch(pg, load_true, b)
        return pg, ls, gs

class NN_SPO(nn.Module):
    # model for one-stage training
    # the target is not scaled
    def __init__(self, model, operator, mean, std, fix_first_b, solver_args = None):
        
        super().__init__()
        self.name = "NN_SPO"
        self.mean = mean
        self.std = std
        self.nn_model = deepcopy(model)
        self.spo = SPO(operator, fix_first_b, mean, std, solver_args)
        
    def forward(self, feature, target, b):
        # brief introduction of the model
        # feature -> NN -> ReLU -> stage_one_layer -> stage_two_layer -> decision -> loss
        #                             ^                    ^
        #                             |                    |
        #                         susceptance       susceptance, load_true
        # add relu so that the output is non-negative for forcasted load
        
        # forecast
        forecast_load = self.nn_model(feature) # (batch_size, no_bus) 
        # dispatch and redispatch NOTE: the target will be normalized in redispatch
        pg, ls, gs = self.spo(forecast_load, target, b)
        
        return forecast_load, pg, ls, gs

def bound_propagation(model, initial_bound):
    """
    using bound propagation to get the bounds of the output
    modified from https://adversarial-ml-tutorial.org/adversarial_examples/
    """
    l, u = initial_bound
    bounds = []
    
    for layer in model:
        if isinstance(layer, Flatten):
            l_ = Flatten()(l)
            u_ = Flatten()(u)
        elif isinstance(layer, nn.Linear):
            l_ = (layer.weight.clamp(min=0) @ l.t() + layer.weight.clamp(max=0) @ u.t() 
                + layer.bias[:,None]).t()
            u_ = (layer.weight.clamp(min=0) @ u.t() + layer.weight.clamp(max=0) @ l.t() 
                + layer.bias[:,None]).t()
        elif isinstance(layer, nn.Conv2d):
            l_ = (nn.functional.conv2d(l, layer.weight.clamp(min=0), bias=None, 
                                    stride=layer.stride, padding=layer.padding,
                                    dilation=layer.dilation, groups=layer.groups) +
                nn.functional.conv2d(u, layer.weight.clamp(max=0), bias=None, 
                                    stride=layer.stride, padding=layer.padding,
                                    dilation=layer.dilation, groups=layer.groups) +
                layer.bias[None,:,None,None])
            
            u_ = (nn.functional.conv2d(u, layer.weight.clamp(min=0), bias=None, 
                                    stride=layer.stride, padding=layer.padding,
                                    dilation=layer.dilation, groups=layer.groups) +
                nn.functional.conv2d(l, layer.weight.clamp(max=0), bias=None, 
                                    stride=layer.stride, padding=layer.padding,
                                    dilation=layer.dilation, groups=layer.groups) + 
                layer.bias[None,:,None,None])
            
        elif isinstance(layer, nn.ReLU):
            l_ = l.clamp(min=0)
            u_ = u.clamp(min=0)
            
        bounds.append((l_, u_))
        l,u = l_, u_
    return bounds

def form_milp(model, initial_bounds):
    """
    form an MILP problem to minimize the forecast load (which can trigger load shedding)
    so that the overall cost can be significantly increased
    """
    
    # bound propagation
    bounds = bound_propagation(model, initial_bounds)
    linear_layers = [(layer, bound) for layer, bound in zip(model,bounds) if isinstance(layer, nn.Linear)]
    d = len(linear_layers)-1
    
    # create cvxpy variables
    z = ([cp.Variable(layer.in_features) for layer, _ in linear_layers] + 
            [cp.Variable(linear_layers[-1][0].out_features)])
    v = [cp.Variable(layer.out_features, boolean=True) for layer, _ in linear_layers]
    
    # extract relevant matrices
    W = [layer.weight.detach().cpu().numpy() for layer,_ in linear_layers]
    b = [layer.bias.detach().cpu().numpy() for layer,_ in linear_layers]
    l = [l[0].detach().cpu().numpy() for _, (l,_) in linear_layers]
    u = [u[0].detach().cpu().numpy() for _, (_,u) in linear_layers]
    l0 = initial_bounds[0][0].view(-1).detach().cpu().numpy()
    u0 = initial_bounds[1][0].view(-1).detach().cpu().numpy()
    
    # add ReLU constraints
    constraints = []
    for i in range(len(linear_layers)):
        constraints += [z[i+1] >= W[i] @ z[i] + b[i], 
                        z[i+1] >= 0,
                        cp.multiply(v[i], u[i]) >= z[i+1],
                        W[i] @ z[i] + b[i] >= z[i+1] + cp.multiply((1-v[i]), l[i])]
        
    constraints += [z[0] >= l0, z[0] <= u0]
    
    return cp.Problem(cp.Maximize(cp.sum(z[d+1])), constraints), z