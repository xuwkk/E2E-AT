"""
funcs for certifying the input space attack (uncertainties)
"""

from copy import deepcopy  
import numpy as np
import cvxpy as cp
from utils.net import bound_propagation
from torch import nn

def input_bound_clamp(feature, max_eps_input, flexible_feature):
    """
    clamp the flexible input features
    """
    feature_min = deepcopy(feature)
    feature_max = deepcopy(feature)
    
    feature_min[:,flexible_feature] = (feature_min[:,flexible_feature] - max_eps_input).clamp(0,1)
    feature_max[:,flexible_feature] = (feature_max[:,flexible_feature] + max_eps_input).clamp(0,1) # for min-max normalization further clamp into [0,1]
    
    return feature_min, feature_max

def return_cost(feature, true_load, nn, operator):

    """
    return cost
    """

    forecast_load = nn(feature).detach().numpy()
    # print(np.sum(forecast_load))
    pg, _, _, cost1, _ = operator.solve_one(forecast_load)
    _, _, _, cost2, _ = operator.solve_two(true_load, pg)
    
    return (cost1 + cost2)[0]

def matrix_kkt(prob):
    """
    return the matrix used for kkt
    Given QP in arbitrary form, return Q, q, A, b, G, C, H, d in standard form as follows
        (1/2)x^TQx + q^Tx
    s.t.    Ax + Gz <= b
            Cx + Hz = d
    where z is the parameter
    
    prob: a cvxpy form QP problem
    """
    
    # define the extended form by viewing z as a variable
    # gurobi by default solve in standard QP form
    data, chain, inverse_data = prob.get_problem_data(solver = cp.GUROBI)
    Q_ext = data['P'].todense().A
    q_ext = data['q']
    A_ext = data['F'].todense().A
    b_ext = data['G']
    C_ext = data['A'].todense().A
    d_ext = data['b']
    
    b = b_ext
    d = d_ext
    
    # get the index of z
    z_start_index = 0
    for i in range(len(prob.variables())):
        if prob.variables()[i].name() == 'parameter':
            z_length = prob.variables()[i].size
            break
        z_start_index += prob.variables()[i].size
    z_end_index = z_start_index + z_length
    
    # Q
    Q_pre, Q_z, Q_post = np.split(Q_ext, [z_start_index, z_end_index], axis = 1)
    Q_z = np.split(Q_z, [z_start_index, z_end_index], axis = 0)[1]
    Q_variable = np.concatenate((Q_pre, Q_post), axis = 1)
    Q_variable_pre, _, Q_variable_post = np.split(Q_variable, [z_start_index, z_end_index], axis = 0)
    Q = np.concatenate((Q_variable_pre, Q_variable_post), axis = 0)
    
    # q
    q_pre, q_z, q_post = np.split(q_ext, [z_start_index, z_end_index])
    q = np.concatenate((q_pre, q_post))
    
    # A and G
    A_pre, A_z, A_post = np.split(A_ext, [z_start_index, z_end_index], axis = 1)
    G = A_z
    A = np.concatenate((A_pre, A_post), axis = 1)
    
    # C and H
    C_pre, C_z, C_post = np.split(C_ext, [z_start_index, z_end_index], axis = 1)
    H = C_z
    C = np.concatenate((C_pre, C_post), axis = 1)
    
    standard_form = {'Q': Q, 'q': q, 'A': A, 'b': b, 'G': G, 'C': C, 'd': d, 'H': H}
    
    return standard_form


def form_certify(model, initial_bounds, standard_form1, standard_form2, M):
    
    """
    neural network as mixed integer linear constaints
    """
    bounds = bound_propagation(model, initial_bounds)
    
    # form model as mixed integer linear constaints
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
    
    # initial bound constraints
    # print(z[0].shape, l0.shape, u0.shape)
    constraints += [z[0] >= l0, z[0] <= u0]
    
    """
    stage one kkt as constraints
    """
    Q1 = standard_form1['Q']
    q1 = standard_form1['q']
    A1 = standard_form1['A']
    b1 = standard_form1['b']
    G1 = standard_form1['G']
    C1 = standard_form1['C']
    d1 = standard_form1['d']
    H1 = standard_form1['H']
    
    x1 = cp.Variable(standard_form1['Q'].shape[0])
    z1 = cp.Variable(standard_form1['G'].shape[1])
    ineq_multiplier1 = cp.Variable(standard_form1['A'].shape[0], nonneg = True)
    eq_multiplier1 = cp.Variable(standard_form1['C'].shape[0])
    phi1 = cp.Variable(standard_form1['A'].shape[0], integer = True)
    
    # stationarity
    constraints += [Q1 @ x1 + q1 + A1.T @ ineq_multiplier1 + C1.T @ eq_multiplier1 == 0]
    # equality
    constraints += [C1 @ x1 + H1 @ z1 - d1 == 0]
    # big-M reformualation of the complementarity condition
    constraints += [A1 @ x1 + G1 @ z1 - b1 <= 0]
    constraints += [ineq_multiplier1 <= cp.multiply(M, phi1),
                    A1 @ x1 + G1 @ z1 - b1 >= cp.multiply((phi1 - 1),  M)
                    ]
    # z value
    constraints += [z1 == z[-1]]
    
    """
    stage two kkt as constraints
    """
    
    Q2 = standard_form2['Q']
    q2 = standard_form2['q']
    A2 = standard_form2['A']
    b2 = standard_form2['b']
    G2 = standard_form2['G']
    C2 = standard_form2['C']
    d2 = standard_form2['d']
    H2 = standard_form2['H']
    
    x2 = cp.Variable(standard_form2['Q'].shape[0])
    z2 = cp.Variable(standard_form2['G'].shape[1])
    ineq_multiplier2 = cp.Variable(standard_form2['A'].shape[0], nonneg = True)
    eq_multiplier2 = cp.Variable(standard_form2['C'].shape[0])
    phi2 = cp.Variable(standard_form2['A'].shape[0], integer = True)
    
    constraints += [Q2 @ x2 + q2 + A2.T @ ineq_multiplier2 + C2.T @ eq_multiplier2 == 0]
    constraints += [C2 @ x2 + H2 @ z2 - d2 == 0]
    constraints += [A2 @ x2 + G2 @ z2 - b2 <= 0]
    constraints += [ineq_multiplier2 <= cp.multiply(M, phi2),
                    A2 @ x2 + G2 @ z2 - b2 >= cp.multiply((phi2 - 1),  M)
                    ]
    
    # the input parameter of stage two is the output of stage one
    constraints += [z2 == x1[:5]]
    
    # affine objective
    objective = cp.Maximize(q1.T @ x1 + q2.T @ x2)
    
    prob = cp.Problem(objective, constraints)
    
    return prob, z