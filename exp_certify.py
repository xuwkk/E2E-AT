"""
experiment for certified attack with MILP formulation
"""

if __name__ == "__main__":
    import json
    import torch
    import numpy as np
    import random
    import argparse
    from tqdm import tqdm
    import cvxpy as cp
    from utils.dataset import MyDataset
    from utils.certify import input_bound_clamp, return_cost, matrix_kkt, form_certify
    from helper import return_operator, return_nn_model
    from utils.net import NN_SPO
    import matplotlib.pyplot as plt

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--case_name', type = str, default = 'case14')
    argparser.add_argument('-t', '--train_method', type = str)
    argparser.add_argument('--eps_input', type = float)

    args = argparser.parse_args()

    with open("config.json") as f:
        config = json.load(f)
    print(config['nn'])
    random_seed = config['random_seed']
    fix_first_b = config['fix_first_b']
    feature_size = config['nn']['feature_size']
    fixed_feature = config['attack']['fixed_feature']
    flexible_feature = list(set(np.arange(feature_size)) - set(fixed_feature))
    certify_no = config['attack']['certify_no']
    is_scale = config['is_scale']
    multirun_no = config['attack']['multirun_no']

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    train_dataset = MyDataset(case_name = args.case_name, mode = "train")
    test_dataset = MyDataset(case_name = args.case_name, mode = "test")

    operator = return_operator(args.case_name)
    nn = return_nn_model(is_load = True, 
                        train_method = args.train_method, 
                        with_relu = True) # with relu so that the forecast is always positive
    
    if is_scale:
        mean = train_dataset.target_mean
        std = train_dataset.target_std
    else:
        mean = 0
        std = 1
    net_spo = NN_SPO(model = nn, operator = operator, mean = mean, std = std, fix_first_b = fix_first_b)

    print('nn layers:')
    for layer in nn:
        print(layer)
    
    feature_all = train_dataset.feature
    target_all = train_dataset.target

    print('feature shape', feature_all.shape, target_all.shape)
    print('flexible feature:', flexible_feature)    

    random_index = np.random.choice(len(feature_all), certify_no, replace = False)
    print('random index:', random_index)

    feature_selected = feature_all[random_index]
    target_selected = target_all[random_index]
    print("feature shape:", feature_selected.shape)

    stat = np.load(f'data/data_{args.case_name}/climate_data_stats.npy', allow_pickle = True).item()
    min_vector = np.concatenate([stat[i]['min'] for i in stat.keys()])
    max_vector = np.concatenate([stat[i]['max'] for i in stat.keys()])

    # milp objective-based attack
    for idx in tqdm(random_index):

        feature = feature_all[idx:idx+1]
        true_load = target_all[idx]
        forecast_load = nn(feature).detach().numpy()

        # clean cost
        clean_cost = return_cost(feature, true_load, nn, operator)

        # certified attack
        feature_min, feature_max = input_bound_clamp(feature, max_eps_input=args.eps_input, 
                                                    flexible_feature=flexible_feature)
        assert torch.all(feature_min[:, fixed_feature] == feature_max[:, fixed_feature])

        stage_one_prob = operator.stage_one_decision()
        stage_one_standard_form = matrix_kkt(stage_one_prob)
        stage_two_prob = operator.stage_two_decision(true_load)
        stage_two_standard_form = matrix_kkt(stage_two_prob)

        prob, z = form_certify(model = nn, initial_bounds=(feature_min, feature_max), standard_form1=stage_one_standard_form, 
                        standard_form2=stage_two_standard_form, M = 5e4)
        
        prob.solve(solver = cp.GUROBI, verbose = False)
        cost_obj = prob.value
        assert np.allclose(z[0].value[fixed_feature], feature[0][fixed_feature].numpy())
        assert np.all(z[0].value[flexible_feature] <= feature_max[0][flexible_feature].numpy() + 1e-3)
        assert np.all(z[0].value[flexible_feature] >= feature_min[0][flexible_feature].numpy() - 1e-3)

        # check the result
        cost_obj_ = return_cost(torch.from_numpy(z[0].value).float(), true_load, nn, operator)
        print('clean cost: ', clean_cost, 'objective-based attack: ', cost_obj, cost_obj_)

        """
        the following code is only for visualization
        """
        # # unscale
        # attack_feature = z[0].value[flexible_feature] * (max_vector - min_vector) + min_vector
        # normal_feature = feature[0][flexible_feature].numpy() * (max_vector - min_vector) + min_vector
        # plt.figure()
        # plt.plot(attack_feature)
        # plt.plot(normal_feature)
        # plt.show()
        # plt.savefig(f'attack_feature_{idx}.pdf')

        # temperature_attack = attack_feature[[i for i in range(0, len(attack_feature), 6)]]
        # temperature_normal = normal_feature[[i for i in range(0, len(normal_feature), 6)]]

        # print('attack temperature:', temperature_attack)
        # print('normal temperature:', temperature_normal)

        # print('attacked feature:', attack_feature[:6])
        # print('normal feature:', normal_feature[:6])

        # load_normal = nn(feature).detach().numpy()
        # load_attack = nn(torch.from_numpy(z[0].value).float()[None,:]).detach().numpy()

        # print('load normal:', load_normal.sum())
        # print('load attack:', load_attack.sum())