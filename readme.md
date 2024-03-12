# E2E-AT: A Unified Framework for Tackling Uncertainty in Task-aware End-to-end Learning

This is the official repo for the paper *E2E-AT: A Unified Framework for Tackling Uncertainty in Task-aware End-to-end Learning*, to be appeared in AAAI-24. You can find the preprint here.

The authors include:
- Wangkun Xu: Control and Power Research Group, Department of EEE, Imperial College London.
- Dr. Jianhong Wang: University of Manchester.
- Dr. Fei Teng: Control and Power Research Group, Department of EEE, Imperial College London.

## Developing plan

This repo is under development but should contain the main functionality to support the paper results. We are currently working on merging the E2E-AT into a more general code base for reasearches in E2E learning and power system community, to support diverse power system operations, such as stochastic programming, integer programming, etc. 

The functions for certifying the robustness will also come later.

## How it works

### Packages

The core packages include `cvxpy` for solving optimization problems, `cvxpylayers` for embedding optimization problems as differentiable layers in NN and `pytorch`. `pypower` is also used to retrieve the power system configuration. We have tested with the versions:
```
torch==2.0.1+cpu
pypower==5.1.16
cvxpy==1.4.1                    
cvxpylayers==0.1.4
```

### Data
We use open source dataset from [A Synthetic Texas Backbone Power System with Climate-Dependent Spatio-Temporal Correlated Profiles](https://arxiv.org/abs/2302.13231). You can download/read the descriptions of the dataset from [the official webpage](https://rpglab.github.io/resources/TX-123BT/).

Download the the `.zip` file into the `data/` folder and change the name into `raw_data.zip`, unzip the file by 
```bash
unzip data/raw_data.zip -d data/
```
This should give you `data/Data_public/` folder.

Then to generate the dataset used in the paper, run
```bash
python clean_data.py --no_bus 14
```

We have modified the standard IEEE case files from [pypower](https://github.com/rwl/PYPOWER/tree/master/pypower). The modifications can be found in 'bus_config.json'.

## Training

All the configurations can be found in `configs.json`.

For clean training using mse loss:
```
python trainer_clean_mse.py
```

For robust training using mse loss (attack on the input only):
```
python trainer_robust_mse.py --max_eps {eps_parameter}
```

For clean training using objective-based loss:
```
python trainer_clean_spo.py -p
```
where `-p` represents to retrain the model using the `clean_mse`. The pre-trained model is saved as `trained_model//full/mse_warm.pth`.

For robust training using objective-based loss:
```
python trainer_robust_spo.py -p spo_clean -t {train_method} -a {parameter} -i {random} --eps_parameter {eps_parameter} --eps_input {eps_input} --alpha {alpha}
```
where `-t`: choose from 'normal' for regular adversarial training; 'free' for adversarial training for free.
`-a`: choose the attack method from 'input', 'parameter', or 'both'.
`-i`: choose the attack initializationg method: 'random' for random initialization in the range, 
                                                'previous' for use the previous generated attack for initialization, or
                                                'default' for start at the clean sample.
`--eps_parameter`: the attack strength on the parameter (in ratio, e.g. 0.1)
`--eps_input`: the attack strength on the input.
`--alpha`: the parameter to balance the clean and adversarial training loss.


## Citation
To cite our paper:
```
@article{xu2023e2e,
  title={E2E-AT: A Unified Framework for Tackling Uncertainty in Task-aware End-to-end Learning},
  author={Xu, Wangkun and Wang, Jianhong and Teng, Fei},
  journal={AAAI24},
  year={2023}
}
```
