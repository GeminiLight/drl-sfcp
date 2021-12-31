import os
import json
import time
import pprint
import socket
import argparse

from utils import read_json, get_pn_dataset_dir_from_setting, get_vns_dataset_dir_from_setting


parser = argparse.ArgumentParser(description='configuration file')

def str2bool(v):
    return v.lower() in ('true', '1')

### Dataset ###
data_arg = parser.add_argument_group('data')
data_arg.add_argument('--pn_setting_path', type=str, default='settings/pn_setting.json', help='')
data_arg.add_argument('--vns_setting_path', type=str, default='settings/vns_setting.json', help='')

### Environment ###
env_arg = parser.add_argument_group('env')
env_arg.add_argument('--summary_dir', type=str, default='records/', help='Save summary')
env_arg.add_argument('--if_save_records', type=str2bool, default=True, help='')
env_arg.add_argument('--if_temp_save_records', type=str2bool, default=True, help='')

### Algo  ###
algo_arg = parser.add_argument_group('algo')
algo_arg.add_argument('--verbose', type=str2bool, default=1, help='')
algo_arg.add_argument('--algo_name', type=str, default='grc_rank', help='Algorithm selected to run')
algo_arg.add_argument('--reusable', type=str2bool, default=False, help='Whether or not to allow to deploy several VN nodes on the same VNF')

### Neural Network ###
net_arg = parser.add_argument_group('net')
# device
net_arg.add_argument('--use_cuda', type=str2bool, default=True, help='Use GPU to accelerate the training process')
net_arg.add_argument('--allow_parallel', type=str2bool, default=False, help='Use mutiple GPUs')
# rl
net_arg.add_argument('--drl_gamma', type=float, default=0.95, help='Cumulative reward discount rate')
net_arg.add_argument('--explore_rate', type=float, default=0.9, help='Epsilon-greedy explore rate')
net_arg.add_argument('--actor_lr', type=float, default=1e-4, help='Actor learning rate')
net_arg.add_argument('--critic_lr', type=float, default=1e-4, help='Critic learning rate')
# nn
net_arg.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension')
net_arg.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension')
net_arg.add_argument('--num_layers', type=int, default=1, help='Number of GRU stacks\' layers')
net_arg.add_argument('--enc_units', type=int, default=64, help='Units of encoder GRU')
net_arg.add_argument('--dec_units', type=int, default=64, help='Units of decoder GRU')
net_arg.add_argument('--gnn_units', type=int, default=64, help='Units of decoder GNN')
net_arg.add_argument('--dropout_rate', type=float, default=0.2, help='Droput rate')
net_arg.add_argument('--l2reg_rate', type=float, default=2.5e-4, help='L2 regularization rate')

### Run ###
run_arg = parser.add_argument_group('run')
run_arg.add_argument('--only_test', type=str2bool, default=True, help='Only test without training')
run_arg.add_argument('--start_epoch', type=int, default=0, help='Start from i')
run_arg.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
run_arg.add_argument('--batch_size', type=int, default=64, help='Batch size of training')
run_arg.add_argument('--random_seed', type=int, default=1234, help='Random seed')
# run_arg.add_argument('--save_model', type=str2bool, default=False, help='Save model')
# run_arg.add_argument('--load_model', type=str2bool, default=False, help='Load model')
run_arg.add_argument('--save_dir', type=str, default='save', help='Save directory')
run_arg.add_argument('--log_dir', type=str, default='logs', help='Log directory')
#run_arg.add_argument('--lr_decay_step', type=int, default=5000, help='Lr1 decay step')
#run_arg.add_argument('--lr_decay_rate', type=float, default=0.96, help='Lr1 decay rate')

### Misc ###
misc_arg = parser.add_argument_group('misc')

def get_config(args=None):
    config = parser.parse_args(args)

    assert config.reusable == False, 'Unsupported currently!'

    # read pn and vns setting
    config.pn_setting = read_json(config.pn_setting_path)
    config.vns_setting = read_json(config.vns_setting_path)

    # get dataset dir
    config.pn_dataset_dir = get_pn_dataset_dir_from_setting(config.pn_setting)
    config.vns_dataset_dir = get_vns_dataset_dir_from_setting(config.vns_setting)

    # time and host
    config.run_time = time.strftime('%Y%m%dT%H%M%S')
    config.host_name = socket.gethostname()

    # preprocess
    config.record_dir = os.path.join(  # save record
        config.summary_dir, 
        f'{config.algo_name}',
    )
    config.log_dir = os.path.join(
        config.log_dir, 
        f'{config.algo_name}', 
        f'{config.host_name}-{config.run_time}'
    )
    config.save_dir = os.path.join(
        config.save_dir, 
        f'{config.algo_name}',
        f'{config.host_name}-{config.run_time}'
    )
    for dir in [config.record_dir]: # config.record_dir, config.log_dir, config.save_dir
        if not os.path.exists(dir):
            os.makedirs(dir)

    if config.verbose >= 1:
        show_config(config)
    return config

def show_config(config):
    pprint.pprint(vars(config))

def save_config(config, fname='args.json'):
    config_path = os.path.join(config.save_dir, fname)
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=True)
    print(f'Save config in {config_path}')

def delete_empty_dir(config):
    for dir in [config.record_dir, config.log_dir, config.save_dir]:
        if os.path.exists(dir) and not os.listdir(dir):
            os.rmdir(dir)


if __name__ == "__main__":
    pass