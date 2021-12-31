import os
import json
from algo.base import Agent, Environment, SolutionStepEnvironment


def read_json(fpath):
    with open(fpath, 'r') as f:
        attrs_dict = json.load(f)
    return attrs_dict

def write_json(dict_data, fpath):
    with open(fpath, 'w+') as f:
        json.dump(dict_data, f)

def load_algo(algo_name):
    # rank
    if algo_name == 'random_rank':
        from algo.heuritics.node_rank import RandomRankAgent
        Env, Agent = SolutionStepEnvironment, RandomRankAgent
    elif algo_name == 'order_rank':
        from algo.heuritics.node_rank import OrderRankAgent
        Env, Agent = SolutionStepEnvironment, OrderRankAgent
    elif algo_name == 'nr_rank':
        from algo.heuritics.node_rank import NRRankAgent
        Env, Agent = SolutionStepEnvironment, NRRankAgent
    elif algo_name == 'grc_rank':
        from algo.heuritics.node_rank import GRCRankAgent
        Env, Agent = SolutionStepEnvironment, GRCRankAgent
    elif algo_name == 'ffd_rank':
        from algo.heuritics.node_rank import FFDRankAgent
        Env, Agent = SolutionStepEnvironment, FFDRankAgent
    elif algo_name == 'nrm_rank':
        from algo.heuritics.node_rank import NRMRankAgent
        Env, Agent = SolutionStepEnvironment, NRMRankAgent
    # joint_pr
    elif algo_name == 'random_joint_pr':
        from algo.heuritics.joint_pr import RandomJointPRAgent
        Env, Agent = SolutionStepEnvironment, RandomJointPRAgent
    elif algo_name == 'order_joint_pr':
        from algo.heuritics.joint_pr import OrderJointPRAgent
        Env, Agent = SolutionStepEnvironment, OrderJointPRAgent
    # rank_bfs
    elif algo_name == 'nr_rank_bfs':
        from algo.heuritics.bfs_trials import NRRankBFSAgent
        Env, Agent = SolutionStepEnvironment, NRRankBFSAgent
    elif algo_name == 'order_rank_bfs':
        from algo.heuritics.bfs_trials import OrderRankBFSAgent
        Env, Agent = SolutionStepEnvironment, OrderRankBFSAgent
    elif algo_name == 'random_rank_bfs':
        from algo.heuritics.bfs_trials import RandomRankBFSAgent
        Env, Agent = SolutionStepEnvironment, RandomRankBFSAgent
    # ml
    elif algo_name == 'mcts_vine':
        from algo.learning.mcts_vine import MCTSAgent
        Env, Agent = SolutionStepEnvironment, MCTSAgent
    elif algo_name == 'graph_vine':
        from algo.learning.graph_vine import GraphViNEAgent
        Env, Agent = SolutionStepEnvironment, GraphViNEAgent
    elif algo_name == 'gnn_ppo':
        from algo.learning.gnn_ppo import GNNPPOEnvironment, GNNPPOAgent
        Env, Agent = GNNPPOEnvironment, GNNPPOAgent
    elif algo_name == 'pg_conv':
        from algo.learning.pg_conv import PGConv
        Env, Agent = PGConv
    elif algo_name == 'ppo_conv':
        from algo.learning.ppo_conv import PPOConvEnv, PPOConvAgent
        Env, Agent = PPOConvEnv, PPOConvAgent
    else:
        raise ValueError('The algorithm is not yet supported; \n Please attempt to select another one.', algo_name)
    return Env, Agent


def get_pn_dataset_dir_from_setting(pn_setting):
    pn_dataset_dir = pn_setting.get('save_dir')
    n_attrs = [n_attr['name'] for n_attr in pn_setting['node_attrs']]
    e_attrs = [e_attr['name'] for e_attr in pn_setting['edge_attrs']]

    pn_dataset_middir = f"{pn_setting['num_nodes']}-{pn_setting['type']}-{pn_setting['wm_alpha']}-{pn_setting['wm_beta']}-" +\
                        f"{n_attrs}-{e_attrs}"
    pn_dataset_dir = os.path.join(pn_dataset_dir, pn_dataset_middir)
    return pn_dataset_dir

def get_vns_dataset_dir_from_setting(vns_setting):
    vns_dataset_dir = vns_setting.get('save_dir')
    n_attrs = [n_attr['name'] for n_attr in vns_setting['node_attrs']]
    e_attrs = [e_attr['name'] for e_attr in vns_setting['edge_attrs']]

    vns_dataset_middir = f"{vns_setting['num_vns']}-[{vns_setting['min_length']}-{vns_setting['max_length']}]-" + \
                        f"{vns_setting['type']}-{vns_setting['aver_lifetime']}-{vns_setting['aver_arrival_rate']}-" + \
                        f"{n_attrs}-{e_attrs}"
    vn_dataset_dir = os.path.join(vns_dataset_dir, vns_dataset_middir)

    return vn_dataset_dir

def generate_file_name(config, epoch_id=0, extra_items=[], **kwargs):
    if not isinstance(config, dict): config = vars(config)
    items = extra_items + ['pn_num_nodes', 'reusable', 'vn_aver_lifetime', 'vn_aver_arrival_rate', 'vn_max_length']

    file_name_1 = f"{config['algo_name']}-records-{epoch_id}-"
    # file_name_2 = '-'.join([f'{k}={config[k]}' for k in items])
    file_name_3 = '-'.join([f'{k}={v}' for k, v in kwargs.items()])
    file_name = file_name_1 + file_name_3 + '.csv'
    return file_name
