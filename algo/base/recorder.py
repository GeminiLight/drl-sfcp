import os
import csv
import numpy as np
import pandas as pd
from collections import defaultdict

from data import VirtualNetwork


class Recorder:
    r"""Record the environment's states during the deployment process"""
    def __init__(self, summary_dir='records/', save_dir='records/', if_temp_save_records=True):
        self.summary_dir = summary_dir
        self.save_dir = save_dir
        self.if_temp_save_records = if_temp_save_records
        self.counter = Counter()
        self.reset()
        self.curr_record = {}

    def reset(self):
        self.memory = []
        self.vn_event_dict = {}  # for querying the record of vn
        self.pn_nodes_for_vn_dict = defaultdict(lambda : [])
        self.state = {
            'vn_count': 0, 
            'success_count': 0, 
            'inservice_count': 0,
            'total_revenue': 0, 
            'total_cost': 0, 
            'total_time_revenue': 0,
            'total_time_cost': 0,
            'num_running_pn_nodes': 0
        }
        if self.if_temp_save_records:
            suffixes = 0
            available_fname = False
            while not available_fname:
                temp_save_path = os.path.join(self.save_dir , f'temp-{suffixes}.csv')
                suffixes += 1
                if not os.path.exists(temp_save_path):
                    available_fname = True
            self.temp_save_path = temp_save_path
            self.written_temp_header = False

    def count_init_pn_info(self, pn):
        self.init_pn_info = {}
        self.init_pn_info['pn_available_resource'] = self.counter.calculate_sum_network_resource(pn)
        self.init_pn_info['pn_node_available_resource'] = self.counter.calculate_sum_network_resource(pn, edge=False)
        self.init_pn_info['pn_edge_available_resource'] = self.counter.calculate_sum_network_resource(pn, node=False)

    def ready(self, event):
        self.state['event_id'] = event['id']
        self.state['event_type'] = event['time']
        self.state['event_type'] = event['type']
    
    def add_info(self, info_dict, **kwargs):
        self.curr_record.update(info_dict)
        self.curr_record.update(kwargs)

    def add_record(self, record, extra_info={}, **kwargs):
        self.curr_record.update(record)
        self.curr_record.update(extra_info)
        self.curr_record.update(kwargs)
        self.memory.append(record)
        if self.if_temp_save_records: self.temp_save_record(record)
        return record

    def count(cls, vn, pn, solution):
        r"""Count the information of environment state and vn solution."""
        # State
        cls.vn_event_dict[solution['vn_id']] = cls.state['event_id']
        cls.state['pn_available_resource'] = cls.counter.calculate_sum_network_resource(pn)
        cls.state['pn_node_available_resource'] = cls.counter.calculate_sum_network_resource(pn, edge=False)
        cls.state['pn_edge_available_resource'] = cls.counter.calculate_sum_network_resource(pn, node=False)
        # Leave event
        if cls.state['event_type'] == 0:
            if cls.get_record(solution['vn_id'])['result']:
                cls.state['inservice_count'] -= 1
                vn_id = solution['vn_id']
                for vnf_id, pn_node_idx in solution['node_slots'].items():
                    cls.pn_nodes_for_vn_dict[pn_node_idx].remove(vn_id)
                cls.state['num_running_pn_nodes'] = len(cls.get_running_pn_nodes())

        # Enter event
        if cls.state['event_type'] == 1:
            cls.state['vn_count'] += 1
            solution = cls.counter.count_solution(vn, solution)
            # Success
            if solution['result']:
                cls.state['success_count'] += 1
                cls.state['inservice_count'] += 1
                cls.state['total_revenue'] += solution['vn_revenue']
                cls.state['total_cost'] += solution['vn_cost']
                cls.state['total_time_revenue'] += solution['vn_time_revenue']
                cls.state['total_time_cost'] += solution['vn_time_cost']
                vn_id = solution['vn_id']
                for vnf_id, pn_node_idx in solution['node_slots'].items():
                    cls.pn_nodes_for_vn_dict[pn_node_idx].append(vn_id)
                cls.state['num_running_pn_nodes'] = len(cls.get_running_pn_nodes())

        record = {**cls.state, **solution.__dict__}
        return record

    def get_running_pn_nodes(cls):
        return [pn_nodes for pn_nodes, vns_list in cls.pn_nodes_for_vn_dict.items() if len(vns_list) != 0]

    def get_record(self, event_id=None, vn_id=None):
        r"""Get the record of the service function chain `vn_id`."""
        if event_id is not None: event_id = event_id
        elif vn_id is not None: event_id = self.vn_event_dict[vn_id]
        else: event_id = self.state['event_id']
        return self.memory[int(event_id)]

    def display_record(self, record, display_items=['result', 'vn_id', 'vn_cost', 'vn_revenue', 'pn_available_resource', 'total_revenue', 'total_cost', 'description'], extra_items=[]):
        display_items = display_items + extra_items
        print(''.join([f'{k}: {v}\n' for k, v in record.items() if k in display_items]))

    def temp_save_record(self, record):
        with open(self.temp_save_path, 'a+', newline='') as f:  # Just use 'w' mode in 3.x
            writer = csv.writer(f)
            if not self.written_temp_header:
                writer.writerow(record.keys())
            writer.writerow(record.values())
        self.written_temp_header = True
        
    def save_records(self, fname):
        r"""Save records to a csv file."""
        save_path = os.path.join(self.save_dir, fname)
        pd_records = pd.DataFrame(self.memory)
        pd_records.to_csv(save_path, index=False)
        os.remove(self.temp_save_path)
        return save_path

    ### summary ###
    def summary_records(self, records):
        return self.counter.summary_records(records)

    def save_summary(self, summary_info, fname='global_summary.csv'):
        summary_path = os.path.join(self.summary_dir,  fname)
        head = None if os.path.exists(summary_path) else list(summary_info.keys())
        with open(summary_path, 'a+', newline='') as csv_file:
            writer = csv.writer(csv_file, dialect='excel', delimiter=',')
            if head is not None: writer.writerow(head)
            writer.writerow(list(summary_info.values()))
        return summary_path


class Counter(object):

    @staticmethod
    def count_solution(curr_vn, solution):
        # Success
        if solution['result']:
            solution['vn_revenue'] = Counter.calculate_sum_network_resource(curr_vn)
            solution['vn_cost'] = Counter.calculate_vn_cost(curr_vn, solution)
            solution['vn_rc_ratio'] = solution['vn_revenue'] /solution['vn_cost']
        # Faliure
        else:
            solution['vn_revenue'] = 0
            solution['vn_cost'] = 0
            solution['vn_rc_ratio'] = 0
            solution['node_slots'] = {}
            solution['edge_paths'] = {}
        solution['vn_time_revenue'] = solution['vn_revenue'] * curr_vn.lifetime
        solution['vn_time_cost'] = solution['vn_cost'] * curr_vn.lifetime
        solution['vn_time_rc_ratio'] = solution['vn_rc_ratio'] * curr_vn.lifetime
        return solution

    @staticmethod
    def summary_records(records):
        if isinstance(records, list):
            records = pd.DataFrame(records)
        elif isinstance(records, pd.DataFrame):
            pass
        else:
            raise TypeError
        summary_info = {}

        # ac rate
        summary_info['success_count'] = records.iloc[-1]['success_count']
        summary_info['acceptance_rate'] = records.iloc[-1]['success_count'] / records.iloc[-1]['vn_count']

        # revenue / cost
        summary_info['total_cost'] = records.iloc[-1]['total_cost']
        summary_info['total_revenue'] = records.iloc[-1]['total_revenue']
        summary_info['total_time_cost'] = records.iloc[-1]['total_time_cost']
        summary_info['total_time_revenue'] = records.iloc[-1]['total_time_revenue']

        # rc ratio
        summary_info['rc_ratio'] = records.iloc[-1]['total_revenue'] / records.iloc[-1]['total_cost']
        summary_info['time_rc_ratio'] = records.iloc[-1]['total_time_revenue'] / records.iloc[-1]['total_time_cost']

        # other
        summary_info['min_pn_available_resource'] = records.loc[:, 'pn_available_resource'].min()
        summary_info['min_pn_node_available_resource'] = records.loc[:, 'pn_node_available_resource'].min()
        summary_info['min_pn_edge_available_resource'] = records.loc[:, 'pn_edge_available_resource'].min()
        summary_info['max_inservice_count'] = records.loc[:, 'inservice_count'].max()
        
        # rl reward
        if 'cumulative_reward' in records.columns:
            cumulative_rewards = records.loc[:, 'cumulative_reward'].dropna()
            summary_info['cumulative_reward'] = cumulative_rewards.iloc[-1]
        else:
            summary_info['cumulative_reward'] = 0
        return summary_info

    @staticmethod
    def summary_csv(fpath):
        records = pd.read_csv(fpath, header=0)
        summary_info = Counter.summary_records(records)
        return summary_info

    @staticmethod
    def calculate_sum_network_resource(network, node=True, edge=True):
        n = np.array(network.get_node_attrs_data(network.get_node_attrs('resource'))).sum() if node else 0
        e = np.array(network.get_edge_attrs_data(network.get_edge_attrs('resource'))).sum() if edge else 0
        return n + e


    @staticmethod
    def calculate_vn_revenue(curr_vn, solution=None):
        r"""Calculate the deployment cost of current vn according to `edge paths`."""
        return Counter.calculate_sum_network_resource(curr_vn)

    @staticmethod
    def calculate_vn_cost(curr_vn, solution=None):
        r"""Calculate the deployment cost of current vn according to `edge paths`."""
        vn_cost = 0
        for edge, path in solution['edge_paths'].items():
            revenue = curr_vn.edges[edge]['bw']
            vn_cost += revenue * (len(path) - 2)
        vn_cost += Counter.calculate_sum_network_resource(curr_vn)
        return vn_cost


class ClassDict(object):
    def __init__(self):
        super(ClassDict, self).__init__()

    def update(self, *args, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    @classmethod
    def from_dict(cls, dict):
        cls.__dict__ = dict
        return cls

    def to_dict(self):
        return self.__dict__

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key, None)
        elif isinstance(key, int):
            return super().__getitem__(key)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)


class Solution(ClassDict):
    def __init__(self, vn=0):
        super(Solution, self).__init__()
        if isinstance(vn, int):
            self.vn_id = vn
        elif isinstance(vn, VirtualNetwork):
            self.vn_id = vn.id
        else:
            raise TypeError('')
        self.reset()

    def reset(self):
        self.node_slots = {}
        self.edge_paths = {}
        self.vn_cost = 0
        self.vn_revenue = 0
        self.vn_rc_ratio = 0
        self.vn_time_cost = 0
        self.vn_time_revenue = 0
        self.vn_time_rc_ratio = 0
        self.result = False
        self.description = ''
        self.place_result = True
        self.route_result = True
        self.early_rejection = False
    
    def count_last_step(self, vn):
        vnf_id = self.node_slots.keys()[-1]
        pn_node_id = self.node_slots.values()[-1]
        n_attrs = vn.get_node_attrs(['resource'])
        e_attrs = vn.get_edge_attrs(['resource'])