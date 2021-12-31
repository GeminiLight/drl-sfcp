import os
import copy
import json
import numpy as np

from .utils import read_json
from .virtual_network import VirtualNetwork


class VNSimulator(object):
    """A simulator for generating virtual network and arrange them"""
    def __init__(self, 
                 num_vns, 
                 type='random',
                 node_attrs=[], 
                 edge_attrs=[], 
                 min_length=2, 
                 max_length=10, 
                 aver_arrival_rate=0.05, 
                 aver_lifetime=500, 
                 **kwargs):
        super(VNSimulator, self).__init__()
        self.num_vns = num_vns
        self.node_attrs = node_attrs
        self.edge_attrs = edge_attrs
        self.type = type
        self.min_length = min_length
        self.max_length = max_length
        self.aver_arrival_rate = aver_arrival_rate
        self.aver_lifetime = aver_lifetime
        self.vns = []
        self.events = []
        self.random_prob = kwargs.get('random_prob', 0.5)
        self.wm_alpha = kwargs.get('wm_alpha', 0.5)
        self.wm_beta = kwargs.get('wm_beta', 0.2)
    
    @staticmethod
    def from_setting(setting):
        setting = copy.deepcopy(setting)
        num_vns = setting.pop('num_vns', 2000)
        type = setting.pop('type', 'random')
        node_attrs = setting.pop('node_attrs')
        edge_attrs = setting.pop('edge_attrs')
        min_length = setting.pop('min_length', 2)
        max_length = setting.pop('max_length', 10)
        aver_arrival_rate = setting.pop('aver_arrival_rate', 0.5)
        aver_lifetime = setting.pop('aver_lifetime', 500)
        return VNSimulator(num_vns, type, node_attrs, edge_attrs, min_length, max_length, 
                            aver_arrival_rate, aver_lifetime, **setting)

    def renew(self, vns=True, events=True):
        if vns == True:
            self.renew_vns()
        if events == True:
            self.renew_events()
        return self.vns, self.events

    def renew_vns(self):
        self.vns = []
        self.arrange_vns()
        for i in range(self.num_vns):
            vn = VirtualNetwork(node_attrs=copy.deepcopy(self.node_attrs), edge_attrs=copy.deepcopy(self.edge_attrs),
                                id=i, arrival_time=self.vns_arrival_time[i], lifetime=self.vns_lifetime[i])
            vn.generate_topology(num_nodes=self.vns_length[i], type=self.type, random_prob=self.random_prob)
            vn.generate_attrs_data()
            self.vns.append(vn)
        return self.vns

    def renew_events(self):
        self.events = []
        arrival_list = [{'vn_id': vn.id, 'time': vn.arrival_time, 'type': 1} for vn in self.vns]
        leave_list = [{'vn_id': vn.id, 'time': vn.arrival_time + vn.lifetime, 'type': 0} for vn in self.vns]
        event_list = arrival_list + leave_list
        self.events = sorted(event_list, key=lambda e: e.__getitem__('time'))
        for i, e in enumerate(self.events): 
            e['id'] = i
        return self.events

    def arrange_vns(self):
        # length: uniform distribution
        self.vns_length = np.random.randint(self.min_length, self.max_length, self.num_vns).tolist()
        # lifetime: exponential distribution
        self.vns_lifetime = np.random.exponential(self.aver_lifetime, self.num_vns).tolist()
        # poisson distribution
        self.vns_arrival_time = np.ceil(np.cumsum(np.array([-np.log(np.random.uniform()) 
                                / self.aver_arrival_rate for i in range(self.num_vns)]))).tolist()
    
    def save_dataset(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        vns_dir = os.path.join(save_dir, 'vns')
        if not os.path.exists(vns_dir):
            os.makedirs(vns_dir)
        # save vns
        for vn in self.vns:
            vn.to_gml(os.path.join(vns_dir, f'vn-{vn.id:05d}.gml'))
        # save events
        with open(os.path.join(save_dir, 'events.json'), 'w+') as f:
            json.dump(self.events, f)
        # save setting
        self.save_setting(os.path.join(save_dir, 'vns_setting.json'))

    @staticmethod
    def load_dataset(dataset_dir):
        # load setting
        if not os.path.exists(dataset_dir):
            raise ValueError(f'Find no dataset in {dataset_dir}.\nPlease firstly generating it.')
        vns_setting = read_json(os.path.join(dataset_dir, 'vns_setting.json'))
        vn_simulator = VNSimulator.from_setting(vns_setting)
        # load vns
        vn_fnames_list = os.listdir(os.path.join(dataset_dir, 'vns'))
        vn_fnames_list.sort()
        for vn_fname in vn_fnames_list:
            vn = VirtualNetwork.from_gml(os.path.join(dataset_dir, 'vns', vn_fname))
            vn_simulator.vns.append(vn)
        # load events
        events = read_json(os.path.join(dataset_dir, 'events.json'))
        vn_simulator.events = events
        return vn_simulator

    def save_setting(self, fpath):
        setting = {}
        for k, v in self.__dict__.items():
            if k not in ['events', 'vns', 'vns_length', 'vns_lifetime', 'vns_arrival_time']:
                setting[k] = v
        with open(fpath, 'w+') as f:
            json.dump(setting, f)


if __name__ == '__main__':
    pass
