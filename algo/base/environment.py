import copy
import time
import socket
import numpy as np
from pprint import pprint

from .recorder import Recorder, Solution
from .controller import Controller
from data import PhysicalNetwork, VNSimulator


class Environment:
    r"""A general environment for various algorithms based on heuristics and RL"""
    def __init__(self, pn, vn_simulator, recorder=None, verbose=1, *args, **kwargs):
        self.pn = pn
        self.init_pn = copy.deepcopy(pn)
        self.vn_simulator = vn_simulator
        self.num_vns = self.vn_simulator.num_vns
        self.num_events = self.num_vns * 2
        self.controller = Controller()
        self.recorder = recorder
        self.verbose = verbose
        self.player = kwargs.get('player', '')
        self.pn_config_str = kwargs.get('pn_config_str', '')
        self.vns_config_str = kwargs.get('vns_config_str', '')
        self.if_save_records = kwargs.get('if_save_records', True)

    @classmethod
    def from_config(cls, config):
        r"""Create an environment following the settings in config."""
        if not isinstance(config, dict): config = vars(config)
        config = copy.deepcopy(config)

        summary_dir = config.pop('summary_dir', 'records/')
        record_dir = config.pop('record_dir', 'records/')
        verbose = config.pop('verbose', 1)
        if_temp_save_records = config.pop('if_temp_save_records', True)

        pn_dataset_dir = config.pop('pn_dataset_dir')
        vns_dataset_dir = config.pop('vns_dataset_dir')

        player = config.pop('algo_name', 'unknown')
        pn_config_str = pn_dataset_dir
        vns_config_str = vns_dataset_dir
        
        pn = PhysicalNetwork.load_dataset(pn_dataset_dir)
        vn_simulator = VNSimulator.load_dataset(vns_dataset_dir)
        recorder = Recorder(summary_dir=summary_dir, save_dir=record_dir, if_temp_save_records=if_temp_save_records)
        return cls(pn, vn_simulator, recorder, verbose, 
                    player=player, pn_config_str=pn_config_str, vns_config_str=vns_config_str, **config)

    def ready(self, event_id=0):
        r"""Ready to attempt to execuate the current events."""
        self.curr_event = self.vn_simulator.events[event_id]
        self.curr_vn = self.vn_simulator.vns[int(self.curr_event['vn_id'])]
        self.curr_vnf_idx = 0
        self.curr_solution = Solution(self.curr_vn)
        self.pn_backup = copy.deepcopy(self.pn)
        self.recorder.ready(self.curr_event)
        self.curr_vn_reward = 0
        if self.verbose >= 2:
            print(f"\nEvent: id={event_id}, type={self.curr_event['type']}")
            print(f"{'-' * 30}")

    def reset(self, seed=None, renew_vn_simulator=False):
        r"""Reset the environment."""
        # np.random.seed(1234 if seed is None else seed)
        self.pn = copy.deepcopy(self.init_pn)
        self.recorder.reset()
        self.recorder.count_init_pn_info(self.pn)
        print(f'temp save record in {self.recorder.temp_save_path}\n') if self.verbose >= 1 else None
        if renew_vn_simulator:
            self.vn_simulator.renew(vns=True, events=True)
        self.cumulative_reward = 0
        self.ready(event_id=0)
        return self.get_observation()

    def step(self, action):
        return NotImplementedError

    def compute_reward(self):
        return NotImplementedError

    def get_observation(self):
        return NotImplementedError

    def render(self, mode="human"):
        return NotImplementedError

    def release(self):
        r"""Release occupied resources when a VN leaves PN."""
        solution = self.recorder.get_record(vn_id=self.curr_vn['id'])
        self.controller.release(self.curr_vn, self.pn, solution)
        self.curr_solution['description'] = 'Leave Event'
        record = self.count_and_add_record()
        return record

    def get_failure_reason(self, solution):
        if solution['early_rejection']:
            return 'reject'
        if not solution['place_result']:
            return 'place'
        elif not solution['route_result']:
            return 'route'
        else:
            return 'unknown'

    def rollback_for_failure(self, reason='place'):
        r"""Restore the state of the physical network and record the reason of failure."""
        self.curr_solution.reset()
        self.pn = copy.deepcopy(self.pn_backup)
        if reason in ['unknown', -1]:
            self.curr_solution['description'] = 'Unknown Reason'
        if reason in ['reject', 0]:
            self.curr_solution['description'] = 'Early Rejection'
            self.curr_solution['early_rejection'] = True
        elif reason in ['place', 1]:
            self.curr_solution['description'] = 'Place Failure'
            self.curr_solution['place_result'] = False
        elif reason in ['route', 2]:
            self.curr_solution['description'] = 'Route Failure'
            self.curr_solution['route_result'] = False
        else:
            return NotImplementedError

    def transit_obs(self):
        """Automatically execute the leave events 
            until the next enter event comes or episode has done.
        
        Return:
            done (bool): whether the current episode has done
        """
        # Leave events transition
        while True:
            next_event_id = int(self.curr_event['id'] + 1)
            # episode end
            if next_event_id > self.num_events - 1:
                host_name = socket.gethostname()
                run_time = time.strftime('%Y%m%dT%H%M%S')
                if self.if_save_records:
                    self.save_records(f'{self.player}-{host_name}-{run_time}.csv')
                extra_dict = {
                    'player': self.player,
                    'pn_config': self.pn_config_str,
                    'vns_config': self.vns_config_str,
                    'host_name': host_name,
                    'run_time': run_time, 
                }
                self.save_summary(**extra_dict)
                return True
            self.ready(next_event_id)
            if self.curr_event['type'] == 0:
                record = self.release()
            else:
                return False

    @property
    def selected_pn_nodes(self):
        return list(self.curr_solution['node_slots'].values())

    @property
    def placed_vn_nodes(self):
        return list(self.curr_solution['node_slots'].keys())

    ### recorder ###
    def add_record(self, record, extra_info={}):
        self.recorder.add_record(record, extra_info)
        if self.verbose >= 2:
            self.display_record(record, extra_items=list(extra_info.keys()))

    def count_and_add_record(self, extra_info={}):
        record = self.recorder.count(self.curr_vn, self.pn, self.curr_solution)
        record = self.add_record(record, extra_info)
        return record

    def display_record(self, record, display_items=['result', 'vn_id', 'vn_cost', 'vn_revenue', 
                        'pn_available_resource', 'total_revenue', 'total_cost', 'description'], extra_items=[]):
        display_items = display_items + extra_items
        print(''.join([f'{k}: {v}\n' for k, v in record.items() if k in display_items]))

    def save_records(self, file_name):
        save_path = self.recorder.save_records(file_name)
        print(f'save records to {save_path}') if self.verbose >= 1 else None

    def save_summary(self, file_name='global_summary.csv', **extra_info):
        summary_info = self.recorder.summary_records(self.recorder.memory)
        for k, v in extra_info.items():
            summary_info[k] = v
        summary_path = self.recorder.save_summary(summary_info, file_name)
        if self.verbose >= 1: pprint(summary_info)
        print(f'save summary to {summary_path}\n') if self.verbose >= 1 else None


class SolutionStepEnvironment(Environment):
    
    def __init__(self, pn, vn_simulator, recorder, verbose=1, **kwargs):
        super(SolutionStepEnvironment, self).__init__(pn, vn_simulator, recorder, verbose=verbose, **kwargs)

    def step(self, solution):
        # Enter 
        # Success
        if solution['result']:
            self.curr_solution = solution
            self.curr_solution['description'] = 'Success'
        # Failure
        else:
            failure_reason = self.get_failure_reason(solution)
            self.rollback_for_failure(reason=failure_reason)
        record = self.count_and_add_record()
        done = self.transit_obs()
        return self.get_observation(), self.compute_reward(), done, record

    def get_observation(self):
        return {'vn': self.curr_vn, 'pn': self.pn}

    def compute_reward(self):
        return 0

    def generate_action_mask(self):
        return np.arange(self.pn.num_nodes)


class JointPRStepEnvironment(Environment):

    def __init__(self, pn, vn_simulator, recorder=None, verbose=1, **kwargs):
        super().__init__(pn, vn_simulator, recorder=recorder, verbose=verbose, **kwargs)

    def step(self, pn_node_idx):
        place_and_route_result = self.controller.place_and_route(self.vn, self.pn, self.curr_vnf_idx, pn_node_idx, self.curr_solution, 
                                        available_network=False, shortest_method='all_shortest', k=1)
        # Step Failure
        if not place_and_route_result:
            failure_reason = self.get_failure_reason(self.curr_solution)
            self.rollback_for_failure(failure_reason)
        # Step Success
        else:
            self.curr_vnf_idx += 1
            # VN Success ?
            if self.curr_vnf_idx == self.curr_vn.num_nodes:
                self.curr_solution['result'] = True
            else:
                return self.get_observation(), self.compute_reward(), False, self.curr_solution.to_dict()

        record = self.count_and_add_record(self.curr_vn, self.pn, self.curr_solution)
        info = record

        # obs transition
        if not place_and_route_result or self.curr_solution['result']:
            done = self.transit_obs()
        else:
            done = False
        return self.get_observation(), self.compute_reward(), done, info

    def get_observation(self):
        return {'vnf': self.curr_vn[self.curr_vnf_idx], 'pn': self.pn}

    def compute_reward(self):
        return 0


if __name__ == '__main__':
    pass
