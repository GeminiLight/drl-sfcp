import gym
import numpy as np
from gym import spaces


from ..base import Environment


class RLEnv(gym.Env):

    def __init__(self, rejection_action=False, **kwargs):
        super(RLEnv, self).__init__()
        self.rejection_action = rejection_action
        self.action_space = spaces.Discrete(self.pn.num_nodes + 1 if rejection_action else self.pn.num_nodes)

    def step(self, action):
       return NotImplementedError

    def compute_reward(self,):
        return NotImplementedError

    def get_observation(self):
        return NotImplementedError

    def get_info(self):
        info = {**self.recorder.state, **self.curr_solution.to_dict()}
        return info

    def generate_action_mask(self):
        candidate_nodes = self.controller.find_candidate_nodes(self.pn, self.curr_vn, self.curr_vnf_idx, filter=self.selected_pn_nodes)
        if self.rejection_action:
            candidate_nodes.append(self.pn.num_nodes)
            mask = np.zeros(self.pn.num_nodes + 1, dtype=bool)
        else:
            mask = np.zeros(self.pn.num_nodes, dtype=bool)
        mask[candidate_nodes] = True
        return mask


class PlaceStepRLEnv(Environment, RLEnv):

    def __init__(self, pn, vn_simulator, recorder, verbose=False, rejection_action=False, **kwargs):
        Environment.__init__(self, pn, vn_simulator, recorder, verbose=verbose, **kwargs)
        RLEnv.__init__(self, rejection_action=rejection_action)

    def step(self, action):
        """
        Two stage: Node Mapping and Link Mapping

        All possible case
            Early Rejection: (rejection_action)
            Uncompleted Success: (Node place)
            Completed Success: (Node Mapping & Link Mapping)
            Falilure: (not Node place, not Link mapping)
        """
        pid = int(action)
        # Reject Actively
        if self.rejection_action and pid == self.pn.num_nodes:
            self.rollback_for_failure(reason='reject')
        else:
            assert pid in list(self.pn.nodes)
            # Try Deploy
            # Stage 1: Node Mapping
            node_place_result = self.controller.place(self.curr_vn, self.pn, self.curr_vnf_idx, pid, self.curr_solution)
            self.curr_vnf_idx += 1
            # Case 1: Node Place Success / Uncompleted
            if node_place_result and self.curr_vnf_idx < self.curr_vn.num_nodes:
                info = self.get_info()
                return self.get_observation(), self.compute_reward(info), False, info
            # Case 2: Node Place Failure
            if not node_place_result:
                self.rollback_for_failure(reason='place')
            # Stage 2: Link Mapping
            # Case 3: Try Link Mapping
            if node_place_result and self.curr_vnf_idx == self.curr_vn.num_nodes:
                link_mapping_result = self.controller.link_mapping(self.curr_vn, self.pn, solution=self.curr_solution, ordered_vn_edges=list(self.curr_vn.edges), 
                                                                    available_network=True, shortest_method='first_shortest', k=1, inplace=True)
                # Link Mapping Failure
                if not link_mapping_result:
                    self.rollback_for_failure(reason='route')
                # Success
                else:
                    self.curr_solution['result'] = True

        record = self.recorder.count(self.curr_vn, self.pn, self.curr_solution)
        reward = self.compute_reward(record)

        extra_info = {'curr_vn_reward': self.curr_vn_reward, 'cumulative_reward': self.cumulative_reward}
        record = self.recorder.add_record(record, extra_info)
        # Leave events transition
        deploy_success = record['result']
        deploy_failure = not record['place_result'] or not record['route_result']
        if deploy_success or deploy_failure:
            done = self.transit_obs()
        else:
            done = False

        return self.get_observation(), reward, done, record


class JointPRStepRLEnv(Environment, RLEnv):

    def __init__(self, pn, vn_simulator, recorder, verbose=False, rejection_action=False, **kwargs):
        Environment.__init__(self, pn, vn_simulator, recorder, verbose=verbose, **kwargs)
        RLEnv.__init__(self, rejection_action=rejection_action)

    def step(self, action):
        """
        Joint Place and Route with action pn node.

        All possible case
            Uncompleted Success: (Node place and Link route successfully)
            Completed Success: (Node Mapping & Link Mapping)
            Falilure: (Node place failed or Link route failed)
        """
        pid = int(action)

        # Reject Actively
        if self.rejection_action and pid == self.pn.num_nodes:
            self.rollback_for_failure(reason='reject')
        else:
            assert pid in list(self.pn.nodes)
            place_and_route_result = self.controller.place_and_route(self.curr_vn, self.pn, self.curr_vnf_idx, pid, self.curr_solution, 
                                                available_network=True, shortest_method='first_shortest', k=1)
            # Step Failure
            if not place_and_route_result:
                failure_reason = self.get_failure_reason(self.curr_solution)
                self.rollback_for_failure(failure_reason)
            else:
                self.curr_vnf_idx += 1
                # VN Success ?
                if self.curr_vnf_idx == self.curr_vn.num_nodes:
                    self.curr_solution['result'] = True
                # Step Success
                else:
                    info = self.get_info()
                    return self.get_observation(), self.compute_reward(info), False, info

        record = self.recorder.count(self.curr_vn, self.pn, self.curr_solution)
        reward = self.compute_reward(record)
        extra_info = {'curr_vn_reward': self.curr_vn_reward, 'cumulative_reward': self.cumulative_reward}
        record = self.recorder.add_record(record, extra_info)

        # Leave events transition
        if not place_and_route_result or self.curr_solution['early_reject'] or self.curr_solution['result']:
            done = self.transit_obs()
        else:
            done = False

        return self.get_observation(), reward, done, record


class SolutionStepRLEnv(Environment, RLEnv):

    def __init__(self, pn, vn_simulator, recorder, verbose=False, rejection_action=False, **kwargs):
        Environment.__init__(self, pn, vn_simulator, recorder, verbose=verbose, **kwargs)
        RLEnv.__init__(self, rejection_action=rejection_action)

    def step(self, action):
        solution = action
        # Success
        if solution['result']:
            self.curr_solution = solution
            self.curr_solution['info'] = 'Success'
        # Failure
        else:
            failure_reason = self.get_failure_reason(solution)
            self.rollback_for_failure(reason=failure_reason)

        record = self.recorder.count(self.curr_vn, self.pn, self.curr_solution)
        reward = self.compute_reward(record)
        extra_info = {'curr_vn_reward': self.curr_vn_reward, 'cumulative_reward': self.cumulative_reward}
        record = self.recorder.add_record(record, extra_info)

        done = self.transit_obs()
        return self.get_observation(), reward, done, record


class NodeSlotsStepRLEnv(Environment, RLEnv):

    def __init__(self, pn, vn_simulator, recorder, verbose=False, rejection_action=False, **kwargs):
        Environment.__init__(self, pn, vn_simulator, recorder, verbose=verbose, **kwargs)
        RLEnv.__init__(self, rejection_action=rejection_action)

    def step(self, action):
        node_slots = action

        if len(node_slots) == self.curr_vn.num_nodes:
            self.curr_solution['node_slots'] = node_slots
            link_mapping_result = self.controller.link_mapping(self.curr_vn, self.pn, solution=self.curr_solution, ordered_vn_edges=list(self.curr_vn.edges), 
                                                                available_network=True, shortest_method='first_shortest', k=1, inplace=True)
            # Link Mapping Failure
            if not link_mapping_result:
                self.rollback_for_failure(reason='route')
            # Success
            else:
                self.curr_solution['result'] = True
        else:
            self.rollback_for_failure(reason='place')

        record = self.recorder.count(self.curr_vn, self.pn, self.curr_solution)
        reward = self.compute_reward(record)
        extra_info = {'curr_vn_reward': self.curr_vn_reward, 'cumulative_reward': self.cumulative_reward}
        record = self.recorder.add_record(record, extra_info)

        done = self.transit_obs()
        return self.get_observation(), reward, done, record


if __name__ == '__main__':
    pass
    