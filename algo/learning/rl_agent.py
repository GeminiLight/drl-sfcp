import os
import torch
import numpy as np
from abc import abstractmethod
from torch.utils.tensorboard import SummaryWriter

from .buffer import RolloutBuffer
from ..base import Agent


class RLAgent(Agent):

    def __init__(self,
                 name,
                 gamma=0.9,
                 max_grad_norm=1.,
                 save_dir='save',
                 log_dir='logs',
                 use_cuda=True,
                 allow_parallel=False,
                 reusable=False, 
                 verbose=1,
                 open_tb=True,
                 **kwargs):
        super(RLAgent, self).__init__(name, reusable=reusable, verbose=verbose, **kwargs)
        self.device = torch.device('cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.allow_parallel = allow_parallel
        self.save_dir = save_dir
        self.writer = SummaryWriter(log_dir) if open_tb else None
        self.buffer = RolloutBuffer()
        if not os.path.exists(self.save_dir): 
            os.makedirs(self.save_dir)

    @abstractmethod
    def train(self, env, num_epochs=1, start_epoch=0):
        return NotImplementedError

    @abstractmethod
    def validate(self, env, checkpoint_path):
        return NotImplementedError

    @abstractmethod
    def preprocess_obs(self, observation):
        return NotImplementedError

    @abstractmethod
    def select_action(self, observation, mask=None, sample=True):
        return NotImplementedError
    
    def log(self, time_step, info_dict_list):
        for info_dict in info_dict_list:
            self.writer.add_scalar(f"{info_dict['mode']}_{info_dict['group']}/{info_dict['name']}", info_dict['value'], time_step)

    def save_net(self, checkpoint_path):
        return NotImplementedError
   
    def load_net(self, checkpoint_path):
        return NotImplementedError

    def run(self, env, num_epochs=1, start_epoch=0):
        for epoch_idx in range(start_epoch, start_epoch + num_epochs):
            print(f'Epoch {epoch_idx}') if self.verbose >= 1 else None
            obs = env.reset()
            while True:
                mask = np.expand_dims(env.generate_action_mask(), axis=0)
                action = self.select_action(obs, mask=mask, sample=False)
                next_obs, reward, done, info = env.step(action)
                if done:
                    break
                obs = next_obs



class PGAgent(RLAgent):

    def __init__(self, name, gamma=0.9, max_grad_norm=1, save_dir='save', log_dir='logs', use_cuda=True, allow_parallel=False, reusable=False, verbose=1, open_tb=True, **kwargs):
        super().__init__(name, gamma=gamma, max_grad_norm=max_grad_norm, save_dir=save_dir, log_dir=log_dir, use_cuda=use_cuda, allow_parallel=allow_parallel, reusable=reusable, verbose=verbose, open_tb=open_tb, **kwargs)
    
    def update(self):
        pass