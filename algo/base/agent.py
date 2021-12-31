import abc
import copy

from .controller import Controller


class Agent:

    def __init__(self, name, reusable=False, verbose=1, **kwargs):
        __metaclass__ = abc.ABCMeta
        self.name = name
        self.reusable = reusable
        self.verbose = verbose
        self.num_arrived_vns = 0
        self.controller = Controller()

    @classmethod
    def from_config(cls, config):
        if not isinstance(config, dict): config = vars(config)
        config = copy.deepcopy(config)
        reusable = config.pop('reusable', False)
        verbose = config.pop('verbose', 1)
        return cls(reusable=reusable, verbose=verbose, **config)

    def run(self, env, num_epochs=1, start_epoch=0):
        for epoch_idx in range(start_epoch, start_epoch + num_epochs):
            print(f'Epoch {epoch_idx}') if self.verbose >= 1 else None
            obs = env.reset()
            while True:
                action = self.select_action(obs)
                next_obs, reward, done, info = env.step(action)
                if done:
                    break
                obs = next_obs

    def select_action(obs):
        return NotImplementedError

