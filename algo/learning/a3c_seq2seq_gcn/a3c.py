import os
import gym
import copy
import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical


from net import ActorCritic
from buffer import RolloutBuffer


class SharedAdam(optim.Adam):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # state initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

    def share_memory(self):
        # share in memory
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)


class BaseAgent:
    """
    An Asynchronous Actor Critic-based reinforcement learning algorithm, 
    using n-step Temporal Difference Estimator to calculate gradients.
    """
    def __init__(self, 
                 gamma=0.99, 
                 coef_critic_loss=0.5,
                 coef_entropy_loss=0.00,
                 max_grad_norm=0.5,
                 norm_advantage=True,
                 clip_grad=True):
        self.device = torch.device('cpu')
        
        self.gamma = gamma
        self.coef_critic_loss = coef_critic_loss
        self.coef_entropy_loss = coef_entropy_loss  # unsupported currently
        self.max_grad_norm = max_grad_norm
        self.norm_advantage = norm_advantage
        self.clip_grad = clip_grad

        self.criterion_cirtic = nn.MSELoss()
        self.buffer = RolloutBuffer()

    def preprocess_obs(self, obs):
        return torch.FloatTensor(obs).to(self.device).unsqueeze(0)

    def select_action(self, obs, sample=True):
        action_logits = self.policy.act(obs)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        if sample:
            action = dist.sample()
        else:
            action = action_probs.argmax(-1)
        action_logprob = dist.log_prob(action)
        # collect
        self.buffer.observations.append(obs)
        self.buffer.actions.append(action)
        self.buffer.action_logprobs.append(action_logprob)
        return action.item()

    def estimate_obs(self, obs):
        value = self.policy.estimate(obs)
        return value.squeeze(-1)

    def train(self, mode=True):
        self.policy.train(mode=mode)

    def eval(self):
        self.policy.eval()


class Master(BaseAgent):

    def __init__(self,
                 obs_dim, 
                 action_dim, 
                 embedding_dim=64,
                 gamma=0.99,
                 lr_actor=1e-3,
                 lr_critic=1e-3,
                 coef_critic_loss=0.5,
                 coef_entropy_loss=0.00,
                 max_grad_norm=0.5,
                 save_dir='save/a3c',
                 norm_advantage=True,
                 clip_grad=True,
                 verbose=True):
        super(Master, self).__init__(gamma, coef_critic_loss, coef_entropy_loss, 
                                        max_grad_norm, norm_advantage, clip_grad)
        print(f'Using {self.device.type}\n')
        self.policy = ActorCritic(obs_dim, action_dim, embedding_dim).to(self.device)
        self.optimizer = SharedAdam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
        ])

        if not os.path.exists(save_dir): 
            os.mkdir(save_dir)
        self.save_dir = save_dir

        self.verbose = verbose  # if opening tqdm, suggest setting verbose == False
        
        self.update_time = mp.Value('i', 0)

    def share_memory(self):
        self.policy.share_memory()
        self.optimizer.share_memory()

    def save(self, fname='model', epoch_idx=0):
        checkpoint_path = os.path.join(self.save_dir, f'{fname}-{epoch_idx}.pkl')
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'lr_scheduler_state_dict': self.lr_scheduler.state_dict()
            }, checkpoint_path)
        print(f'Save checkpoint to {checkpoint_path}\n')

    def load(self, checkpoint_path):
        try:
            print(f'Loading checkpoint from {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            print(f'Load successfully!\n')
        except:
            print(f'Load failed! Initialize parameters randomly\n')


class Worker(mp.Process, BaseAgent):

    def __init__(self, master, rank, lock):
        mp.Process.__init__(self, name=f'worker-{rank:02d}')
        BaseAgent.__init__(self, master.gamma, master.coef_critic_loss, master.coef_entropy_loss, 
                            master.max_grad_norm, master.norm_advantage, master.clip_grad)
        self.master = master
        self.rank = rank
        self.lock = lock
        self.policy = copy.deepcopy(master.policy)
        self.optimizer = master.optimizer
        self.verbose = True if rank == 0 else False

    def sync_from_master(self):
        self.policy.load_state_dict(self.master.policy.state_dict())

    def share_grads_to_master(self):
        for param, shared_param in zip(self.policy.parameters(), self.master.policy.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def update(self, next_obs):
        action_logprobs = torch.cat(self.buffer.action_logprobs, dim=-1)
        masks = torch.IntTensor(self.buffer.masks).to(self.device)
        rewards = torch.FloatTensor(self.buffer.rewards).to(self.device)
        # estimate observations
        self.buffer.observations.append(next_obs)
        observations = torch.cat(self.buffer.observations, dim=0)
        values = self.estimate_obs(observations)
        # calculate expected return (n-step Temporal Difference Estimator)
        returns = torch.zeros_like(rewards).to(self.device)
        for i in reversed(range(len(rewards))):
            pre_return = values[-1].detach() if i == len(rewards)-1 else returns[i + 1]
            returns[i] = rewards[i] + self.gamma * pre_return * masks[i]
        # calculate advantage 
        advantage = returns - values[:-1].detach()
        if self.norm_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-9)
        # calculate loss = actor_loss + critic_loss
        # actor_loss = - (advantage * action_log_prob)
        actor_loss = - (advantage * action_logprobs).mean()
        # critic_loss = MSE(returns, values)
        critic_loss = self.criterion_cirtic(returns, values[:-1])
        loss = actor_loss + self.coef_critic_loss * critic_loss
        # update parameters
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad:
            actor_grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            critic_grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        self.share_grads_to_master()
        self.optimizer.step()
        self.sync_from_master()
        # self.lr_scheduler.step()
        with self.lock:
            self.master.update_time.value += 1
        self.buffer.clear()
        return loss.detach()


def train_worker(env, master, rank, lock, batch_size=64, num_epochs=100, start_epoch=0, max_step=200, render=False):
    cumulative_rewards = []
    env = copy.deepcopy(env)
    worker = Worker(master, rank, lock)
    for epoch_idx in range(start_epoch, start_epoch + num_epochs):
        worker.train()
        one_epoch_rewards = []
        obs = env.reset()
        for step_idx in range(max_step):
            env.render() if render and worker.rank == 0 else None
            action = worker.select_action(worker.preprocess_obs(obs))
            next_obs, reward, done, info = env.step(action)
            # collect experience
            worker.buffer.rewards.append(reward)
            worker.buffer.masks.append(not done)
            one_epoch_rewards.append(reward)
            # obs transition
            obs = next_obs
            # update model
            if worker.buffer.size() == batch_size:
                worker.update(worker.preprocess_obs(obs))
            # episode done
            if done:
                cumulative_rewards.append(sum(one_epoch_rewards))
                print(f'{worker.name} | epoch {epoch_idx:3d} | cumulative reward (max): {cumulative_rewards[-1]:4.1f} ' + 
                    f'({max(cumulative_rewards):4.1f})') if worker.verbose else None
                break
    env.close()

def train(env, master, num_processes=4, batch_size=64, num_epochs=10, start_epoch=0, max_step=200, render=False):
    assert num_processes <= mp.cpu_count()
    master.share_memory()
    lock = mp.Lock()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train_worker, args=(env, master, rank, lock, batch_size, num_epochs, start_epoch, max_step, render))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    master.save(epoch_idx=num_epochs * num_processes-1)

def evaluate(env, master, checkpoint_path, num_epochs=10, max_step=200, render=False):
    master.load(checkpoint_path)
    master.eval()
    cumulative_rewards = []
    for epoch_idx in range(num_epochs):
        one_epoch_rewards = []
        obs = env.reset()
        for step_idx in range(max_step):
            env.render() if render else None
            action = master.select_action(master.preprocess_obs(obs), sample=False)
            next_obs, reward, done, info = env.step(action)
            # collect experience
            one_epoch_rewards.append(reward)
            # obs transition
            obs = next_obs
            # episode done
            if done:
                cumulative_rewards.append(sum(one_epoch_rewards))
                print(f'epoch {epoch_idx:3d} | cumulative reward (max): {cumulative_rewards[-1]:4.1f} ' + 
                    f'({max(cumulative_rewards):4.1f})')
                break
    env.close()


if __name__ == '__main__':
    # os.environ['OMP_NUM_THREADS'] = '1'
    # config
    env_name = 'CartPole-v0'
    num_processes = 4
    embedding_dim = 64
    num_epochs = 100
    start_epoch = 0
    batch_size = 64
    max_step = 200

    # initialize
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    master = Master(obs_dim=obs_dim, action_dim=action_dim, embedding_dim=embedding_dim)

    # train
    train(env, master, num_processes=num_processes, batch_size=batch_size, 
        num_epochs=num_epochs, start_epoch=start_epoch, max_step=max_step, render=False)

    # evaluate
    checkpoint_path = f'save/a3c/model-{num_epochs*num_processes-1}.pkl'
    evaluate(env, master, checkpoint_path, num_epochs=10, max_step=max_step, render=True)