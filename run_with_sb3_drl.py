import os
# os.chdir(os.path.join(os.getcwd(), 'code/virne'))

from config import get_config, show_config, save_config

from data.generator import Generator


def load_algo(config):
    algo_name = config.algo_name
    # deep_rl
    if algo_name == 'ppo_conv':
        from algo.learning.ppo_conv.ppo_conv_agent import PPOConv
        env, agent = PPOConv.from_config(config)
    elif algo_name == 'a3c_gcn':
        from algo.learning.a3c_gcn import A3CGCN
        env, agent = A3CGCN.from_config(config)
    elif algo_name == 'ppo_gat':
        from algo.learning.ppo_gat import PPOGAT
        env, agent = PPOGAT.from_config(config)
    elif algo_name == 'ppo_conv_time_reward':
        from algo.learning.ppo_conv_time_reward import PPOConvTimeReward
        env, agent = PPOConvTimeReward.from_config(config)
    else:
        raise NotImplementedError
    # Create env and agent
    return env, agent

def train(env, agent, config, total_timesteps=1000000):
    # learn
    agent.learn(total_timesteps=total_timesteps)
    # save
    agent.save(os.path.join(config.save_dir, config.algo_name))
    return env, agent

def test(env, agent, config):
    obs = env.reset()
    epoch_reward = 0

    while True:
        mask = env.generate_action_mask()
        action, _states = agent.predict(obs, action_masks=mask, deterministic=True)
        next_obs, reward, done, info = env.step(action)

        epoch_reward += reward
        obs = next_obs

        if done: break

    print(f'cumulative reward in test: {epoch_reward}')


def run(config):
    # Load environment and algorithm
    env, agent = load_algo(config)
    
    # Run agent and env
    print(f"\n{'-' * 20}    Start     {'-' * 20}\n")

    train(env, agent, config, total_timesteps=1000000)

    print(f"\n{'-' * 20}     Test     {'-' * 20}\n")

    test(env, agent, config)

    print(f"\n{'-' * 20}   Complete   {'-' * 20}\n")

if __name__ == '__main__':
    config = get_config()

    # Generator.generate_dataset(config)

    config.verbose = 1
    # ppo_conv, a3c_gcn
    config.algo_name = 'ppo_gat'
    run(config)
