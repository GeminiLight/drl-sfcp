from config import get_config, show_config, save_config

from data.generator import Generator
from algo.base.environment import SolutionStepEnvironment


def load_algo(config):
    algo_name = config.algo_name
    if algo_name == 'mcts_vine':
        from algo.learning.mcts_vine import MCTSAgent
        env, agent = SolutionStepEnvironment.from_config(config), MCTSAgent.from_config(config)
    elif algo_name == 'graph_vine':
        from algo.learning.graph_vine import GraphViNEAgent
        env, agent = SolutionStepEnvironment.from_config(config), GraphViNEAgent.from_config(config)
    elif algo_name == 'neuro_vine':
        from algo.learning.neuro_vine.neuro_vine import NeuroVineAgent
        env, agent = SolutionStepEnvironment.from_config(config), NeuroVineAgent.from_config(config)
    # drl
    elif algo_name == 'pg_conv':
        from algo.learning.pg_conv import PGConv
        env, agent = PGConv.from_config(config)
        config.only_test = False
    elif algo_name == 'pg_mlp':
        from algo.learning.pg_mlp import PGMLP
        env, agent = PGMLP.from_config(config)
        config.only_test = False
    else:
        return NotImplementedError
    return env, agent


def run(config):
    # Load environment and algorithm
    env, agent = load_algo(config)
    
    # train
    if not config.only_test:
        num_train_epochs = 100
        agent.train(env, num_epochs=num_train_epochs, start_epoch=config.start_epoch)

    # Run agent and env
    print(f"\n{'-' * 20}    Start     {'-' * 20}\n")

    agent.run(env, num_epochs=config.num_epochs, start_epoch=config.start_epoch)

    print(f"\n{'-' * 20}   Complete   {'-' * 20}\n")


if __name__ == '__main__':
    config = get_config()

    # Generator.generate_dataset(config)

    config.num_epochs = 1
    config.verbose = 1
    config.algo_name = 'graph_vine'
    run(config)