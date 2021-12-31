# import os
# os.chdir(os.path.join(os.getcwd(), 'code/virne'))

from config import get_config, show_config
from utils import load_algo
from data import Generator


def generate_dataset(config):
    Generator.generate_dataset(config)

def main(config):
    # Load environment and agent
    env, agent = load_algo(config.algo_name)

    # Run agent and env
    print(f"\n{'-' * 20}    Start     {'-' * 20}\n")

    agent.run(env, num_epochs=config.num_epochs, start_epoch=config.start_epoch)

    print(f"\n{'-' * 20}   Complete   {'-' * 20}\n")


if __name__ == '__main__':
    assert 1 == 2, 'The interface has not been unified.\nPlease run algorithms in run_type.py'

    config = get_config()

    # generate_dataset(config)

    config.num_epochs = 1
    config.verbose = True
    for algo_name in ['grc_rank']:
        config.algo_name = algo_name
        main(config)