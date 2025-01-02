# DRL-SFCP

This is the implementation of our paper titled "[DRL-SFCP: Adaptive Service Function Chains Placement with Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/9500964)", which is accepted by ICC 2021.


**Note: 
This algorithm has been integrated into [Virne](https://github.com/GeminiLight/virne), an NFV simulator, where you can find more details.**

## Installation

```shell
# only cpu
bash install.sh -c 0

# use cuda (optional version: 10.2, 11.3)
bash install.sh -c 11.3
```

## Quick Start

```shell
python main.py --solver_name=$SOLVER_NAME
```

Here, you can choose `SOLVER_NAME` from `a3c_gcn_seq2seq`, `grc_rank`, `mcts`, etc. And you can find more detailed usage in `main.py` and `config.py`.

## Simulation Settings

Please refer to `settings/p_net_setting.yaml` and `settings/v_sim_setting.yaml` for more details.

## Citation

If you find this code useful in your research, please consider citing it:

```bibtex
@INPROCEEDINGS{tfw-icc-2021-drl-sfcp,
  author={Wang, Tianfu and Fan, Qilin and Li, Xiuhua and Zhang, Xu and Xiong, Qingyu and Fu, Shu and Gao, Min},
  booktitle={ICC 2021 - IEEE International Conference on Communications}, 
  title={DRL-SFCP: Adaptive Service Function Chains Placement with Deep Reinforcement Learning}, 
  year={2021},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/ICC42927.2021.9500964}
}
```

## More Resources

- [Virne: An Comprehensive Simulator for Resource Allocation in NFV Networks](https://github.com/GeminiLight/virne)
- [SDN-NFV Papers on Resource Management](https://github.com/GeminiLight/sdn-nfv-papers)
