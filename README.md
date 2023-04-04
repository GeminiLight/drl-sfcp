# DRL-SFCP

PyTorch implementation of our paper "[DRL-SFCP: Adaptive Service Function Chains Placement with Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/9500964)" which is accepted by ICC 2021.


**Note: 
This algorithm has been integrated into [Virne](https://github.com/GeminiLight/virne), a NFV simulator, where you can find more details.**

## Installation

```shell
# only cpu
bash install.sh -c 0

# use cuda (optional version: 10.2, 11.3)
bash install.sh -c 11.3
```

## Quick Start

```shell
python main.py
```

You can find more detailed usage in `main.py` and `config.py`.

## Simulation Settings

Please refer to `settings/p_net_setting.yaml` and `v_sim_setting.yaml` for more details.

## Citation

If you find this code useful in your research, please consider citing:

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