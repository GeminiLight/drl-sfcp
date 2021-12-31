# DRL-SFCP

PyTorch implement of "DRL-SFCP: Adaptive Service Function Chains Placement with Deep Reinforcement Learning" which is accepted by ICC 2021.

## Run

### Generate Test Dataset

```python
Generator.generate_dataset(config)
```

### Run with config

```python
config = get_config()

agent, env = load_algo(config)

agent.run(env)
```

## Citation

```latex
@INPROCEEDINGS{9500964,
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
