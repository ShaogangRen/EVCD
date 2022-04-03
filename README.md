# EVCD
This is a demo implementation of paper  'Causal Discovery with Flow-based Conditional Density Estimation', ICDM2021.

## Run the code

To run the code, you need to install Causal Discovery Toolbox (CDT)
(https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/index.html).

Run the grid search example code:

```
python3 run_evcd_grid.py
```
## Setups
You can modify run_evcd_grid.py to reset the dataset, learning rate, etc. 

For the tuebingen dataset in Causal Discovery Toolbox, the setup in run_evcd_grid.py
can achieve a better result (0.701) than the value reported in the paper.

Some of the code files were adapted based on ffjord (https://github.com/rtqichen/ffjord).


## Citation

```
@inproceedings{ren2021causal,
  title={Causal Discovery with Flow-based Conditional Density Estimation},
  author={Ren, Shaogang and Yin, Haiyan and Sun, Mingming and Li, Ping},
  booktitle={2021 IEEE International Conference on Data Mining (ICDM)},
  pages={1300--1305},
  year={2021},
  organization={IEEE}
}
```
