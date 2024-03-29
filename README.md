# EVCD Causal Discovery
This is a demo implementation of the paper  'Causal Discovery with Flow-based Conditional Density Estimation', ICDM2021.
Paper: https://github.com/ShaogangRen/EVCD/blob/main/Causal%20Discovery%20with%20Flow-based%20Conditional%20Density%20Estimation.pdf
## Run the code

To run the code, you need to install Causal Discovery Toolbox (CDT)
(https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/index.html).

Run the the example code (grid search on parameters):

```
python3 run_evcd_grid.py
```
## Change the Setup
You can modify run_evcd_grid.py to reset the dataset, learning rate, etc. 

For the Tuebingen dataset in Causal Discovery Toolbox, the setup in run_evcd_grid.py
can achieve  better results (0.707) than the value reported in the paper.

Some of the code files were adapted from ffjord (https://github.com/rtqichen/ffjord).


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
