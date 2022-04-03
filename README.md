# EVCD
This is a demo implementation of paper  'Causal Discovery with Flow-based Conditional Density Estimation', ICDM2021.

## Run the code

To run the code, you need to install Causal Discovery Toolbox (CDT)
(https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/index.html).

Run the grid search example code:

'''
python3 run_evcd_grid.py
'''

You can modify run_evcd_grid.py to reset the dataset, learning rate, etc. 

For the tuebingen dataset in Causal Discovery Toolbox, the setup in run_evcd_grid.py
can achieve a better result (0.701) than the value reported in the paper.

Some of the code files are adapted based on ffjord(https://github.com/rtqichen/ffjord).



