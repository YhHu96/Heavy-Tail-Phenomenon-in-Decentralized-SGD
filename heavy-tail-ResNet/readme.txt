
This code is to plot tail index figures for ResNet-20 with Cifar10.
To run this code, you should have several GPUs in hand.

Step 1. Download and install BlueFog from https://github.com/Bluefog-Lib/bluefog. We strongly suggest you install it via Docker, see the instructions
in https://bluefog-lib.github.io/bluefog/docker.html. Pull the steady version gpu-0.3.0 in https://hub.docker.com/r/bluefoglib/bluefog/tags

Step 2. Run generate_heavy_tail_scripts.py to generate a scipt file "run_heavy_tail.sh". This file contains the commands to run distributed neural
network training. You can adjust hyperparameters such as learning rate, batch-size, decentralized optimizers in generate_heavy_tail_scripts.py 

Step 3. Run "sh run_heavy_tail.sh" in command line. This will run all commands in run_heavy_tail.sh and output various .mat files storing evalued
tail index. 

Step 4. Use "Plot_tail_index_figures.ipynb" to visualize tail index and plot figures.