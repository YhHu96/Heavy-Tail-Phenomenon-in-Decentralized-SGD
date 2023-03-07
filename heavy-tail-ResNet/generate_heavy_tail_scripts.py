# This code will generate run_heavy_tail.sh 
# which contains all commands to run the simulation
# After generating the .sh file, run "sh run_heavy_tail.sh" 
# in the command line.

import time
import os

f = open("./run_heavy_tail.sh", "w")

lr_list = ["0.02", "0.04", "0.06", "0.08", "0.1", "0.12", "0.14", "0.16", "0.18", "0.2"]

for base_lr in lr_list:

    # ====== ring with DSGD
    epochs = "200"
    # dist_optimizer = "gradient_allreduce" # this is for centralized SGD
    dist_optimizer = "neighbor_allreduce"   # this is for decentralizied SGD
    # dist_optimizer = "empty"              # this is for disconnected SGD
    seed = "42"
    topology = "ring" # can be star, grid, hypercube, fully, 
    momentum = "0.0" 
    wd = "3e-4"
    bs = "8"

    timearray=time.localtime(float(time.time()))
    tt=time.strftime('%Y-%m-%d-%H-%M-%S',timearray)
    filename="heavy_tail_epochs" + epochs + "_seed_" + seed + "_optimizer_" + dist_optimizer + "_topology_" + topology \
        + "_base_lr_" + base_lr + "_bs_" + bs + "_momentum_" + momentum + "_wd_" + wd + "_" + tt + ".log"

    # run decentralized SGD with 24 nodes
    command = "BLUEFOG_OPS_ON_CPU=1 bfrun -np 24 python decentralized_heavy_tail.py --epochs " + epochs + " --dist-optimizer " + dist_optimizer \
        + " --seed " + seed + " --dirichlet-beta -1 --nu 1 --topology " + topology + " --base-lr " + base_lr + " --batch-size " + bs + " --momentum " + momentum \
        + " --wd " + wd + " | tee " + filename + "\n"

    filepath = "./scripts"
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    f = open("./run_heavy_tail.sh", "a")
    f.write(command)