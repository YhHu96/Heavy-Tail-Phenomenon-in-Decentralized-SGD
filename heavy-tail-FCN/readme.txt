This code is to plot tail index figures for 3fcn with MNIST.
To run the experiment results of 3fcn on MNIST, you need to:

1. Run 3fcn-MNIST.py and 3fcn-MNIST-decentralized.py. These python codes do not require cuda and GPU.

2. These two codes will save several models under certain folders.

3. Use calculate_tail_index_MNIST.ipynb to calculate the tail index of the different network by passing the folders' directory in the notebook.

4. Use plot_MNIST_figure4(a).ipynb to generate Figure in the paper.
