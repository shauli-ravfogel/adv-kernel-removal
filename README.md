This repository contains the code for the experiments included in the paper [Kernelized Adversarial Concept Erasure](https://arxiv.org/abs/2201.12191), accepted in EMNLP 2022.

In the paper, we run a minimax concept erasure game in kernel space, in which we aim to prevent a given kernel classifier from reconstructing a given concept (e.g. gender) from neural representations.

## Algorithm
Our method is based on applying Nystrom feature mapping to approximate the kernel features, following by the adversarial formulation presented in the papr [Linear Adversarial Concept Erasure](https://arxiv.org/abs/2201.12091) (RLACE).

The file `run_kernels.py' contains the function `calc_nystrom' to calculate the nystrom features for a given kernel, and the function `calc_preimage_nystrom_mse' that calculates an approximate preiamge of the clean features in kernel space (after the removal of, e.g., gender-encoding features). The file `relaxed_inlp.py' contains an implementation of the RLACE algorithm that we run over the nystrom features. 

## Experiments

We rely on the package [MKLpy](https://github.com/IvanoLauriola/MKLpy) for multiple kernel learning.

First, run the file `run_kernels.py' in order to calcualte the nystrom mapping, run the RLACE algorithm, and store the approximate preimage features:

```
python3 run_kernels.py --run_id 1 --mode glove --device 0 --normalize 1
```

Then, run the script `eval_kernels.py' in order to train new kernel adversaries over the preimage features, and record their performance.

```
python3 eval_kernels.py --run_id 1 --normalize 1
```
    
## Evalaution

The notebook `eval_kernels.ipynb' collects the accuracies of the different kernel adversaries and generates the accurcy tabels in the paper. The notebook `additional_analysis.ipynb' contains a semantic analysis of the clean preimage vectors -- specifically, WEAT, closet-neighbors test and visualization.

