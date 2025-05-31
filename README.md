# Model Guidance via Robust Feature Attribution

A repository containing the implementation and experiments for the paper "Model Guidance via Robust Feature Attribution".

## Installation

This repository is built on top of a codebase containing the implementation of Abstract Gradient Training (AGT) methods for certified poisoning, unlearning, and privacy in gradient-based training. As such, before using this repository, one needs to follow the installation procedure of [`AbstractGradientTraining` package](https://github.com/psosnin/AbstractGradientTraining):

Install the package using pip:
```pip install git+https://github.com/psosnin/AbstractGradientTraining```

## Repository Description and Usage

The core of this repository resides under the `R4/` folder. One can find the datasets used in the experiments under `R4/datasets`, the network models and robust regularizer under `R4/models`, the plots and figures included in the paper (as well as some extra figures) under `R4/plots`, and the final experimental results reported in the paper under `R4/experiment_results`. The latter contains the results and hyperparameters for each dataset in the format `<DATASET NAME>.yaml` and `<DATASET NAME>_params.yaml`, respectively, as well as ablations' results. Lastly, the files under `R4/` root are either logic for ablations and metrics, or notebooks which have been used to generate the experimental results.

To reproduce results, just run the corresponding dataset notebook in `R4/` folder. For example, to reproduce the results for the DecoyMNIST dataset, run `R4/DECOY MNIST - R4.ipynb`. The notebooks are designed to be self-contained and will automatically load the necessary data and models. Before running a regularization method, be sure to double check all the right hyperparameters match the ones in the corresponding `R4/experiment_results/<DATASET NAME>_params.yaml` file.

## References

- [Model Guidance via Robust Feature Attribution](https://www.example.com/)
- [Certified Robustness to Data Poisoning in Gradient-Based Training](https://arxiv.org/pdf/2406.05670v1)
- [Certificates of Differential Privacy and Unlearning for Gradient-Based Training](https://arxiv.org/abs/2406.13433)