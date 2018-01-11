# Repository info
This project implements an unsupervised generative modeling technique called Wasserstein Auto-Encoders (WAE), proposed by [Tolstikhin, Bousquet, Gelly, Schoelkopf (2017)](https://arxiv.org/abs/1711.01558).

# Repository structure
wae.py          -   everything specific to WAE, including encoder-decoder losses, various forms of
a distribution matching penalties, and training pipelines

run.py          -   master script to train a specific model on a selected dataset with specified hyperparameters

eval.py         -   evaluate a specified trained model, including various plots, FID scores, etc.

# Use cases

1.  Train a model
    a. Load the dataset
    b. Build a model architecture
    c. Start training model parameters by optimizing the objective
    d. Print debug information from time to time
    e. Save checkpoints regularly and make it possible to continue training from a checkpoint

    Common changes to experiments:
    a. Change a dataset
    b. Change hyperparameters and architectures (make sure to separate fixed hyperparams from those which are changed often)
    c. Change the matching penalty and the cost function
    d. Change details of debug information

2. Evaluate trained models
    a. Load the model from checkpoints
    b. Make various plots, compute various metrics
