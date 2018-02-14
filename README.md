# Repository info
This project implements an unsupervised generative modeling technique called Wasserstein Auto-Encoders (WAE), proposed by [Tolstikhin, Bousquet, Gelly, Schoelkopf (2017)](https://arxiv.org/abs/1711.01558).

# Repository structure
wae.py          -   everything specific to WAE, including encoder-decoder losses, various forms of
a distribution matching penalties, and training pipelines

run.py          -   master script to train a specific model on a selected dataset with specified hyperparameters
