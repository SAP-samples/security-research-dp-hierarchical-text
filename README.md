# Differentially Private Hierarchical Text Classification

[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/security-research-dp-hierarchical-text)](https://api.reuse.software/info/github.com/SAP-samples/security-research-dp-hierarchical-text)

## Description

SAP Security Research sample code to reproduce the research done in our paper "On the privacy-utility trade-off in
differentially private hierarchical text classification"[1].

## Requirements

- [Python](https://www.python.org/) 3.7
- [Tensorflow Privacy](https://github.com/tensorflow/privacy)
- [Tensorflow](https://github.com/tensorflow)
- [transformers](https://github.com/huggingface/transformers/)
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [pandas](https://pandas.pydata.org/)
- [anytree](https://anytree.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)
- [tqdm](https://tqdm.github.io/)

## Download and Installation

### Differentially Private Hierarchical Text Classification Framework

Implementation of several hierarchical text classification (HTC) neural networks
and the corresponding differential privacy (DP) adversary to quantify information leakage in the trained HTC models.

### Install

Running `make install` in the root folder should be enough for most use cases.

The command will create the basic project directory structure and installs dph as well as other requirements.
You can use pip as your package manager and install the `dph` package via `python -m pip install -e ./`
For other package managers you need to install dpa using `setup.py`.

### Directory Structure

After having run `make install`, the following directory structure should be created in your local 
file system. Note: Everything that must not be tracked by git is already in `.gitignore`.

```
DPAttack/
     |-- Makefile
     |-- setup.py
     |-- requirements.txt
     |-- data/              # dataset files
     |-- logs/		        # experiment logs 
     |-- notebooks/         # evaluation notebooks
     |-- dph/			    # source root
          |--core/	            # hierarchical text classification framework
          |--mia/	            # membership inference attack framework
          |--projects/	        # project implementations using dph
          |--utils/             # utility modules

```

### Using the Framework

For every dph project (dataset), a subdirectory in `./dph/projects` is recommended.
Inside the folder, a subclass of an [`HTC Experiment`](dph/core/experiment.py) should be created as it is done for the existing datasets.
To train an HTC model, call the `start` method of an experiment.
To train a DP-HTC model, pass an instance of [`DPParameters`](dph/core/parameters/parameters.py) to the `Experiment` constructor.

To train an attack model for an (DP-)HTC model, a [`AttackExperiment`](dph/mia/mia_experiment.py) has to be created.
To start an attack, call the `start` method of the attack experiment.

### Reproducing the tables and diagrams of the paper

In the following, we explain how the different tables and diagrams of the paper can be reproduced. 
Running the python scripts for [BestBuy], [Reuters] and [DBPedia] will execute the different project's experiments.
The scripts' parameters can be adapted to execute all possible scenarios that we examined in the paper. 
The resulting experiment dictionary in the `logs` folder will contain several json files with the target/attack model metrics (train and test).
The results can then be retrieved and plotted in a notebook, as we have done in [`notebooks/plots-dp.ipynb`](notebooks/plots-dp.ipynb).

[BestBuy]: dph/projects/bestbuy/mia/experiment.py
[Reuters]: dph/projects/reuters/mia/experiment.py
[DBPedia]: dph/projects/dbpedia/mia/experiment.py

### Contributors

- Dominik Wunderlich
- Daniel Bernau

## Citations
If you use this code in your research, please cite:

```
@article{WBA+21,
      author    =   {Dominik Wunderlich and 
                    Daniel Bernau and 
                    Francesco Aldà and 
                    Javier Parra-Arnau and 
                    Thorsten Strufe},
      title     =   {On the privacy-utility trade-off in differentially private hierarchical text classification}, 
      eprint    =   {2103.02895},
      archivePrefix={arXiv},
      url       =   {http://arxiv.org/abs/2103.02913},
}
@article{BKG+21,
}
```

## References
[1] Dominik Wunderlich, Daniel Bernau, Francesco Aldà, Javier Parra-Arnau, Thorsten Strufe:
On the privacy-utility trade-off in differentially private hierarchical text classification.
arXiv:2103.02895
http://arxiv.org/abs/2103.02913

## License

Copyright (c) 2021 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache
Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSES/Apache-2.0.txt) file.
