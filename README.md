# Fedland: Loss Landscapes in Federated Learning
## Summary:
Loss landscapes in the federated learning [1] setting are relatively more complex and often fragmented as compared to the centralized setting, owing to variations in local data distributions [2]. The project will study and contrast local and global loss landscapes to derive insights about a local client’s contribution to the overall loss landscape. Understanding these dynamics can potentially enhance our ability to optimize federated learning systems.

## Task:
- Training a neural network model in two settings:
    1. In a centralized environment
    2. In a federated environment

- Conduct experiments with the federated setting focusing on analyzing the loss landscapes of the models trained in these settings, including both the global model and individual client-side local models.

## Project Layout:

Since we are using FEDn, we use their recommended project structure.

I have added pyproject.toml for managing our dependencies. I have written it to use [Poetry](https://python-poetry.org/). You probably want to use an virtual environment or tool to manage your virtualenv. To install the dependencies using poetry, simply run `poetry install`. I set it to use python 3.11

I have also added a directory /core to add shared pieces of logic to import for our tests (assuming FEDn allows me to import and package my own modules).
```txt
project
├ client
│ ├ fedn.yaml
│ ├ python_env.yaml
│ ├ model.py
│ ├ data.py
│ ├ train.py
│ └ validate.py
├ data
│ └ mnist.npz
├ fed_land
│ └ __init__.py
├ core
├ README.md
├ pyproject.toml
├ scripts / notebooks
└ Dockerfile
```
