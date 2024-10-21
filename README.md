# Fedland: Loss Landscapes in Federated Learning
## Summary:
Loss landscapes in the federated learning [1] setting are relatively more complex and often fragmented as compared to the centralized setting, owing to variations in local data distributions [2]. The project will study and contrast local and global loss landscapes to derive insights about a local client’s contribution to the overall loss landscape. Understanding these dynamics can potentially enhance our ability to optimize federated learning systems.


## Task:
- Training a neural network model in two settings:
    1. In a centralized environment
    2. In a federated environment, including:
        - Balanced/unbalanced
        - IID/non-IID

- Conduct experiments with the federated setting focusing on analyzing the loss landscapes of the models trained in these settings, including both the global model and individual client-side local models.

## Project Layout:

Since I am using FEDn, the project follows their recommended project structure.

I have added pyproject.toml for managing our dependencies. I have written it to use [Poetry](https://python-poetry.org/). You probably want to use an virtual environment or tool to manage your virtualenv. To install the dependencies using poetry, simply run `poetry install`. I set it to use python 3.11

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
├ fedland (module)
│ └ __init__.py
├ README.md
├ pyproject.toml
├ scripts / notebooks
├ compose.yaml
└ Dockerfile
```

### What is /fedland?
The idea behind writing fedland into its own package is it would abstract repetitive calculatations and 
functionality into a tiny library available on PyPi in order to pull it into clients environments.

In the future it may include logging everything to a centralized store (like mongo since it's already there). This way anyone who wants to run tests can all log their experiments to the same place and make sure calculations
and data processing are consistent across experiments.

if you just want to use fedland metrics and such (its pretty bare rn) instal with `pip install fedland`


## Running the project
### In docker:
```bash
cd fedland
docker compose up --build -d
```

### Locally (plus with FEDn Studio)
1. Setup a venv
2. Run `poetry install` in venv
3. Create and fill in the .env fields based on .env.example
    - run: `cp .env.example .env`
4. Run whatever scripts you want in /scripts
    - eg: `python scripts/centralized_base.py`
