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

I have added pyproject.toml for managing dependencies. I have written it to use [Poetry](https://python-poetry.org/). You probably want to use an virtual environment or tool to manage your virtualenv locally. This will give you code completion and allow you to run `main.py` and any other scripts outside docker (if you want). To install the dependencies using poetry, simply run `poetry install`. I set it to use python 3.11

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
│ ├ /db/mongo.files
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

I also thought it would be nice to define and queue experiments in a declarative fashion so you can also
log everything to the centralized store (mongo since it's already there in FEDn's stack).
This way anyone who wants to run tests can define them all log their experiments to
the same place and make sure calculations and data processing are consistent across experiments.

If you just want to use fedland metrics and such (its pretty bare rn) instal with `pip install fedland`


## Running the project:
### In docker (psuedo distributed mode)
1. Start everything
```bash
cd fedland
docker compose up --build -d
```

2. Define the `Experiments` you want to run in `main.py`
```python
# main.py
ROUNDS = 10  # How many communication rounds to do
CLIENT_LEVEL = 3
EXPERIMENTS = [
        Experiment(
            id="",  # gets generated by mongo
            description="This is an example experiment on MNIST with non-IID and imbalanced clients.",
            dataset_name=DatasetIdentifier.MNIST,
            model="FedNet",
            timestamp=datetime.now().isoformat(),
            target_balance_ratios=[  # which classes each client should have
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # len 10 since MNIST has 10 digits
                [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0]
                ],
            subset_fractions=[1, 0.8, 0.3],  # how much data each client should have
            client_stats=[],  # optional
            ), #... add all the other experiments you want to run 
        ]
```

3. Run main.py and scale clients
```python
python main.py
# start however many clients your experiment requires
docker compose scale client=3
```

4. Wait and watch
- You can verify the process is running by seeing the clients' logs using
```python
docker compose logs client -f
```
- More clients takes longer to run. In general I check the database to see how it's coming along
    - The database is at [localhost:8081](http://localhost:8081/db/fedn-network)
    - Login=`fedn_admin` and password=`password` and are also in `compose.yaml`
    - Terrible naming, but the training stats are in `local.client_stats` and you can watch the experiment data come in.

### Locally (plus with FEDn Studio)
Requirements: [poetry](https://python-poetry.org/)
1. Setup a venv
2. Run `poetry install --with dev` in venv
3. Create and fill in the .env fields based on .env.example
    - run: `cp .env.example .env`
4. Run whatever scripts you want in /scripts
    - eg: `python scripts/centralized_base.py`
5. Follow FEDn's getting started as if this home dir were a directory in the pytorch-mnist example


## Helpful Commands:

### Running tests with:
```bash
poetry run pytest tests
```
OR in docker container with 
```bash
docker compose run --build client poetry run pytest tests/
```

### Running Mongo Shell (mongo db instance)
The default db is the same as the network id (fedn-network)
```bash
mongosh "mongodb://0.0.0.0:6534/?authSource=admin" --apiVersion 1 --username fedn_admin
```
The db can also be viewed using [Mongo Express webapp here](http://localhost:8081/db/fedn-network/)

Backup/Restore the database inside the docker container with:
[Backup File Here 10/31 _Spooky_](https://drive.google.com/file/d/127J3TzYofcE0dxDgYN9Yzes0HptyxMMx/view?usp=drive_link)
```bash
# backup
mongodump -d fedn-network -u fedn_admin -p password --port=6534 --out=/data/db/backup
# restore
mongorestore /data/backup/fedn-network --port=6534 -p password
```
* Note that from within the proj dir, we have the shared volume mounted to ./data 

_As a footnote: When I started writing this I thought "That'll be easy to automate the experiments. Then I can just define them and then let them run overnight and lazily scale the clients and rerun". It didn't turn out to be so easy so sorry for the complexity and code slop. It just turned into "Make it work" about 3/4 the way through._
