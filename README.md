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
cd fedland # This repo's root, not the fedland package
docker compose up --build -d
```
2. Set the combiner host so local clients can resolve to docker

__macOS/Linux__
```bash
# Edit hosts file with sudo privileges
sudo nano /etc/hosts

# Add this line to the file:
127.0.0.1       combiner

# Save and exit:
# For nano: Ctrl + X, then Y, then Enter
# For vi: Esc, then :wq, then Enter

# Test the configuration
ping combiner
```

__Windows__
```powershell
# Open PowerShell as Administrator and edit hosts file
notepad C:\Windows\System32\drivers\etc\hosts

# Add this line to the file:
127.0.0.1       combiner

# Save the file (may need to confirm overwrite)

# Test the configuration
ping combiner
```

3. Define the `Experiments` you want to run in `main.py`
```python
# main.py
ROUNDS = 10  # How many communication rounds to do
CLIENT_LEVEL = 3
EXPERIMENTS = [
    Experiment(
        id=str(uuid.uuid4()),
        description="EXAMPLE: CIFAR-10, uneven classes",
        dataset_name=DatasetIdentifier.CIFAR.value,
        model="CifarFedNet",
        timestamp=datetime.now().isoformat(),
        target_balance_ratios=[
            [0.01] * 10,
            [
                float(x)
                for x in (
                    np.exp(-0.07 * np.arange(10)) / sum(np.exp(-0.07 * np.arange(10)))
                )
            ],
        ],
        client_stats=[],
        aggregator="fedavg"  # OR "fedopt"
    ),
]
```

4. Run main.py
```python
python main.py
# Clients get started and are run based on your env
```

5. Wait and watch
- You can verify the process is running by seeing the clients' logs which are dumped to `debug.log`
- You can also watch the output in `results/`
    - `results/experiments.json` is the key for experiment ids which
    each experiment has directory linked to its ID. Within each experiment
    directory there are subdirs marking the client id.
    - within `results/<experiemnt id>/<client id>/` there are the training results
    as well as the clients info so we can verify the results are what we expect them
    to be.
- Your cpu will likely go brr on validation rounds (good way to tell whats happening 
while you work on other things and wait.)

### Locally (plus with FEDn Studio)
_I have not tested this in a while_
Requirements: [poetry](https://python-poetry.org/)
1. Setup a venv
2. Run `poetry install --with dev` in venv
3. Create and fill in the .env fields based on .env.example
    - run: `cp .env.example .env`
4. Run whatever scripts you want in /scripts
    - eg: `python scripts/centralized_base.py`
5. Follow FEDn's getting started as if this home dir were a directory in the pytorch-mnist example


## Helpful Commands:

### Running clients manually (for debugging)
```bash
RESULTS_DIR=/your/abs/path/fedland/results TEST_ID=<TARGET TEST UUID> CLIENT_ID=0 fedn client start -n 0 --init settings-client-local.yaml
```

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
