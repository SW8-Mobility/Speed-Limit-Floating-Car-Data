# Speed-Limit-Floating-Car-Data
This is a project from 8th semester software on Aalborg University. It revolves around using Floating Car Data to categorise Speed Limits within Denmark.

## Docker

This docker container was made so that the setup for the python environment for using FCD was easier

### Setup

To setup the python container
1. Copy `.env.example` and rename it `.env`
2. Insert your API key from https://fcd-share.civil.aau.dk/ into `FCD_API_KEY`
3. Run `docker compose up -d` in this directory

You can now use the terminal feature within docker desktop or use the command `docker compose exec -it python /bin/bash`

Any changes you make to `./scripts/api-script.py` will be reflected into the container instantly.

### Running a Python script in the docker container

From the root directory run
`docker compose run [OPTIONS] python <python-file>`

Useful options can be
- --rm
    - Removes the container after run. Should be used unless you need the container after runtime
- -d
    - Detached mode, aka. you can still use the terminal when running a process. Should be used if you want a process to continue running after closing your terminal

### Saving data to file

You can save data by using the append operator `>>` eg: 

```
docker compose exec -ti python python /scripts/api-script.py >> data.json
```

### folder structure
Anything todo with the pipeline, will be in the pipeline folder.
Files in the preprocessing folder are for data cleaning and processing data, for calculating our features and annotating with ground truth. 
More folders to come, for creating models...

## geo_json_metrics
script in open_street folder is for reading the open street map files, extracting road and their information.
script in geo_json_metrics is for requesting from the https://fcd-share.civil.aau.dk/ api and saving the response as json. 

## Be aware
The osm_id column contains None / null values. This may correspond to update fallouts.

## Memory issues 
The 3 dataframes are very large. For model training, perhaps create a random forest on each dataset, and combine them:

## Mypy
If errors are not shown in pull request, but mypy fails, run mypy locally with the command:
```
python -m mypy .
```

## Pytest
Run all tests:
```
python -m pytest
```

## Imports
example filestructure:
```
\A
    a.py > def func()...
\B
    \C
        c.py
```
If you want to import a.py in your c.py, you must have a __init__.py file, in the A directory.
Ways of importing:
```
from A.a import func
import A.a as ModuleAlias
```
If it does not work, run the following command in the root of the directory, to discover new modules:
```
pip install -e .
```


