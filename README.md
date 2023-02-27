# python-docker

This docker container was made so that the setup for the python environment for using FCD was easier

## Setup

To setup the python container
1. Copy `.env.example` and rename it `.env`
2. Insert your API key from https://fcd-share.civil.aau.dk/ into `FCD_API_KEY`
3. Run `docker compose up -d` in this directory

You can now use the terminal feature within docker desktop or use the command `docker exec -ti python-docker-python-1 python /scripts/api-script.py`

Any changes you make to `./scripts/api-script.py` will be reflected into the container instantly.


## Saving data to file

You can save data by using the append operator `>>` eg: 

```
docker exec -ti python-docker-python-1 python /scripts/api-script.py >> data.json
```