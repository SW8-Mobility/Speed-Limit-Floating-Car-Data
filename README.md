# Speed-Limit-Floating-Car-Data
This is a project from 8th semester software on Aalborg University. It revolves around using Floating Car Data to categorise Speed Limits within Denmark.

## Docker

This docker container was made so that the setup for the python environment for using FCD was easier

### Setup

To setup the python container
1. Copy `.env.example` and rename it `.env`
2. Insert your API key from https://fcd-share.civil.aau.dk/ into `FCD_API_KEY`
3. Run `docker compose up -d` in this directory

You can now use the terminal feature within docker desktop or use the command `docker compose exec -ti python python /scripts/api-script.py`

Any changes you make to `./scripts/api-script.py` will be reflected into the container instantly.


### Saving data to file

You can save data by using the append operator `>>` eg: 

```
docker compose exec -ti python python /scripts/api-script.py >> data.json
```

# Old readme
## geo_json_metrics
script in open_street folder is for reading the open street map files, extracting road and their information.
script in geo_json_metrics is for requesting from the https://fcd-share.civil.aau.dk/ api and saving the response as json. 

## Be aware
The osm_id column contains None / null values. This may correspond to update fallouts.

## Memory issues 
The 3 dataframes are very large. For model training, perhaps create a random forest on each dataset, and combine them:

example:
```
from sklearn.ensemble import VotingClassifier

est_AB = AdaBoostClassifier()
score_AB=est_AB.fit(X_train,y_train).score(X_test,y_test)

est_RF = RandomForestClassifier()
score_RF=est_RF.fit(X_train,y_train).score(X_test,y_test)

est_Ensemble = VotingClassifier(estimators=[('AB', est_AB), ('RF', est_RF)],
                        voting='soft',
                        weights=[1, 1])

score_Ensemble=est_Ensemble.fit(X_train,y_train).score(X_test,y_test)
```