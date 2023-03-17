import requests 
import json 
import os

api_key = os.environ.get("FCD_API_KEY")
url = f'https://fcd-share.civil.aau.dk/api/points/?year=2014&month=11&format=json&apikey={api_key}'

res = requests.get(url)
my_json = res.content.decode('utf8').replace("'", '"')
data = json.loads(my_json)
s = json.dumps(data, indent=4, sort_keys=True)

print(s)