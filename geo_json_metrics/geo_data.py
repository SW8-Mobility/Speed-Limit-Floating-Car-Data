import requests 
import json 

def save_data(data: dict, id: int) -> None:
    json_str = json.dumps(data, indent=4, sort_keys=True)
    with open(f'data/2014_{id}.json', 'w+') as f:
        f.write(json_str)

def make_request(url: str) -> dict: # json as dict 
    res = requests.get(url)
    my_json = res.content.decode('utf8').replace("'", '"')
    data = json.loads(my_json)
    return data # return json response as dictionary 

def run_requests(start_url: str) -> None:
    iteration = 1
    next_url = start_url

    while (next_url != None):
        data = make_request(next_url)
        next_url = data['next']
        save_data(data, iteration)
        iteration += 1

def main():
    # url = f'https://fcd-share.civil.aau.dk/api/linestrings/?year=2014&osm_id=10240935&apikey={api_key}'
    url = f'https://fcd-share.civil.aau.dk/api/linestrings/?year=2014&format=json&apikey={API_KEY}'
    run_requests(url)


if __name__ == '__main__':
    main()