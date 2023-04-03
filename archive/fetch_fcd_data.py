"""Script used to fetch all data from 2012-2014 from the FCD api.
Will create a json file for each request. 
"""

import requests
import json


def save_json(json_data: dict, id: int) -> None:
    """Save the json dictionary to a file.

    Args:
        json_data (dict): the json data as a dict
        id (int): appended to filename for unique filename
    """
    json_str = json.dumps(json_data, indent=4, sort_keys=True)
    with open(f"data/2014_{id}.json", "w+") as f:
        f.write(json_str)


def make_request(url: str) -> dict: 
    """Make request for a FCD api url. Return the json response as dict.

    Args:
        url (str): url for FCD api

    Returns:
        dict: the JSON response as a dict
    """
    res = requests.get(url)
    my_json = res.content.decode("utf8").replace("'", '"')
    return json.loads(my_json) # return json response as dictionary


def run_requests(start_url: str) -> None:
    """Given an initial FCD url, it
    will keep requesting the url found in the "next" property
    found in the json response. Will do so, until next is None ie.
    no more data for the url query.

    Args:
        start_url (str): intial FCD url. ex. all data from 2012.
    """
    iteration = 1
    next_url = start_url

    while next_url != None:
        json_dict = make_request(next_url)
        next_url = json_dict["next"]
        save_json(json_dict, iteration)
        iteration += 1


def main():
    url = f"https://fcd-share.civil.aau.dk/api/linestrings/?year=2014&format=json&apikey={API_KEY}"
    run_requests(url)


if __name__ == "__main__":
    main()
