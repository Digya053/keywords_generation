import math
import json

import requests
from pathlib import Path
from argparse import ArgumentParser

DAACS = ["GHRC_CLOUD", "OB_DAAC", "LAADS", "LPDAAC_ECS", "SEDAC", "GHRC", "ORNL_DAAC", "GES_DISC"]
URL = "https://cmr.earthdata.nasa.gov/search/collections.umm_json?page_size=2000&"

BASE_DATA_DIR = Path('pybert/data')

def download_daac_metadata(daac):
    """
    daac -> str: identifier for daac in CMR
    Returns a list of all the metadata that corresponds to the daac.
    Main elements we are interested in are: Title, Description, and Science Keywords
    Follow the following paths to get to these elements from each item in items:
    title: item['umm']['EntryTitle']
    science_keywords: item['umm']['ScienceKeywords']
    description: item['umm']['Abstract'] or item['umm']['Description']
    ScienceKeywords are our ground truth. They are hierarcical in nature.
    Category > Topic >Term > VariableLevel1 > VariableLevel2 > VariableLevel3
    """
    url = f"{URL}provider_id={daac}"
    response = requests.get(url)
    parsed_response = json.loads(response.text)
    all_responses = parsed_response['items']
    total_pages = math.ceil(parsed_response['hits'] / 2000)
    if total_pages > 1:
        for page_num in range(total_pages + 1):
            response = requests.get(f"{url}&page_num={page_num}")
            all_responses += json.loads(response.text)['items']
    return all_responses


def save_json_responses(text_file):
    """
    Prepare data in json format and save it to data/data.txt.
    """
    responses = {}
    responses['daacs'] = []
    for daac in DAACS:
        responses['daacs'].append({
            'daac_name': daac,
            'metadata': download_daac_metadata(daac)
        })

    with open(BASE_DATA_DIR/text_file, 'w+') as outfile:
        json.dump(responses, outfile)

    print("Data saved to data/data.txt")

def main():
    parser = ArgumentParser()
    parser.add_argument("--text_file", default='data.txt', type=str)
    args = parser.parse_args()

    save_json_responses(args.text_file)

if __name__ == "__main__":
    main()