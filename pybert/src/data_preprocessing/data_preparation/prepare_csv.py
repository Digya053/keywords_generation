import json

import numpy as np
import pandas as pd

from pathlib import Path
from argparse import ArgumentParser

BASE_DATA_DIR = Path('pybert/data')

def json_parser(text_file):
    """Extracts the required keys from json and stores it in dictionary.

    Parameters
    ----------
    text_file : str
        The textfile which has to be parsed.

    Returns
    -------
    list of dict:
        a list of dictinary of extracted keys to be used as columns in dataframe
    """

    df_dict = []
    with open(BASE_DATA_DIR/text_file) as json_file:
        data = json.load(json_file)
        for d in data['daacs']:
            for m in d['metadata']:
                for s in m['umm']['ScienceKeywords']:
                    dict = {}
                    dict['daac_name'] = d['daac_name']
                    dict['title'] = m['umm']['EntryTitle']
                    dict['abstract'] = m['umm']['Abstract']
                    dict['l_sciencekeywords'] = len(m['umm']['ScienceKeywords'])
                    dict['term'] = s['Term']

                    if 'VariableLevel1' in s:
                        dict['vl_1'] = s['VariableLevel1']

                        if 'VariableLevel2' in s:
                            dict['vl_2'] = s['VariableLevel2']

                            if 'VariableLevel3' in s:
                                dict['vl_3'] = s['VariableLevel3']
                                dict['most_depth'] = s['VariableLevel3']

                            else:
                                dict['vl_3'] = np.nan
                                dict['most_depth'] = s['VariableLevel2']
                        else:
                            dict['vl_2'] = np.nan
                            dict['most_depth'] = s['VariableLevel1']

                    else:
                        dict['vl_1'] = np.nan
                        dict['most_depth'] = s['Term']


                    df_dict.append(dict)
    return df_dict

def convert_to_csv(df_dict):
    """Converts the dictionary of items into dataframe and saves to csv file.

    Parameters
    ----------
    df_dict : list of dict
        A list of dictinary of extracted keys to be used as columns in dataframe
    """
    df = pd.DataFrame(df_dict)
    column_order = ['daac_name', 'title', 'abstract', 'l_sciencekeywords', 'term', 'vl_1', 'vl_2', 'vl_3', 'most_depth']
    df[column_order].to_csv(BASE_DATA_DIR/'data_tabular.csv', index=False)
    print("Data saved to data/data_tabular.csv")

def main():
    parser = ArgumentParser()
    parser.add_argument("--text_file", default='data.txt', type=str)
    args = parser.parse_args()

    df_dict = json_parser(args.text_file)
    convert_to_csv(df_dict)

if __name__ == "__main__":
    main()