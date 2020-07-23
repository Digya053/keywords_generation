import pickle
import logging

logger = logging.getLogger()

def load_pickle(input_file):
    """ Loads input data through pickle

    :param input_file: file_name 
    :return: data from the file
    """
    
    with open(str(input_file), 'rb') as f:
        data = pickle.load(f)
    return data

def convert_dict_to_list(dict):
    """Convert the dictionary to list

    :param dict: dictionary format to be converted
    :return: converted list format
    """
    return [(k, v) for k, v in dict.items()]