import pymongo
import textdistance
import re
import numpy as np
import configparser
import sys
from os.path import join

signature_collection = "signature"
similarity_collection = "similarity"

"""
Similarity Part:
This part relys on the database so I put it into this file
"""

API_def = {}
API_args = {}


def string_similar(s1, s2):
    return textdistance.levenshtein.normalized_similarity(s1, s2)


def loadAPIs(api_file='../data/torch_APIdef.txt'):
    global API_def, API_args
    with open(api_file, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.strip()
            API_name = line.split("(")[0]
            API_args_match = re.search("\((.*)\)", line)
            try:
                API_args_text = API_args_match.group(1)
            except:
                # with open("log/tf/api_def_error.txt", 'a') as f:
                #     f.write(line + "\n")
                # continue
                raise ValueError(line)
            # print(API_args_text)
            if API_name not in API_def.keys():
                API_def[API_name] = line
                API_args[API_name] = API_args_text


def query_argname(arg_name):
    '''
    Return a list of APIs with the exact argname
    '''
    def index_name(api_name, arg_name):
        arg_names = DB[signature_collection].find_one({"api": api_name})["args"]
        for idx, name in enumerate(arg_names):
            if name == arg_name:
                return f"parameter:{idx}"
        return None

    APIs = []
    for api_name in API_args.keys():
        # search from the database
        # if arg_name exists in the records of api_name, append api_name into APIs
        if api_name not in DB.list_collection_names(
        ) or arg_name not in API_args[api_name]:
            continue
        temp = DB[api_name].find_one({arg_name: {"$exists": True}})
        if temp == None:
            # since there are two forms of names for one argument, {arg_name} and parameter:{idx}
            # we need to check the parameter:{idx}
            idx_name = index_name(api_name, arg_name)
            if idx_name and DB[api_name].find_one({idx_name: {"$exists": True}}):
                APIs.append(api_name)
        else:
            APIs.append(api_name)
    return APIs


def mean_norm(x):
    return (x - np.mean(x)) / (np.max(x) - np.min(x))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def similarAPI(API, argname):
    '''
    Return a list of similar APIs (with the same argname) and their similarities
    '''
    API_with_same_argname = query_argname(argname)
    if len(API_with_same_argname) == 0:
        return [], []
    probs = []
    original_def = API_def[API]
    for item in API_with_same_argname:
        to_compare = API_def[item]
        probs.append(string_similar(original_def, to_compare))
    prob_norm2 = softmax(probs)
    return API_with_same_argname, prob_norm2




"""
Writing Parts (Data Preprocessing):
This part is to write API signature, argument value space and similarity
to database.

You SHOULD call these functions in this order!
    1. write_API_signature
    2. write_similarity
"""

def write_API_signature(library_name='torch'):
    """
    API's signature will be stored in 'signature' collection with the form
        api: the name of api
        args: the list of arguments' names
    """
    names = DB.list_collection_names()
    for api_name in names:
        if not api_name.startswith(library_name):
            continue
        if api_name not in API_args.keys():
            DB[signature_collection].insert_one({"api": api_name, "args": []})
            continue

        arg_names = []
        for temp_name in API_args[api_name].split(","):
            temp_name = temp_name.strip()
            if len(temp_name) == 0 or temp_name == "*":
                continue
            if "=" in temp_name:
                temp_name = temp_name[:temp_name.find("=")]
            arg_names.append(temp_name)
        DB[signature_collection].insert_one({
            "api": api_name,
            "args": arg_names
        })


def write_similarity(library_name='torch'):
    """
    Write the similarity of (api, arg) in 'similarity' with the form:
        api: the name of api
        arg: the name of arg
        APIs: the list of similar APIs
        probs: the probability list
    """
    names = DB.list_collection_names()
    for api_name in names:
        if not api_name.startswith(library_name):
            continue

        print(api_name)
        arg_names = DB["signature"].find_one({"api": api_name})["args"]
        for arg_name in arg_names:
            APIs, probs = similarAPI(api_name, arg_name)
            sim_dict = {}
            sim_dict["api"] = api_name
            sim_dict["arg"] = arg_name
            sim_dict["APIs"] = APIs
            sim_dict["probs"] = list(probs)
            DB[similarity_collection].insert_one(sim_dict)


if __name__ == "__main__":
    target = sys.argv[1]
    if target not in ["torch", "tf"]:
        print("Only support 'torch' or 'tf'!")
        assert(0)

    """
    Database Settings
    """
    config_name = f"demo_{target}.conf"
    freefuzz_cfg = configparser.ConfigParser()
    freefuzz_cfg.read(join("config", config_name))

    # database configuration
    mongo_cfg = freefuzz_cfg["mongodb"]
    host = mongo_cfg["host"]
    port = int(mongo_cfg["port"])

    DB = pymongo.MongoClient(host, port)[mongo_cfg[f"{target}_database"]]

    loadAPIs(join("..", "data", f"{target}_APIdef.txt"))
    write_API_signature(target)
    write_similarity(target)
