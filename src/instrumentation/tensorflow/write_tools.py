import pymongo

"""
You should configure the database
"""
tf_db = pymongo.MongoClient(host="localhost", port=27017)["freefuzz-tf"]

def write_fn(func_name, params, input_signature, output_signature):
    params = dict(params)
    out_fname = "tf." + func_name
    if input_signature != None:
        params['input_signature'] = input_signature
    params['output_signature'] = output_signature
    tf_db[out_fname].insert_one(params)
