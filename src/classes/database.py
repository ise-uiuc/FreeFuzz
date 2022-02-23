import pymongo
from numpy.random import choice
"""
This file is the interfere with database
"""

class Database:
    """Database setting"""
    signature_collection = "signature"
    similarity_collection = "similarity"
    argdef_collection = "api_args"

    def __init__(self) -> None:
        pass

    def database_config(self, host, port, database_name):
        self.DB = pymongo.MongoClient(host=host, port=port)[database_name]

    def index_name(self, api_name, arg_name):
        record = self.DB[self.signature_collection].find_one({"api": api_name})
        if record == None:
            print(f"No such {api_name}")
            return None
        arg_names = record["args"]
        for idx, name in enumerate(arg_names):
            if name == arg_name:
                return f"parameter:{idx}"
        return None

    def select_rand_over_db(self, api_name, arg_name):
        if api_name not in self.DB.list_collection_names():
            return None, False
        arg_names = self.DB[self.signature_collection].find_one({"api": api_name})["args"]
        if arg_name.startswith("parameter:"):
            index = int(arg_name[10:])
            if index >= len(arg_names):
                return None, False
            arg_name = arg_names[index]

        sim_dict = self.DB[self.similarity_collection].find_one({
            "api": api_name,
            "arg": arg_name
        })
        if sim_dict == None:
            return None, False
        APIs = sim_dict["APIs"]
        probs = sim_dict["probs"]
        if len(APIs) == 0:
            return None, False
        target_api = choice(APIs, p=probs)
        # compare the time of 2 operations
        idx_name = self.index_name(target_api, arg_name)
        if idx_name == None:
            return None, False
        select_data = self.DB[target_api].aggregate([{
            "$match": {
                "$or": [{
                    arg_name: {
                        "$exists": True
                    },
                }, {
                    idx_name: {
                        "$exists": True
                    }
                }]
            }
        }, {
            "$sample": {
                "size": 1
            }
        }])
        if not select_data.alive:
            # not found any value in the (target_api, arg_name)
            print(f"ERROR IN SIMILARITY: {target_api}, {api_name}")
            return None, False
        select_data = select_data.next()
        if arg_name in select_data.keys():
            return select_data[arg_name], True
        else:
            return select_data[idx_name], True


    def get_rand_record(self, api_name):
        record = self.DB[api_name].aggregate([{"$sample": {"size": 1}}])
        if not record.alive:
            print(f"NO SUCH API: {api_name}")
            assert(0)
        record = record.next()
        record.pop("_id")
        assert("_id" not in record.keys())
        return record
    
    def get_all_records(self, api_name):
        if api_name not in self.DB.list_collection_names():
            print(f"NO SUCH API: {api_name}")
            return []
        temp = self.DB[api_name].find({}, {"_id": 0})
        records = []
        for t in temp:
            assert("_id" not in t.keys())
            records.append(t)
        return records
    
    def get_signature(self, api_name):
        record = self.DB[self.signature_collection].find_one({"api": api_name}, {"_id": 0})
        if record == None:
            print(f"NO SIGNATURE FOR: {api_name}")
            assert(0)
        return record["args"]

    @staticmethod
    def get_api_list(DB, start_str):
        api_list = []
        for name in DB.list_collection_names():
            if name.startswith(start_str):
                api_list.append(name)
        return api_list

class TorchDB(Database):
    def __init__(self) -> None:
        super().__init__()

    def get_api_list(self):
        self.api_list = super().get_api_list(self.DB, "torch.")
        return self.api_list

class TFDB(Database):
    def __init__(self) -> None:
        super().__init__()

    def get_api_list(self):
        self.api_list = super().get_api_list(self.DB, "tf.")
        return self.api_list
"""
Database for each library
NOTE:
You must config the database by using `database_config(host, port, name)` before use!!!
Like TFDatabase.database_config("127.0.0.1", 27109, "tftest")
"""
TorchDatabase = TorchDB()
TFDatabase = TFDB()