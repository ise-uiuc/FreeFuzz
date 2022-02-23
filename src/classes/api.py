import inspect
from numpy.random import randint, choice
from classes.argument import ArgType, Argument, OracleType
from utils.probability import *



class API:
    def __init__(self, api_name):
        self.api = api_name

    def mutate(self):
        pass

    def to_code(self) -> str:
        pass

    def to_dict(self) -> dict:
        pass

    @staticmethod
    def generate_args_from_record(record: dict) -> dict:
        pass

    @staticmethod
    def indent_code(code):
        codes = code.split("\n")
        result = []
        for code in codes:
            if code == "":
                continue
            result.append("  " + code)
        return "\n".join(result) + "\n"

