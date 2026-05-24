import pandas as pd
import numpy
import regex as re
import json
import random
import os
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm

def jsonl_to_list(path_to_json):
    datas = []
    with open(path_to_json,"r") as file:
        for line in file:
            data = json.loads(line.strip())
            datas.append(data)
    return datas

def read_jsonl_basic(filepath):
    """Read JSONL file line by line"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                record = json.loads(line)
                records.append(record)
    return records
