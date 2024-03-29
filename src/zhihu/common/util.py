""" utility for zhihu project """
import json
import random
import string
import os
from time import strftime, localtime
from typing import Union

def print_json(data):
    """ print json data """
    if hasattr(data, "model_dump_json"):
        data = json.load(data.model_dump_json())
    if isinstance(data, list):
        for item in data:
            print_json(item)
    elif isinstance(data, dict):
        print(json.dumps(data, indent=4, ensure_ascii=False))
    else:
        print(data)

def random_str(length):
    """ generate random string with specified length """
    letters = string.ascii_letters
    result_str = ''.join(random.choice(letters) for _ in range(length))
    return result_str

def write_log_file(file_name, content:Union[str, list[str]]):
    """ write the content to a specified file_name """
    log_path = "logs/zhihu"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    time_stamp = strftime("%Y%m%d_%H%M%S", localtime())
    with open(f"{log_path}/{file_name}_{time_stamp}.log", "w", encoding="utf-8") as f:
        if isinstance(content, str):
            f.write(content)
        if isinstance(content, list):
            for c in content:
                f.write(c + "\n")
