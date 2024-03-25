import json
import random
import string

def print_json(data):
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
    letters = string.ascii_letters
    result_str = ''.join(random.choice(letters) for _ in range(length))
    return result_str