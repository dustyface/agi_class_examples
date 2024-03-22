import json

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