import json


def save_json(file_path, mappings):
    with open(file_path, 'w+') as f:
        json.dump(mappings, f)
        print(f'clusters saved in: {file_path}')
