import json

class util:
    def __init__(self):
        pass

    def load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def save_json(self, data, path):
        with open(path, 'w') as f:
            json.dump(data, f)