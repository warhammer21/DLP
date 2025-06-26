import json

class Config:
    def __init__(self, data, train, model):
        self.data = data
        self.train = train
        self.model = model

    @classmethod
    def from_json(cls, path):  # still supports loading from file if needed
        with open(path, 'r') as f:
            raw_dict = json.load(f)
        return cls.from_dict(raw_dict)

    @classmethod
    def from_dict(cls, raw_dict):  # ✅ new method for dict input
        params = json.loads(json.dumps(raw_dict), object_hook=HelperObject)
        return cls(params.data, params.train, params.model)

class HelperObject:
    def __init__(self, dict_):
        self.__dict__.update(dict_)
