import json

class Config:
    def __init__(self, data, train, model):
        self.data = data
        self.train = train
        self.model = model

    @classmethod
    def from_json(cls, cfg):
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data, params.train, params.model)

class HelperObject:
    def __init__(self, dict_):
        self.__dict__.update(dict_)
