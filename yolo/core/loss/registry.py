class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class LossRegistry(metaclass=Singleton):
    def __init__(self):
        self.plugins = {}

    def register(self, cls):
        self.plugins[cls.__name__] = cls
        return cls

    def __iter__(self):
        return iter(self.plugins.values())
    
    def __getitem__(self, key):
        return self.plugins[key]
