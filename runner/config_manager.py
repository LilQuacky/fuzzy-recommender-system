import json

class ConfigManager:
    """
    ConfigManager handles loading and saving configuration files in JSON format for experiments.

    Methods:
        - load(): Loads and returns the configuration from the specified JSON file.
        - save(config, path=None): Saves the given configuration dictionary to a JSON file. If no path is provided, saves to the original config path.
    """
    def __init__(self, config_path):
        self.config_path = config_path

    def load(self):
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def save(self, config, path=None):
        path = path or self.config_path
        with open(path, 'w') as f:
            json.dump(config, f, indent=4) 
