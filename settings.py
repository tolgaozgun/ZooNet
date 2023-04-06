import configparser

config_path = "./config"

class Settings():
    def __init__(self):
        self.run_prefix = "run"
        self.output_dir = "/out"
        self.imagenet_path = "/data"
        self.config = configparser.ConfigParser()
    
    def load_from_file(self):
        self.config.read(config_path)
        # Load from config
        # Set the values of the class
        self.run_prefix = self.config.get("directories", "run_prefix")
        self.output_dir = self.config.get("directories", "output_dir")
        self.imagenet_path = self.config.get("directories", "imagenet_path")
