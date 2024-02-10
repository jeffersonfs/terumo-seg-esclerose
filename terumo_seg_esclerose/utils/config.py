from mmengine.config import Config



class ConfigLoaded():
    _config = None
    def load_config(self, config_path):
        global _config
        _config = Config.fromfile(config_path)
        return _config

    def get_config(self):
        global _config
        return _config
