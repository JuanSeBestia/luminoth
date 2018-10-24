class Config(object):
    DEBUG = False
    TESTING = False
    SECRET_KEY=b'JuanSeBestia'

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False


class DebugConfig(Config):
    DEBUG = True
    TESTING = False
