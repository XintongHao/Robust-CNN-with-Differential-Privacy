from models import pixeldp_cnn

def module_from_name(name):
    if name == 'pixeldp_cnn':
        return pixeldp_cnn

def name_from_module(module):
    return module.__name__.split('.')[-1]
