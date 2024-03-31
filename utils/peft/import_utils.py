import importlib


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None
