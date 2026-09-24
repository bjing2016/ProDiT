import importlib
import contextlib
import numpy as np
import pkgutil
import inspect


def import_subclasses(package, path, cls):
    """Import all submodules of a module, recursively

    :param package_name: Package name
    :type package_name: str
    :rtype: dict[types.ModuleType]
    """
    result = {}
    for _, name, _ in pkgutil.walk_packages(path):

        module = importlib.import_module(package + "." + name)
        for key, var in module.__dict__.items():
            if inspect.isclass(var) and issubclass(var, cls):
                result[key] = var
    return result


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def autoimport(name):
    module, name_ = name.rsplit(".", 1)
    return getattr(importlib.import_module(module), name_)
