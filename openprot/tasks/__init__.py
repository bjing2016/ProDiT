from ..utils.misc_utils import import_subclasses
from .task import OpenProtTask

globals().update(import_subclasses(__name__, __path__, OpenProtTask))
