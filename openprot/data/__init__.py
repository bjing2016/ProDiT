from ..utils.misc_utils import import_subclasses
from .data import OpenProtDataset

globals().update(import_subclasses(__name__, __path__, OpenProtDataset))
