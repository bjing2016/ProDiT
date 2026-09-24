from ..utils.misc_utils import import_subclasses
from .eval import OpenProtEval

globals().update(import_subclasses(__name__, __path__, OpenProtEval))
