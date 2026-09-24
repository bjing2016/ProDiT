from ..utils.misc_utils import import_subclasses
from .track import OpenProtTrack

globals().update(import_subclasses(__name__, __path__, OpenProtTrack))
