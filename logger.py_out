# logger.py

# controls verbosity of printing throughout the codebase
# for printing use:
#   from logger import logger
#   logger.print("message", level="i")  # info level
#   logger.pprint(obj, level="d")        # debug level
# both print and pprint only print if the level is enabled

import pprint

class VerboseLogger:
    def __init__(self):
        # Empty set means no verbosity
        self.levels = set()

    def set(self, verbose_string: str):
        """Set verbosity from a string like 'i', 'd', or 'id'."""
        if verbose_string in ("none", None, ""):
            self.levels = set()
        else:
            self.levels = set(verbose_string)

    def print(self, *args, level="i", **kwargs):
        """Print only if this level is enabled."""
        if level in self.levels:
            print(*args, **kwargs)

    def pprint(self, obj, level="i"):
        """
        Pretty-print a Python object only if the verbosity level allows it.
        Uses pprint.pformat() so that formatting happens before printing.
        """
        if level in self.levels:
            print(pprint.pformat(obj))


# This is the shared logger imported everywhere
logger = VerboseLogger()
