import os
import importlib

current_dir = os.path.dirname(__file__)
for fname in os.listdir(current_dir):
    if fname.endswith(".py") and fname not in ["__init__.py", "registry.py"]:
        importlib.import_module(f"{__name__}.{fname[:-3]}")

from .registry import get_multifield_loss, register_multifield_loss
