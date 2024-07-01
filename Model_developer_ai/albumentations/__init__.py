import os

from albumentations.check_version import check_for_updates

from albumentations._version import __version__  # noqa: F401
from albumentations.augmentations import *
from albumentations.core.composition import *
from albumentations.core.serialization import *
from albumentations.core.transforms_interface import *

# Perform the version check after all other initializations
if os.getenv("NO_ALBUMENTATIONS_UPDATE", "").lower() not in {"true", "1"}:
    check_for_updates()
