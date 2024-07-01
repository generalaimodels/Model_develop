import sys
from pathlib import Path
current_file = Path(__file__).resolve()
package_root = current_file.parents[3]
sys.path.append(str(package_root))
# print(f"Running {current_file.name} with PYTHONPATH set to {package_root}")

from timm.data.readers.reader_factory import create_reader
from timm.data.readers.img_extensions import *
from timm.data.readers.class_map import *
from timm.data.readers.img_extensions import *
from timm.data.readers.reader import *
from timm.data.readers.reader_hfds import *
from timm.data.readers.reader_factory import *
from timm.data.readers.reader_image_folder import *
# from timm.data.readers.reader_image_in_tar import *
from timm.data.readers.reader_wds   import *
# from timm.data.readers.reader_tfds import *
from timm.data.readers.reader_image_tar import *
from timm.data.readers.shared_count import *
