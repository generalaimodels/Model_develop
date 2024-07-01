import sys
from pathlib import Path
current_file = Path(__file__).resolve()
package_root = current_file.parents[2]
sys.path.append(str(package_root))
# print(f"Running {current_file.name} with PYTHONPATH set to {package_root}")
from timm.data.auto_augment import RandAugment, AutoAugment, rand_augment_ops, auto_augment_policy,\
       rand_augment_transform, auto_augment_transform
from timm.data.config import resolve_data_config, resolve_model_data_config
from timm.data.constants import *
from timm.data.dataset import ImageDataset, IterableImageDataset, AugMixDataset
from timm.data.dataset_factory import create_dataset
from timm.data.dataset_info import DatasetInfo, CustomDatasetInfo
from timm.data.imagenet_info import ImageNetInfo, infer_imagenet_subset
from timm.data.loader import create_loader
from timm.data.mixup import Mixup, FastCollateMixup
from timm.data.readers import create_reader
from timm.data.readers import get_img_extensions, is_img_extension, set_img_extensions, add_img_extensions, del_img_extensions
from timm.data.real_labels import RealLabelsImagenet
from timm.data.transforms import *
from timm.data.transforms_factory import create_transform
