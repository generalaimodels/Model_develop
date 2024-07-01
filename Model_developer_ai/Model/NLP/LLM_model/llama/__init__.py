# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
import sys
from pathlib import Path
current_file=Path(__file__).resolve()
package_root = current_file.parents[1]
sys.path.append(str(package_root))
print(package_root)
from llama.generation import Llama
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Dialog, Tokenizer
