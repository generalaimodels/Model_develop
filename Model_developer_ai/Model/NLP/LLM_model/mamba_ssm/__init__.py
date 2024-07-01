import sys
from pathlib import Path
current_file=Path(__file__).resolve()
parent_dir=current_file.parents[4]
sys.path.append(str(parent_dir))
print(parent_dir)
__version__ = "2.0.3"

from Model.NLP.LLM_model.mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from Model.NLP.LLM_model.mamba_ssm.modules.mamba_simple import Mamba
from Model.NLP.LLM_model.mamba_ssm.modules.mamba2 import Mamba2
from Model.NLP.LLM_model.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
