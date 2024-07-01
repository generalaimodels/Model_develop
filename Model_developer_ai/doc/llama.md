```python
import torch
from Model.NLP.LLM_model.mistral_inference.model import Transformer, ModelArgs
args=ModelArgs(
    dim=1024,
    n_layers=12,
    head_dim=64,
    hidden_dim=4096,
    n_heads=16,
    n_kv_heads=16,
    norm_eps=1e-5,
    vocab_size=32128,
)
model=Transformer(
    args,
    pipeline_rank=0,
    num_pipeline_ranks=4,
)
print(model)
```