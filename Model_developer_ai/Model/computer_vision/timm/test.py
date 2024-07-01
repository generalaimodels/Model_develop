import sys
from pathlib import Path
current_file = Path(__file__).resolve()
package_root = current_file.parents[1]
sys.path.append(str(package_root))
print(f"Running {current_file.name} with PYTHONPATH set to {package_root}")
from timm.loss import BinaryCrossEntropy
import torch
loss=BinaryCrossEntropy()

output=loss(torch.rand(100),torch.rand(100))
print(output)