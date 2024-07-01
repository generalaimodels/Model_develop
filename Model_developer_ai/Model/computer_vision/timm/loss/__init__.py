import sys
from pathlib import Path
current_file = Path(__file__).resolve()
package_root = current_file.parents[2]
sys.path.append(str(package_root))
print(f"Running {current_file.name} with PYTHONPATH set to {package_root}")
from timm.loss.asymmetric_loss import AsymmetricLossMultiLabel, AsymmetricLossSingleLabel
from timm.loss.binary_cross_entropy import BinaryCrossEntropy
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.loss.jsd import JsdCrossEntropy
