import os
import logging
from typing import Optional, Union
import torch
import torch.distributed as dist
from torch.backends import cudnn

class DistributedBackendDetector:
    def __init__(self):
        self.logger = self._setup_logger()
        self.backend = None
        self.device = None

    @staticmethod
    def _setup_logger():
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def detect_and_set_backend(self) -> None:
        """Detect OS, CPU/GPU availability and set the best backend."""
        self._detect_os()
        self._detect_device()
        self._set_backend()

    def _detect_os(self) -> None:
        os_name = os.name
        self.logger.info(f"Detected OS: {os_name}")

    def _detect_device(self) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.logger.info("CUDA is available. Using GPU.")
        else:
            self.device = torch.device("cpu")
            self.logger.info("CUDA is not available. Using CPU.")

    def _set_backend(self) -> None:
        if self.device.type == "cuda":
            if dist.is_nccl_available():
                self.backend = "nccl"
                self.logger.info("Using NCCL backend for GPU.")
            else:
                self.backend = "gloo"
                self.logger.warning("NCCL not available. Falling back to Gloo backend for GPU.")
        else:
            self.backend = "gloo"
            self.logger.info("Using Gloo backend for CPU.")

    def init_process_group(self, rank: int, world_size: int, 
                           init_method: Optional[str] = None) -> None:
        """Initialize the distributed process group."""
        try:
            dist.init_process_group(
                backend=self.backend,
                init_method=init_method,
                rank=rank,
                world_size=world_size
            )
            self.logger.info(f"Process group initialized with backend: {self.backend}")
        except Exception as e:
            self.logger.error(f"Failed to initialize process group: {str(e)}")
            raise

    def optimize_backend(self) -> None:
        """Optimize backend settings."""
        if self.device.type == "cuda":
            cudnn.benchmark = True
            self.logger.info("CUDNN benchmark enabled for improved performance.")

    def get_device(self) -> torch.device:
        """Return the detected device."""
        return self.device

    def get_backend(self) -> Union[str, None]:
        """Return the detected backend."""
        return self.backend
