import os
import platform
import logging
from typing import Optional
import torch
import torch.distributed as dist
import logging
from typing import Dict, Optional, Union
import torch
import torch.cuda as cuda
from torch.types import Device
import logging
from typing import Optional, List, Dict, Any
import torch
import torch.cuda as cuda

class CUDAMemoryInfo:
    """
    A class to detect and provide detailed information about CUDA memory.
    
    Example
    if __name__ == "__main__":
    cuda_info = CUDAMemoryInfo()
    cuda_info.print_memory_summary()
    """

    def __init__(self, device: Optional[Union[Device, int]] = None):
        self.logger = self._setup_logger()
        self.device = self._get_device(device)
        self._lazy_init()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _get_device(self, device: Optional[Union[Device, int]]) -> int:
        if device is None:
            return cuda.current_device()
        return cuda._get_device_index(device)

    def _lazy_init(self) -> None:
        if not cuda.is_initialized():
            self.logger.info("Initializing CUDA")
            cuda.init()

    def get_memory_info(self) -> Dict[str, Union[int, float, str]]:
        """
        Get detailed information about CUDA memory.
        """
        try:
            total, free = cuda.mem_get_info(self.device)
            used = total - free
            
            memory_stats = cuda.memory_stats(self.device)
            
            info = {
                "device": self.device,
                "total_memory": self._format_bytes(total),
                "free_memory": self._format_bytes(free),
                "used_memory": self._format_bytes(used),
                "memory_utilization": f"{(used / total) * 100:.2f}%",
                "allocated_memory": self._format_bytes(memory_stats["allocated_bytes.all.current"]),
                "cached_memory": self._format_bytes(memory_stats["reserved_bytes.all.current"] - memory_stats["allocated_bytes.all.current"]),
                "num_allocations": memory_stats["allocation.all.current"],
                "num_segments": memory_stats["segment.all.current"],
                "max_memory_allocated": self._format_bytes(memory_stats["allocated_bytes.all.peak"]),
                "max_memory_cached": self._format_bytes(memory_stats["reserved_bytes.all.peak"])
            }
            
            self.logger.info(f"CUDA memory info retrieved for device {self.device}")
            return info
        except cuda.CUDAError as e:
            self.logger.error(f"Error retrieving CUDA memory info: {str(e)}")
            return {}

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if num_bytes < 1024:
                return f"{num_bytes:.2f} {unit}"
            num_bytes /= 1024
        return f"{num_bytes:.2f} PB"

    def print_memory_summary(self) -> None:
        """
        Print a summary of CUDA memory information.
        """
        info = self.get_memory_info()
        if not info:
            self.logger.warning("No CUDA memory information available")
            return

        self.logger.info("CUDA Memory Summary:")
        for key, value in info.items():
            self.logger.info(f"{key.replace('_', ' ').title()}: {value}")


class DistributedBackendSelector:
    """
    

    selector = DistributedBackendSelector(log_level=logging.DEBUG)
    selector.init_process_group()
    # Your distributed model code here
    selector.cleanup()
    """
    def __init__(self, log_level: int = logging.INFO):
        self.logger = self._setup_logger(log_level)
        self.os_name = platform.system()
        self.cuda_available = torch.cuda.is_available()
        self.backend = self._select_backend()

    def _setup_logger(self, log_level: int) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _select_backend(self) -> Optional[str]:
        if self.cuda_available:
            if self.os_name == "Linux":
                self.logger.info("CUDA available on Linux. Selecting NCCL backend.")
                return "nccl"
            else:
                self.logger.info("CUDA available, but not on Linux. Falling back to Gloo backend.")
                return "gloo"
        else:
            if self.os_name == "Linux":
                self.logger.info("CUDA not available. On Linux, selecting Gloo backend.")
                return "gloo"
            elif self.os_name == "Windows":
                self.logger.warning("CUDA not available on Windows. Distributed capabilities may be limited.")
                return "gloo"
            else:
                self.logger.error(f"Unsupported OS: {self.os_name}")
                return None

    def init_process_group(self, init_method: str = "env://", world_size: int = -1, rank: int = -1) -> None:
        if not dist.is_available():
            self.logger.error("torch.distributed is not available.")
            return

        if self.backend is None:
            self.logger.error("No suitable backend found. Cannot initialize process group.")
            return

        try:
            dist.init_process_group(
                backend=self.backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank
            )
            self.logger.info(f"Process group initialized with backend: {self.backend}")
        except Exception as e:
            self.logger.error(f"Failed to initialize process group: {str(e)}")

    def get_backend(self) -> Optional[str]:
        return self.backend

    def cleanup(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
            self.logger.info("Process group destroyed.")


    
    
    


class CUDAManager:
    """
    
    EXample:
    if __name__ == "__main__":
    cuda_manager = CUDAManager()
    cuda_manager.print_all_devices_info()

    if cuda_manager.is_cuda_available and cuda_manager.device_count > 0:
        cuda_manager.set_device(0)
        print(f"Memory Usage: {cuda_manager.get_memory_usage()}")
        print(f"Utilization: {cuda_manager.get_utilization()}%")
        print(f"Temperature: {cuda_manager.get_temperature()}°C")
        print(f"Power Draw: {cuda_manager.get_power_draw()} W")
        print(f"Clock Rate: {cuda_manager.get_clock_rate()} Hz")

        stream = cuda_manager.create_stream()
        cuda_manager.set_stream(stream)
        cuda_manager.synchronize()
    else:
        print("No CUDA devices available")
    """
    def __init__(self):
        self.logger = self._setup_logger()
        self.is_cuda_available = cuda.is_available()
        self.device_count = cuda.device_count() if self.is_cuda_available else 0
        self.current_device = None
        self.devices_info = self._get_devices_info()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _get_devices_info(self) -> List[Dict[str, Any]]:
        devices_info = []
        if self.is_cuda_available:
            for i in range(self.device_count):
                device_properties = cuda.get_device_properties(i)
                devices_info.append({
                    'index': i,
                    'name': cuda.get_device_name(i),
                    'capability': cuda.get_device_capability(i),
                    'total_memory': device_properties.total_memory,
                    'multi_processor_count': device_properties.multi_processor_count
                })
        return devices_info

    def set_device(self, device_index: int) -> None:
        if 0 <= device_index < self.device_count:
            cuda.set_device(device_index)
            self.current_device = device_index
            self.logger.info(f"Set current device to: {device_index}")
        else:
            self.logger.error(f"Invalid device index: {device_index}")

    def get_memory_usage(self, device_index: Optional[int] = None) -> Dict[str, int]:
        device = device_index if device_index is not None else self.current_device
        if device is not None and 0 <= device < self.device_count:
            return cuda.memory_usage(device)
        self.logger.error(f"Invalid device index: {device}")
        return {}

    def get_utilization(self, device_index: Optional[int] = None) -> int:
        device = device_index if device_index is not None else self.current_device
        if device is not None and 0 <= device < self.device_count:
            return cuda.utilization(device)
        self.logger.error(f"Invalid device index: {device}")
        return -1

    def get_temperature(self, device_index: Optional[int] = None) -> int:
        device = device_index if device_index is not None else self.current_device
        if device is not None and 0 <= device < self.device_count:
            return cuda.temperature(device)
        self.logger.error(f"Invalid device index: {device}")
        return -1

    def get_power_draw(self, device_index: Optional[int] = None) -> float:
        device = device_index if device_index is not None else self.current_device
        if device is not None and 0 <= device < self.device_count:
            return cuda.power_draw(device)
        self.logger.error(f"Invalid device index: {device}")
        return -1.0

    def get_clock_rate(self, device_index: Optional[int] = None) -> int:
        device = device_index if device_index is not None else self.current_device
        if device is not None and 0 <= device < self.device_count:
            return cuda.clock_rate(device)
        self.logger.error(f"Invalid device index: {device}")
        return -1

    def synchronize(self, device_index: Optional[int] = None) -> None:
        device = device_index if device_index is not None else self.current_device
        if device is not None and 0 <= device < self.device_count:
            cuda.synchronize(device)
            self.logger.info(f"Synchronized device: {device}")
        else:
            self.logger.error(f"Invalid device index: {device}")

    def create_stream(self) -> cuda.Stream:
        return cuda.Stream()

    def set_stream(self, stream: cuda.Stream) -> None:
        cuda.set_stream(stream)
        self.logger.info("Set new CUDA stream")

    def can_device_access_peer(self, device_index: int, peer_device_index: int) -> bool:
        return cuda.can_device_access_peer(device_index, peer_device_index)

    def get_device_info(self, device_index: Optional[int] = None) -> Dict[str, Any]:
        device = device_index if device_index is not None else self.current_device
        if device is not None and 0 <= device < self.device_count:
            return self.devices_info[device]
        self.logger.error(f"Invalid device index: {device}")
        return {}

    def print_device_info(self, device_index: Optional[int] = None) -> None:
        device_info = self.get_device_info(device_index)
        if device_info:
            self.logger.info(f"Device Information:")
            for key, value in device_info.items():
                self.logger.info(f"{key}: {value}")
        else:
            self.logger.warning("No device information available")

    def print_all_devices_info(self) -> None:
        if self.is_cuda_available:
            for device in self.devices_info:
                self.print_device_info(device['index'])
        else:
            self.logger.warning("CUDA is not available on this system")

