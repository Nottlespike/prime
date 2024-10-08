import hashlib
import time
from typing import Any
import torch
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed._tensor.api import DTensor

from zeroband.utils.logging import get_logger


__all__ = ["get_sharding_strategy", "get_peak_flops", "get_num_flop_per_token", "get_num_params"]


def get_sharding_strategy(sharding_strategy: str) -> ShardingStrategy:
    if sharding_strategy == "FULL_SHARD":
        return ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "SHARD_GRAD_OP":
        return ShardingStrategy.SHARD_GRAD_OP
    elif sharding_strategy == "NO_SHARD":
        return ShardingStrategy.NO_SHARD
    elif sharding_strategy == "HYBRID_SHARD":
        return ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "_HYBRID_SHARD_ZERO2":
        return ShardingStrategy._HYBRID_SHARD_ZERO2
    else:
        raise ValueError(
            f"Invalid sharding_strategy: {sharding_strategy}. Please choose 'FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD', 'HYBRID_SHARD', or '_HYBRID_SHARD_ZERO2'."
        )


### code above inspired and copied from https://github.com/pytorch/torchtitan/blob/4b3f2e41a084bf79a8540068ed525539d1244edd/torchtitan/utils.py#L119


# hardcoded BF16 type peak flops for NVIDIA A100 and H100 GPU
def get_peak_flops(device_name: str) -> int:
    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for H100 SXM and other variants
            return 989e12
    else:  # for other GPU types, assume A100
        return 312e12


def get_num_flop_per_token(num_params: int, model_config, seq_len) -> int:
    l, h, q, t = (  # noqa: E741
        model_config.n_layers,
        model_config.n_heads,
        model_config.dim // model_config.n_heads,
        seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * l * h * q * t

    return flop_per_token


def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= model.tok_embeddings.weight.numel()
    return num_params


class PerfCounter:
    """A class to count tokens per second with a rolling window.
    we use a rollowing window because time perf counter is not precise enough in some case
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.tokens = []
        self.times = []

    def count_tokens(self, tokens: int):
        self.tokens.append(tokens)
        self.times.append(time.perf_counter())
        if len(self.tokens) > self.window_size:
            self.tokens.pop(0)
            self.times.pop(0)

    def get_tokens_per_second(self) -> float | None:
        if len(self.tokens) < 2:
            return None
        return sum(self.tokens[1:]) / (self.times[-1] - self.times[0])


TENSOR_SIG_SAMPLE_SIZE = 100


def get_tensor_signature(a: torch.Tensor | torch.nn.Parameter) -> str:
    """
    Get the tensor signature
    """
    while isinstance(a, torch.nn.Parameter):
        a = a.data

    if isinstance(a, DTensor):
        a = a.full_tensor()

    if a.numel() < TENSOR_SIG_SAMPLE_SIZE:
        b = a.as_strided(size=(a.numel(),), stride=(1,))
    else:
        step_size = a.numel() // TENSOR_SIG_SAMPLE_SIZE
        b = a.as_strided(size=(TENSOR_SIG_SAMPLE_SIZE,), stride=(step_size,))
    element_str = "".join([f"{x:.3e}" for x in b])
    element_hash = hashlib.md5(element_str.encode("utf-8")).hexdigest()
    return f"{a.dtype}{a.shape}{a.stride()}<{element_hash}>"


def get_module_signature(module: torch.nn.Module, compress: bool = True) -> str:
    """
    Get the module signature
    """
    state_dict_sig = {name: get_tensor_signature(param) for name, param in module.named_parameters()}
    if compress:
        return hashlib.md5(str(state_dict_sig).encode("utf-8")).hexdigest()
    else:
        return "\n".join(f"{name}: {sig}" for name, sig in state_dict_sig.items())


def get_optimizer_signature(optimizer: torch.optim.Optimizer, compress: bool = True) -> str:
    """
    Get the optimizer signature
    """

    def unwrap_tensor(state_dict: dict) -> dict:
        new_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, dict):
                new_dict[key] = unwrap_tensor(value)
            elif isinstance(value, torch.Tensor):
                new_dict[key] = get_tensor_signature(value)
            else:
                new_dict[key] = str(value)
        return new_dict

    state_dict_sig = unwrap_tensor(optimizer.state_dict())

    if compress:
        return hashlib.md5(str(state_dict_sig).encode("utf-8")).hexdigest()
    else:
        return "\n".join(f"{name}: {sig}" for name, sig in state_dict_sig.items())


def get_tensor_list_signature(tensor_list: list[torch.Tensor]) -> str:
    return hashlib.md5(str(tensor_list).encode("utf-8")).hexdigest()


class GPUMemoryMonitor:
    # inspired from https://github.com/pytorch/torchtitan/blob/eef8bb2b1b6f0875ab0581079e1511d51654910e/torchtitan/metrics.py#L32
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)  # device object
        self.device_capacity = torch.cuda.get_device_properties(self.device).total_memory
        self.device_capacity_gib = self._to_gib(self.device_capacity)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        self._logger = get_logger()

    def _to_gib(self, memory_in_bytes):
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib

    def _to_pct(self, memory):
        return 100 * memory / self.device_capacity

    def get_peak_stats(self) -> dict[str, Any]:
        cuda_info = torch.cuda.memory_stats(self.device)

        max_active = cuda_info["active_bytes.all.peak"]
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = cuda_info["reserved_bytes.all.peak"]
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        return {
            "gpu_max_active_gib": max_active_gib,
            "gpu_max_active_pct": max_active_pct,
            "gpu_max_reserved_gib": max_reserved_gib,
            "gpu_max_reserved_pct": max_reserved_pct,
        }

    def reset_peak_stats(self):
        torch.cuda.reset_peak_memory_stats()

    def format_peak_states(self, peak_stats: dict[str, Any] | None = None) -> str:
        if peak_stats is None:
            peak_stats = self.get_peak_stats()
        return f"Active {peak_stats['gpu_max_active_gib']:.2f} GiB ({peak_stats['gpu_max_active_pct']:.2f}%), Reserved {peak_stats['gpu_max_reserved_gib']:.2f} GiB ({peak_stats['gpu_max_reserved_pct']:.2f}%)"
