# SPDX-License-Identifier: Apache-2.0
# Standard
import shutil
import tempfile

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.paged_cpu_gpu_memory_allocator import (
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.minimal_disk_backend import MinimalDiskBackend


def _create_test_key(chunk_hash: int = 123456) -> CacheEngineKey:
    """Create a sample CacheEngineKey for testing.

    Args:
        chunk_hash: Chunk hash integer.

    Returns:
        A CacheEngineKey instance.
    """
    return CacheEngineKey(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        chunk_hash=chunk_hash,
        dtype=torch.bfloat16,
    )


def _setup_backend(tmp_dir: str):
    """Helper to initialize LocalCPUBackend and MinimalDiskBackend."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_disk=tmp_dir,
        max_local_cpu_size=0.5,
    )
    shape = torch.Size([28, 2, 256, 8, 128])
    dtype = torch.bfloat16

    allocator = PagedCpuGpuMemoryAllocator()
    allocator.init_cpu_memory_allocator(
        size=29360128 * 4, # align bytes 29360128 * 4
        shapes=[shape],
        dtypes=[dtype],
        fmt=MemoryFormat.KV_2LTD,
    )
    local_cpu = LocalCPUBackend(config, memory_allocator=allocator)
    backend = MinimalDiskBackend(config=config, local_cpu_backend=local_cpu)
    return backend, local_cpu, shape, dtype


def test_minimal_disk_backend_put_get():
    """Test put and get with random KV cache data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        backend, local_cpu, shape, dtype = _setup_backend(tmp_dir)

        key = _create_test_key(chunk_hash=101)
        mem_obj = local_cpu.allocate(shape, dtype, MemoryFormat.KV_2LTD)
        assert mem_obj is not None

        # Fill with random tensor data
        random_tensor = torch.randn(shape, dtype=dtype)
        assert mem_obj.tensor is not None
        mem_obj.tensor.copy_(random_tensor)

        # 1. Verify it does not exist initially
        assert not backend.contains(key)
        assert backend.get(key) is None

        # 2. Put unconditionally to disk
        backend.put(key, mem_obj)
        assert backend.contains(key)

        # 3. Synchronously read from disk path
        loaded_obj = backend.get(key)
        assert loaded_obj is not None
        assert loaded_obj.tensor is not None
        assert torch.equal(mem_obj.tensor, loaded_obj.tensor)

        # 4. Remove from disk
        assert backend.remove(key)
        assert not backend.contains(key)
        assert backend.get(key) is None

        local_cpu.get_memory_allocator().close()


def test_minimal_disk_backend_batched():
    """Test batched put and batched get."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        backend, local_cpu, shape, dtype = _setup_backend(tmp_dir)

        keys = [_create_test_key(chunk_hash=i) for i in range(200, 203)]
        mem_objs = []
        for _ in keys:
            obj = local_cpu.allocate(shape, dtype, MemoryFormat.KV_2LTD)
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.copy_(torch.randn(shape, dtype=dtype))
            mem_objs.append(obj)

        # Batched put
        backend.batched_submit_put_task(keys, mem_objs)
        assert backend.batched_contains(keys) == len(keys)

        # Batched get
        loaded_objs = backend.batched_get_blocking(keys)
        assert len(loaded_objs) == len(keys)
        for orig, loaded in zip(mem_objs, loaded_objs, strict=True):
            assert loaded is not None
            assert orig.tensor is not None
            assert loaded.tensor is not None
            assert torch.equal(orig.tensor, loaded.tensor)

        # Batched remove
        assert backend.batched_remove(keys) == len(keys)
        assert backend.batched_contains(keys) == 0

        local_cpu.get_memory_allocator().close()


def run_demo() -> None:
    """Standalone runner for quick verification without pytest."""
    tmp_dir = tempfile.mkdtemp(prefix="lmcache_minimal_test_")
    print(f"=== Running MinimalDiskBackend Verification in {tmp_dir} ===")
    try:
        backend, local_cpu, shape, dtype = _setup_backend(tmp_dir)
        key = _create_test_key(chunk_hash=999)

        print("1. Allocating memory and generating random KV cache...")
        mem_obj = local_cpu.allocate(shape, dtype, MemoryFormat.KV_2LTD)
        assert mem_obj is not None
        random_tensor = torch.randn(shape, dtype=dtype)
        assert mem_obj.tensor is not None
        mem_obj.tensor.copy_(random_tensor)

        print(f"2. PUT key {key.chunk_hash} to disk...")
        backend.put(key, mem_obj)
        print(f"   contains={backend.contains(key)}")

        print("3. GET key from disk (inspecting file path)...")
        loaded_obj = backend.get(key)
        assert loaded_obj is not None
        assert loaded_obj.tensor is not None

        print("4. Verifying tensor equality...")
        assert torch.equal(mem_obj.tensor, loaded_obj.tensor)
        print("   SUCCESS: Loaded KV cache matches original random tensor!")

        local_cpu.get_memory_allocator().close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("Cleanup completed.")


if __name__ == "__main__":
    run_demo()
