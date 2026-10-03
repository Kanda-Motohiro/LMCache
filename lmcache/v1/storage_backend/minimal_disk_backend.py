# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
import os

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.path_sharder import PathSharder

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


class MinimalDiskBackend(StorageBackendInterface):
    """Minimal synchronous disk storage backend without async workers,
    locks, or cache eviction.

    - All get and put operations are performed synchronously.
    - No threading locks (disk.lock is omitted).
    - put writes unconditionally to disk.
    - get directly inspects the filesystem path.
    - Eviction and capacity tracking are omitted.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: Optional[Any] = None,
        local_cpu_backend: Optional["LocalCPUBackend"] = None,
        dst_device: str = torch_device_type,
        lmcache_worker: Optional[Any] = None,
        metadata: Optional[LMCacheMetadata] = None,
    ) -> None:
        """Initialize the MinimalDiskBackend.

        Args:
            config: LMCache engine configuration.
            loop: Event loop (unused, kept for interface compatibility).
            local_cpu_backend: Local CPU backend for allocating MemoryObjs on get.
            dst_device: Destination device (e.g. 'cuda' or 'cpu').
            lmcache_worker: Cache controller worker (unused).
            metadata: LMCache engine metadata for default shapes and dtypes.
        """
        if torch_dev.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        self.dst_device = dst_device
        self.local_cpu_backend = local_cpu_backend
        self.metadata = metadata
        self.dict: dict[CacheEngineKey, DiskCacheMetadata] = {}

        if config.local_disk is not None:
            sharder = PathSharder(
                raw_csv=config.local_disk,
                strategy=config.local_disk_path_sharding,
                dst_device=dst_device,
                create_dirs=True,
            )
            self.path: str = sharder.selected
        else:
            self.path = "/tmp/lmcache"
            os.makedirs(self.path, exist_ok=True)

        logger.info(
            "Minimal disk cache path: %s (device %s)",
            self.path,
            dst_device,
        )

    def __str__(self) -> str:
        """Return the string representation of this backend."""
        return "MinimalDiskBackend"

    def _key_to_path(self, key: CacheEngineKey) -> str:
        """Convert a CacheEngineKey to a filesystem path on disk.

        Args:
            key: The cache engine key.

        Returns:
            The absolute file path on disk.
        """
        return os.path.join(self.path, key.to_string().replace("/", "-") + ".pt")

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check whether the cache chunk exists on disk by checking its file path.

        Args:
            key: The cache engine key to check.
            pin: Unused parameter for interface compatibility.

        Returns:
            True if the file exists on disk, False otherwise.
        """
        path = self._key_to_path(key)
        return os.path.exists(path)

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Check whether consecutive keys exist on disk (prefix matching).

        Args:
            keys: List of cache engine keys.
            pin: Unused parameter for interface compatibility.

        Returns:
            Number of consecutive prefix hits found on disk.
        """
        hit_chunks = 0
        for key in keys:
            if not self.contains(key, pin=pin):
                break
            hit_chunks += 1
        return hit_chunks

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Async interface for batched contains check.

        Args:
            lookup_id: Lookup identifier (unused).
            keys: List of cache engine keys.
            pin: Unused parameter for interface compatibility.

        Returns:
            Number of consecutive prefix hits found on disk.
        """
        return self.batched_contains(keys, pin=pin)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """Check whether key is in ongoing put tasks.

        Since puts are synchronous, this always returns False.

        Args:
            key: The cache engine key.

        Returns:
            Always False.
        """
        return False

    def put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """Synchronously write a KV chunk to disk unconditionally.

        Args:
            key: The cache engine key.
            memory_obj: The memory object containing the KV data.
            on_complete_callback: Optional callback invoked after writing completes.
        """
        assert memory_obj.tensor is not None, "memory_obj.tensor must not be None"
        path = self._key_to_path(key)
        buffer = memory_obj.byte_array

        # Unconditionally write to path
        with open(path, "wb") as f:
            f.write(buffer)

        # Store metadata for subsequent get operations
        self.dict[key] = DiskCacheMetadata(
            path=path,
            size=memory_obj.get_physical_size(),
            shape=memory_obj.metadata.shape,
            dtype=memory_obj.metadata.dtype,
            cached_positions=memory_obj.metadata.cached_positions,
            fmt=memory_obj.metadata.fmt,
            pin_count=0,
        )

        if on_complete_callback is not None:
            try:
                on_complete_callback(key)
            except Exception as e:
                logger.warning("on_complete_callback failed for key %s: %s", key, e)

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """Synchronous put task (implements StorageBackendInterface).

        Args:
            key: The cache engine key.
            memory_obj: The memory object containing the KV data.
            on_complete_callback: Optional callback invoked after write completes.
        """
        self.put(key, memory_obj, on_complete_callback=on_complete_callback)

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """Synchronously write multiple KV chunks to disk.

        Args:
            keys: Cache keys for the KV chunks.
            memory_objs: Memory objects containing the KV data.
            transfer_spec: Optional transfer specification (unused).
            on_complete_callback: Optional callback invoked once per key.
        """
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.put(key, memory_obj, on_complete_callback=on_complete_callback)

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """Synchronously load a cached KV chunk from disk by inspecting the path.

        Args:
            key: The cache engine key to load.

        Returns:
            A MemoryObj containing the loaded KV data, or None if the file
            does not exist on disk or loading fails.
        """
        path = self._key_to_path(key)
        if not os.path.exists(path):
            self.dict.pop(key, None)
            return None

        # Resolve metadata
        if key in self.dict:
            meta = self.dict[key]
            shape = meta.shape
            dtype = meta.dtype
            fmt = meta.fmt
            cached_positions = meta.cached_positions
        elif self.metadata is not None:
            shape = self.metadata.kv_shape
            dtype = self.metadata.kv_dtype
            fmt = MemoryFormat.KV_2LTD
            cached_positions = None
        else:
            logger.warning(
                "Cannot load key %s from disk: metadata is not available.", key
            )
            return None

        if self.local_cpu_backend is None:
            logger.error("local_cpu_backend is required to allocate memory for get.")
            return None

        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        if memory_obj is None:
            logger.error("Memory allocation failed during disk load for key %s", key)
            return None

        try:
            with open(path, "rb") as f:
                f.readinto(memory_obj.byte_array)
        except Exception as e:
            logger.warning("Failed to read file %s: %s", path, e)
            memory_obj.ref_count_down()
            return None

        memory_obj.metadata.cached_positions = cached_positions
        return memory_obj

    def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Synchronously load a cached KV chunk from disk.

        Args:
            key: The cache engine key to load.

        Returns:
            MemoryObj if found and loaded, None otherwise.
        """
        return self.get_blocking(key)

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """Synchronously load multiple KV chunks from disk.

        Args:
            keys: List of cache engine keys.

        Returns:
            List of loaded MemoryObjs (or None for misses).
        """
        return [self.get_blocking(k) for k in keys]

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """Non-blocking batched get interface, executed synchronously.

        Args:
            lookup_id: Lookup identifier (unused).
            keys: List of cache engine keys.
            transfer_spec: Optional transfer specification (unused).

        Returns:
            List of successfully loaded MemoryObjs.
        """
        results: list[MemoryObj] = []
        for key in keys:
            obj = self.get_blocking(key)
            if obj is not None:
                results.append(obj)
        return results

    def pin(self, key: CacheEngineKey) -> bool:
        """Pin key (no-op since eviction is disabled).

        Args:
            key: Cache key.

        Returns:
            True if key exists on disk, False otherwise.
        """
        return self.contains(key)

    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin key (no-op since eviction is disabled).

        Args:
            key: Cache key.

        Returns:
            Always True.
        """
        return True

    def touch_cache(self) -> None:
        """Touch cache (no-op since eviction is disabled)."""
        pass

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """Remove a cached chunk from disk.

        Args:
            key: Cache key to remove.
            force: Unused parameter for interface compatibility.

        Returns:
            True if the file existed and was removed, False otherwise.
        """
        self.dict.pop(key, None)
        path = self._key_to_path(key)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def batched_remove(
        self,
        keys: list[CacheEngineKey],
        force: bool = True,
    ) -> int:
        """Remove multiple cached chunks from disk.

        Args:
            keys: List of cache keys to remove.
            force: Unused parameter for interface compatibility.

        Returns:
            Number of successfully removed files.
        """
        return sum(self.remove(key, force=force) for key in keys)

    def get_allocator_backend(self) -> Optional["LocalCPUBackend"]:
        """Get the memory allocator backend.

        Returns:
            The associated LocalCPUBackend instance.
        """
        return self.local_cpu_backend

    def close(self) -> None:
        """Close the backend."""
        pass
