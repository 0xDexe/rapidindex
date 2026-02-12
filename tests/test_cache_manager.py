# tests/unit/test_cache_manager.py
"""
Unit tests for CacheManager.
"""

import pytest
import asyncio
from rapidindex.cache.cache_manager import (
    CacheManager,
    CacheConfig,
    CacheLayer,
    MemoryCacheBackend,
    DiskCacheBackend
)


class TestMemoryCacheBackend:
    """Test memory cache backend."""
    
    @pytest.fixture
    def cache(self):
        return MemoryCacheBackend(maxsize=10, ttl=60)
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        await cache.set("key1", "value1")
        value = await cache.get("key1")
        assert value == "value1"
    
    @pytest.mark.asyncio
    async def test_delete(self, cache):
        await cache.set("key1", "value1")
        assert await cache.exists("key1")
        
        await cache.delete("key1")
        assert not await cache.exists("key1")
    
    @pytest.mark.asyncio
    async def test_clear(self, cache):
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        assert cache.size() == 2
        
        await cache.clear()
        assert cache.size() == 0


class TestDiskCacheBackend:
    """Test disk cache backend."""
    
    @pytest.fixture
    def cache(self, tmp_path):
        return DiskCacheBackend(
            directory=str(tmp_path / "cache"),
            size_limit=10 * 1024 * 1024
        )
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        await cache.set("key1", {"data": "value1"})
        value = await cache.get("key1")
        assert value == {"data": "value1"}
    
    @pytest.mark.asyncio
    async def test_ttl(self, cache):
        await cache.set("key1", "value1", ttl=1)
        assert await cache.exists("key1")
        
        await asyncio.sleep(2)
        value = await cache.get("key1")
        assert value is None


class TestCacheManager:
    """Test CacheManager."""
    
    @pytest.fixture
    def cache(self):
        config = CacheConfig(
            memory_enabled=True,
            disk_enabled=True,
            redis_enabled=False
        )
        return CacheManager(config)
    
    @pytest.mark.asyncio
    async def test_multi_tier_get(self, cache):
        # Set in all layers
        await cache.set("key1", "value1")
        
        # Should hit memory
        value = await cache.get("key1")
        assert value == "value1"
        assert cache.stats.memory_hits == 1
        
        # Clear memory, should hit disk
        await cache.clear(layers=[CacheLayer.MEMORY])
        value = await cache.get("key1")
        assert value == "value1"
        assert cache.stats.disk_hits == 1
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, cache):
        value = await cache.get("nonexistent")
        assert value is None
        assert cache.stats.misses == 1
    
    @pytest.mark.asyncio
    async def test_stats(self, cache):
        # Perform operations
        await cache.set("key1", "value1")
        await cache.get("key1")
        await cache.get("missing")
        
        stats = cache.get_stats()
        assert stats['total_sets'] == 1
        assert stats['total_gets'] == 2
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
    
    @pytest.mark.asyncio
    async def test_warm_cache(self, cache):
        data = {f"key{i}": f"value{i}" for i in range(10)}
        await cache.warm_cache(data)
        
        stats = cache.get_stats()
        assert stats['memory_size'] == 10