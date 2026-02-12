# rapidindex/cache/cache_manager.py
"""
Multi-tier caching system for RapidIndex.

Implements a sophisticated caching strategy with:
- L1: In-memory LRU cache (fastest, volatile)
- L2: Disk cache (persistent, local)
- L3: Redis cache (optional, distributed)
"""

from typing import Optional, Any, Dict, List, Union
from abc import ABC, abstractmethod
from enum import Enum
import pickle
import hashlib
import time
import os
from pathlib import Path
from datetime import datetime, timedelta

from cachetools import TTLCache, LRUCache
from diskcache import Cache as DiskCache
from loguru import logger
from pydantic import BaseModel, Field

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CacheLayer(str, Enum):
    """Cache layer types."""
    MEMORY = "memory"
    DISK = "disk"
    REDIS = "redis"


class CacheConfig(BaseModel):
    """Configuration for cache manager."""
    
    # Memory cache (L1)
    memory_enabled: bool = Field(default=True)
    memory_maxsize: int = Field(default=1000, ge=10)
    memory_ttl: int = Field(default=300, ge=1)  # 5 minutes
    
    # Disk cache (L2)
    disk_enabled: bool = Field(default=True)
    disk_dir: str = Field(default=".cache/disk")
    disk_size_limit: int = Field(default=1024 * 1024 * 1024)  # 1GB
    
    # Redis cache (L3, optional)
    redis_enabled: bool = Field(default=False)
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_ttl: int = Field(default=3600, ge=1)  # 1 hour
    
    # General settings
    default_ttl: int = Field(default=3600, ge=1)
    enable_compression: bool = Field(default=False)
    key_prefix: str = Field(default="rapidindex")
    
    # Eviction policy
    eviction_policy: str = Field(default="lru")  # lru, lfu, ttl


class CacheStats(BaseModel):
    """Cache statistics."""
    
    total_gets: int = 0
    total_sets: int = 0
    
    memory_hits: int = 0
    disk_hits: int = 0
    redis_hits: int = 0
    misses: int = 0
    
    memory_size: int = 0
    disk_size: int = 0
    
    uptime_seconds: float = 0
    
    def hit_rate(self) -> float:
        """Calculate overall hit rate."""
        total_attempts = self.total_gets
        if total_attempts == 0:
            return 0.0
        hits = self.memory_hits + self.disk_hits + self.redis_hits
        return hits / total_attempts
    
    def memory_hit_rate(self) -> float:
        """Calculate memory cache hit rate."""
        if self.total_gets == 0:
            return 0.0
        return self.memory_hits / self.total_gets
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_gets': self.total_gets,
            'total_sets': self.total_sets,
            'memory_hits': self.memory_hits,
            'disk_hits': self.disk_hits,
            'redis_hits': self.redis_hits,
            'misses': self.misses,
            'hit_rate': round(self.hit_rate(), 3),
            'memory_hit_rate': round(self.memory_hit_rate(), 3),
            'memory_size': self.memory_size,
            'disk_size': self.disk_size,
            'uptime_seconds': round(self.uptime_seconds, 2)
        }


class BaseCacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached data."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get cache size (number of items)."""
        pass


class MemoryCacheBackend(BaseCacheBackend):
    """In-memory LRU cache backend (L1)."""
    
    def __init__(self, maxsize: int = 1000, ttl: int = 300):
        """
        Initialize memory cache.
        
        Args:
            maxsize: Maximum number of items
            ttl: Time-to-live in seconds
        """
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
        logger.info(
            "Memory cache initialized",
            maxsize=maxsize,
            ttl=ttl
        )
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from memory cache."""
        return self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in memory cache."""
        try:
            self.cache[key] = value
            return True
        except Exception as e:
            logger.warning(f"Memory cache set failed: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete from memory cache."""
        try:
            del self.cache[key]
            return True
        except KeyError:
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory."""
        return key in self.cache
    
    async def clear(self) -> None:
        """Clear memory cache."""
        self.cache.clear()
        logger.info("Memory cache cleared")
    
    def size(self) -> int:
        """Get number of items in memory cache."""
        return len(self.cache)


class DiskCacheBackend(BaseCacheBackend):
    """Disk-based cache backend (L2)."""
    
    def __init__(self, directory: str = ".cache/disk", size_limit: int = 1024**3):
        """
        Initialize disk cache.
        
        Args:
            directory: Cache directory path
            size_limit: Maximum cache size in bytes
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        
        self.cache = DiskCache(
            str(self.directory),
            size_limit=size_limit
        )
        
        logger.info(
            "Disk cache initialized",
            directory=directory,
            size_limit_mb=size_limit // (1024 * 1024)
        )
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from disk cache."""
        try:
            value = self.cache.get(key)
            return value
        except Exception as e:
            logger.warning(f"Disk cache get failed: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in disk cache."""
        try:
            self.cache.set(key, value, expire=ttl)
            return True
        except Exception as e:
            logger.warning(f"Disk cache set failed: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete from disk cache."""
        try:
            return self.cache.delete(key)
        except Exception as e:
            logger.warning(f"Disk cache delete failed: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists on disk."""
        return key in self.cache
    
    async def clear(self) -> None:
        """Clear disk cache."""
        try:
            self.cache.clear()
            logger.info("Disk cache cleared")
        except Exception as e:
            logger.error(f"Disk cache clear failed: {e}")
    
    def size(self) -> int:
        """Get number of items in disk cache."""
        return len(self.cache)


class RedisCacheBackend(BaseCacheBackend):
    """Redis cache backend (L3, optional)."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis not installed. Install with: pip install redis"
            )
        
        self.redis_url = redis_url
        self.redis_client = None
        self._connected = False
        
        logger.info("Redis cache backend created", url=redis_url)
    
    async def _ensure_connected(self):
        """Ensure Redis connection is established."""
        if not self._connected:
            try:
                self.redis_client = redis.from_url(
                    self.redis_url,
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test connection
                await self.redis_client.ping()
                self._connected = True
                logger.info("Connected to Redis")
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from Redis."""
        try:
            await self._ensure_connected()
            value = await self.redis_client.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in Redis."""
        try:
            await self._ensure_connected()
            serialized = pickle.dumps(value)
            
            if ttl:
                await self.redis_client.setex(key, ttl, serialized)
            else:
                await self.redis_client.set(key, serialized)
            
            return True
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete from Redis."""
        try:
            await self._ensure_connected()
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.warning(f"Redis delete failed: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            await self._ensure_connected()
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            logger.warning(f"Redis exists check failed: {e}")
            return False
    
    async def clear(self) -> None:
        """Clear all keys (use with caution!)."""
        try:
            await self._ensure_connected()
            await self.redis_client.flushdb()
            logger.warning("Redis database flushed")
        except Exception as e:
            logger.error(f"Redis clear failed: {e}")
    
    def size(self) -> int:
        """Get number of keys in Redis (sync operation)."""
        # This is a rough estimate - Redis doesn't provide exact count easily
        return 0
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self._connected = False
            logger.info("Redis connection closed")


class CacheManager:
    """
    Multi-tier cache manager.
    
    Provides intelligent caching across multiple layers:
    - L1: Memory (fastest, volatile)
    - L2: Disk (persistent, slower)
    - L3: Redis (distributed, optional)
    
    Example:
        >>> config = CacheConfig(memory_maxsize=1000, disk_enabled=True)
        >>> cache = CacheManager(config)
        >>> await cache.set("key", "value", ttl=3600)
        >>> value = await cache.get("key")
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize cache manager.
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self.stats = CacheStats()
        self._start_time = time.time()
        
        # Initialize backends
        self.backends: Dict[CacheLayer, BaseCacheBackend] = {}
        
        # L1: Memory
        if self.config.memory_enabled:
            self.backends[CacheLayer.MEMORY] = MemoryCacheBackend(
                maxsize=self.config.memory_maxsize,
                ttl=self.config.memory_ttl
            )
        
        # L2: Disk
        if self.config.disk_enabled:
            self.backends[CacheLayer.DISK] = DiskCacheBackend(
                directory=self.config.disk_dir,
                size_limit=self.config.disk_size_limit
            )
        
        # L3: Redis (optional)
        if self.config.redis_enabled and REDIS_AVAILABLE:
            try:
                self.backends[CacheLayer.REDIS] = RedisCacheBackend(
                    redis_url=self.config.redis_url
                )
            except Exception as e:
                logger.warning(f"Redis backend failed to initialize: {e}")
        
        logger.info(
            "CacheManager initialized",
            layers=list(self.backends.keys()),
            memory_enabled=self.config.memory_enabled,
            disk_enabled=self.config.disk_enabled,
            redis_enabled=CacheLayer.REDIS in self.backends
        )
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache with L1 → L2 → L3 fallback.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        self.stats.total_gets += 1
        prefixed_key = self._make_key(key)
        
        # Try L1: Memory
        if CacheLayer.MEMORY in self.backends:
            value = await self.backends[CacheLayer.MEMORY].get(prefixed_key)
            if value is not None:
                self.stats.memory_hits += 1
                logger.debug("Cache hit", layer="memory", key=key)
                return value
        
        # Try L2: Disk
        if CacheLayer.DISK in self.backends:
            value = await self.backends[CacheLayer.DISK].get(prefixed_key)
            if value is not None:
                self.stats.disk_hits += 1
                logger.debug("Cache hit", layer="disk", key=key)
                
                # Promote to L1
                if CacheLayer.MEMORY in self.backends:
                    await self.backends[CacheLayer.MEMORY].set(
                        prefixed_key,
                        value
                    )
                
                return value
        
        # Try L3: Redis
        if CacheLayer.REDIS in self.backends:
            value = await self.backends[CacheLayer.REDIS].get(prefixed_key)
            if value is not None:
                self.stats.redis_hits += 1
                logger.debug("Cache hit", layer="redis", key=key)
                
                # Promote to L1 and L2
                await self._promote_to_upper_layers(prefixed_key, value)
                
                return value
        
        # Cache miss
        self.stats.misses += 1
        logger.debug("Cache miss", key=key)
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        layers: Optional[List[CacheLayer]] = None
    ) -> bool:
        """
        Set value in cache across all layers.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = use default)
            layers: Specific layers to cache in (None = all enabled)
            
        Returns:
            True if set in at least one layer
        """
        self.stats.total_sets += 1
        prefixed_key = self._make_key(key)
        ttl = ttl or self.config.default_ttl
        
        # Determine target layers
        target_layers = layers or list(self.backends.keys())
        
        success = False
        
        # Set in all target layers
        for layer in target_layers:
            if layer in self.backends:
                result = await self.backends[layer].set(prefixed_key, value, ttl)
                if result:
                    success = True
                    logger.debug(
                        "Cache set",
                        layer=layer.value,
                        key=key,
                        ttl=ttl
                    )
        
        return success
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from all cache layers.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if deleted from at least one layer
        """
        prefixed_key = self._make_key(key)
        success = False
        
        for layer, backend in self.backends.items():
            result = await backend.delete(prefixed_key)
            if result:
                success = True
                logger.debug("Cache delete", layer=layer.value, key=key)
        
        return success
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in any cache layer.
        
        Args:
            key: Cache key
            
        Returns:
            True if exists in any layer
        """
        prefixed_key = self._make_key(key)
        
        for backend in self.backends.values():
            if await backend.exists(prefixed_key):
                return True
        
        return False
    
    async def clear(self, layers: Optional[List[CacheLayer]] = None) -> None:
        """
        Clear cache layers.
        
        Args:
            layers: Specific layers to clear (None = all)
        """
        target_layers = layers or list(self.backends.keys())
        
        for layer in target_layers:
            if layer in self.backends:
                await self.backends[layer].clear()
                logger.info(f"Cleared {layer.value} cache")
        
        # Reset stats
        self.stats = CacheStats()
        self._start_time = time.time()
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern.
        
        Args:
            pattern: Key pattern (supports * wildcard)
            
        Returns:
            Number of keys deleted
        """
        # This is a simplified implementation
        # For production, you'd want to implement proper pattern matching
        deleted = 0
        
        # For now, just log a warning
        logger.warning(
            "Pattern-based clearing not fully implemented",
            pattern=pattern
        )
        
        return deleted
    
    async def warm_cache(self, keys_values: Dict[str, Any], ttl: Optional[int] = None):
        """
        Warm cache with multiple key-value pairs.
        
        Args:
            keys_values: Dictionary of keys and values
            ttl: Time-to-live for all entries
        """
        logger.info(f"Warming cache with {len(keys_values)} entries")
        
        for key, value in keys_values.items():
            await self.set(key, value, ttl=ttl)
        
        logger.success(f"Cache warmed with {len(keys_values)} entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        self.stats.uptime_seconds = time.time() - self._start_time
        
        # Update sizes
        if CacheLayer.MEMORY in self.backends:
            self.stats.memory_size = self.backends[CacheLayer.MEMORY].size()
        if CacheLayer.DISK in self.backends:
            self.stats.disk_size = self.backends[CacheLayer.DISK].size()
        
        return self.stats.to_dict()
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = CacheStats()
        self._start_time = time.time()
        logger.info("Cache statistics reset")
    
    async def _promote_to_upper_layers(self, key: str, value: Any) -> None:
        """Promote cached value to upper (faster) layers."""
        if CacheLayer.MEMORY in self.backends:
            await self.backends[CacheLayer.MEMORY].set(key, value)
        
        if CacheLayer.DISK in self.backends:
            await self.backends[CacheLayer.DISK].set(key, value)
    
    def _make_key(self, key: str) -> str:
        """
        Create prefixed cache key.
        
        Args:
            key: Original key
            
        Returns:
            Prefixed key
        """
        return f"{self.config.key_prefix}:{key}"
    
    async def close(self):
        """Close all cache backends."""
        for layer, backend in self.backends.items():
            if isinstance(backend, RedisCacheBackend):
                await backend.close()
        
        logger.info("CacheManager closed")
    
    def __repr__(self) -> str:
        """String representation."""
        layers = ", ".join(layer.value for layer in self.backends.keys())
        return f"CacheManager(layers=[{layers}])"


# Convenience function for simple use
async def create_cache_manager(
    memory_size: int = 1000,
    disk_enabled: bool = True,
    redis_enabled: bool = False,
    redis_url: str = "redis://localhost:6379/0"
) -> CacheManager:
    """
    Create a cache manager with simple configuration.
    
    Args:
        memory_size: Memory cache size
        disk_enabled: Enable disk cache
        redis_enabled: Enable Redis cache
        redis_url: Redis connection URL
        
    Returns:
        Configured CacheManager instance
    """
    config = CacheConfig(
        memory_maxsize=memory_size,
        disk_enabled=disk_enabled,
        redis_enabled=redis_enabled,
        redis_url=redis_url
    )
    
    return CacheManager(config)


# Export
__all__ = [
    'CacheManager',
    'CacheConfig',
    'CacheLayer',
    'CacheStats',
    'create_cache_manager'
]