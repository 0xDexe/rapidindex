from diskcache import Cache
import hashlib
import json
from typing import Optional, Dict, Any
import os

class ReasoningCache:
    """Cache for LLM reasoning results"""
    
    def __init__(self, cache_dir: str = '.cache/reasoning'):
        """Initialize disk-based cache"""
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = Cache(cache_dir)
        self.hits = 0
        self.misses = 0
    
    def get(self, query: str, section_ids: list) -> Optional[Dict]:
        """Get cached reasoning result"""
        key = self._make_key(query, section_ids)
        result = self.cache.get(key)
        
        if isinstance(result, dict):
            self.hits += 1
            return result
        else:
            self.misses += 1
            return None
    
    def set(self, query: str, section_ids: list, result: Dict, ttl: int = 86400):
        """Cache reasoning result (default 24h TTL)"""
        key = self._make_key(query, section_ids)
        self.cache.set(key, result, expire=ttl)
    
    def _make_key(self, query: str, section_ids: list) -> str:
        """Generate cache key from query and sections"""
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Sort section IDs for consistency
        sorted_ids = sorted(section_ids)
        
        # Create hash
        content = f"{normalized_query}:{','.join(sorted_ids)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def clear(self):
        """Clear all cached results"""
        self.cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache) # type: ignore
        }