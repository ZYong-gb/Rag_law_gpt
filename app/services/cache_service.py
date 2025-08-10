"""
Redis缓存服务
"""
import json
import time
import hashlib
from typing import Any, Optional, List, Dict
from loguru import logger
from app.database.connection import get_redis
from app.config import settings


class CacheService:
    """缓存服务"""
    
    def __init__(self):
        self.redis_client = None
        self.default_ttl = settings.redis_cache_ttl
    
    def _get_client(self):
        """获取Redis客户端"""
        if not self.redis_client:
            self.redis_client = get_redis()
        return self.redis_client
    
    async def get(self, key: str) -> Optional[str]:
        """获取缓存值"""
        try:
            client = self._get_client()
            value = client.get(key)
            return value
            
        except Exception as e:
            logger.error(f"缓存获取失败 {key}: {e}")
            return None
    
    async def set(self, 
                 key: str, 
                 value: str, 
                 expire: Optional[int] = None) -> bool:
        """设置缓存值"""
        try:
            client = self._get_client()
            ttl = expire or self.default_ttl
            
            result = client.setex(key, ttl, value)
            return result
            
        except Exception as e:
            logger.error(f"缓存设置失败 {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            client = self._get_client()
            result = client.delete(key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"缓存删除失败 {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        try:
            client = self._get_client()
            result = client.exists(key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"缓存存在性检查失败 {key}: {e}")
            return False
    
    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """获取JSON格式的缓存值"""
        try:
            value = await self.get(key)
            if value:
                return json.loads(value)
            return None
            
        except Exception as e:
            logger.error(f"JSON缓存获取失败 {key}: {e}")
            return None
    
    async def set_json(self, 
                      key: str, 
                      value: Dict[str, Any], 
                      expire: Optional[int] = None) -> bool:
        """设置JSON格式的缓存值"""
        try:
            json_value = json.dumps(value, ensure_ascii=False)
            return await self.set(key, json_value, expire)
            
        except Exception as e:
            logger.error(f"JSON缓存设置失败 {key}: {e}")
            return False
    
    def generate_cache_key(self, prefix: str, *args) -> str:
        """生成缓存键"""
        # 将参数转换为字符串并生成哈希
        content = ":".join(str(arg) for arg in args)
        hash_value = hashlib.md5(content.encode()).hexdigest()
        return f"{prefix}:{hash_value}"
    
    async def cache_query_result(self, 
                               question: str, 
                               answer: str, 
                               documents: List[Dict[str, Any]],
                               confidence: float) -> bool:
        """缓存查询结果"""
        try:
            cache_key = self.generate_cache_key("query", question.strip().lower())
            
            cache_data = {
                "question": question,
                "answer": answer,
                "documents": documents,
                "confidence": confidence,
                "cached_at": time.time()
            }
            
            return await self.set_json(cache_key, cache_data)
            
        except Exception as e:
            logger.error(f"查询结果缓存失败: {e}")
            return False
    
    async def get_cached_query_result(self, question: str) -> Optional[Dict[str, Any]]:
        """获取缓存的查询结果"""
        try:
            cache_key = self.generate_cache_key("query", question.strip().lower())
            return await self.get_json(cache_key)
            
        except Exception as e:
            logger.error(f"获取缓存查询结果失败: {e}")
            return None
    
    async def cache_document_embedding(self, 
                                     document_id: str, 
                                     embeddings: List[float]) -> bool:
        """缓存文档嵌入向量"""
        try:
            cache_key = f"embedding:{document_id}"
            return await self.set_json(cache_key, {"embeddings": embeddings})
            
        except Exception as e:
            logger.error(f"文档嵌入缓存失败: {e}")
            return False
    
    async def get_cached_document_embedding(self, document_id: str) -> Optional[List[float]]:
        """获取缓存的文档嵌入向量"""
        try:
            cache_key = f"embedding:{document_id}"
            data = await self.get_json(cache_key)
            return data.get("embeddings") if data else None
            
        except Exception as e:
            logger.error(f"获取缓存文档嵌入失败: {e}")
            return None
    
    async def increment_counter(self, key: str, expire: Optional[int] = None) -> int:
        """递增计数器"""
        try:
            client = self._get_client()
            
            # 使用管道操作保证原子性
            pipe = client.pipeline()
            pipe.incr(key)
            if expire:
                pipe.expire(key, expire)
            results = pipe.execute()
            
            return results[0]
            
        except Exception as e:
            logger.error(f"计数器递增失败 {key}: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            client = self._get_client()
            info = client.info()
            
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_commands_processed": info.get("total_commands_processed", 0)
            }
            
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {}
    
    async def clear_cache(self, pattern: Optional[str] = None) -> bool:
        """清空缓存"""
        try:
            client = self._get_client()
            
            if pattern:
                # 删除匹配模式的键
                keys = client.keys(pattern)
                if keys:
                    client.delete(*keys)
            else:
                # 清空所有缓存
                client.flushdb()
            
            logger.info(f"缓存清理完成，模式: {pattern or 'ALL'}")
            return True
            
        except Exception as e:
            logger.error(f"缓存清理失败: {e}")
            return False


# 全局缓存服务实例
cache_service = CacheService()
