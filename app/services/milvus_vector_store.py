"""
基于Milvus的向量存储服务
与FAISS版本的对比实现
"""
import os
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from uuid import UUID
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from loguru import logger

# Milvus Lite import (conditional)
try:
    from milvus import default_server
    MILVUS_LITE_AVAILABLE = True
except ImportError:
    logger.warning("Milvus Lite不可用")
    MILVUS_LITE_AVAILABLE = False
    default_server = None

from app.config import settings
from app.services.embedding_service import embedding_service


class MilvusVectorStore:
    """Milvus向量存储"""
    
    def __init__(self):
        self.collection_name = "law_documents"
        self.collection = None
        self.dimension = settings.embedding_dimension
        self._initialize_milvus()
    
    def _initialize_milvus(self):
        """初始化Milvus连接和集合"""
        try:
            # 连接到Milvus
            connections.connect(
                alias="default",
                host=os.getenv("MILVUS_HOST", "localhost"),
                port=os.getenv("MILVUS_PORT", "19530")
            )
            
            # 检查集合是否存在
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logger.info(f"连接到现有Milvus集合: {self.collection_name}")
            else:
                self._create_collection()
            
            # 加载集合到内存
            self.collection.load()
            
        except Exception as e:
            logger.error(f"Milvus初始化失败: {e}")
            # 如果Milvus不可用，使用Milvus Lite（嵌入式版本）
            self._initialize_milvus_lite()
    
    def _initialize_milvus_lite(self):
        """初始化Milvus Lite（嵌入式版本）"""
        try:
            if not MILVUS_LITE_AVAILABLE:
                raise ImportError("Milvus Lite不可用")

            # 启动Milvus Lite服务器
            default_server.start()
            
            # 连接到Milvus Lite
            connections.connect(
                alias="default",
                host="127.0.0.1",
                port=default_server.listen_port
            )
            
            # 创建或连接集合
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
            else:
                self._create_collection()
            
            self.collection.load()
            
            logger.info("Milvus Lite初始化成功")
            
        except Exception as e:
            logger.error(f"Milvus Lite初始化失败: {e}")
            raise
    
    def _create_collection(self):
        """创建Milvus集合"""
        try:
            # 获取实际的嵌入维度
            self.dimension = embedding_service.get_embedding_dimension()
            
            # 定义字段模式
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="chunk_index", dtype=DataType.INT64)
            ]
            
            # 创建集合模式
            schema = CollectionSchema(
                fields=fields,
                description="法律文档向量集合"
            )
            
            # 创建集合
            self.collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            
            # 创建索引
            index_params = {
                "metric_type": "IP",  # 内积相似度
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info(f"创建Milvus集合成功: {self.collection_name}, 维度: {self.dimension}")
            
        except Exception as e:
            logger.error(f"创建Milvus集合失败: {e}")
            raise
    
    def add_documents(self, 
                     embeddings: np.ndarray,
                     document_ids: List[UUID],
                     metadata: List[Dict[str, Any]]):
        """
        添加文档到Milvus
        
        Args:
            embeddings: 文档嵌入向量
            document_ids: 文档ID列表
            metadata: 文档元数据列表
        """
        try:
            # 准备插入数据
            entities = [
                [str(doc_id) for doc_id in document_ids],  # document_id
                [meta.get('chunk_id', '') for meta in metadata],  # chunk_id
                embeddings.tolist(),  # embedding
                [meta.get('content', '')[:65535] for meta in metadata],  # content (截断到最大长度)
                [meta.get('title', '')[:500] for meta in metadata],  # title
                [meta.get('category', '') or '' for meta in metadata],  # category
                [meta.get('source', '') or '' for meta in metadata],  # source
                [meta.get('chunk_index', 0) for meta in metadata]  # chunk_index
            ]
            
            # 插入数据
            insert_result = self.collection.insert(entities)
            
            # 刷新集合以确保数据持久化
            self.collection.flush()
            
            logger.info(f"向Milvus添加了 {len(document_ids)} 个文档向量")
            
        except Exception as e:
            logger.error(f"向Milvus添加文档失败: {e}")
            raise
    
    def search(self, 
               query_embedding: np.ndarray,
               top_k: int = 5,
               score_threshold: float = 0.0) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        在Milvus中搜索相似文档
        
        Args:
            query_embedding: 查询向量
            top_k: 返回的文档数量
            score_threshold: 相似度阈值
            
        Returns:
            (document_id, score, metadata) 的列表
        """
        try:
            # 确保查询向量格式正确
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # 搜索参数
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10}
            }
            
            # 执行搜索
            search_results = self.collection.search(
                data=query_embedding.tolist(),
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["document_id", "chunk_id", "content", "title", "category", "source", "chunk_index"]
            )
            
            results = []
            for hits in search_results:
                for hit in hits:
                    if hit.score >= score_threshold:
                        metadata = {
                            'chunk_id': hit.entity.get('chunk_id'),
                            'document_id': hit.entity.get('document_id'),
                            'title': hit.entity.get('title'),
                            'content': hit.entity.get('content'),
                            'category': hit.entity.get('category'),
                            'source': hit.entity.get('source'),
                            'chunk_index': hit.entity.get('chunk_index')
                        }
                        
                        results.append((
                            hit.entity.get('chunk_id'),
                            float(hit.score),
                            metadata
                        ))
            
            logger.info(f"Milvus检索到 {len(results)} 个相关文档")
            return results
            
        except Exception as e:
            logger.error(f"Milvus搜索失败: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取Milvus统计信息"""
        try:
            if self.collection:
                # 获取集合统计信息
                stats = self.collection.get_stats()
                
                return {
                    "total_vectors": self.collection.num_entities,
                    "dimension": self.dimension,
                    "index_type": "IVF_FLAT",
                    "metric_type": "IP",
                    "collection_name": self.collection_name,
                    "framework": "Milvus"
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"获取Milvus统计失败: {e}")
            return {}
    
    def clear_collection(self):
        """清空集合"""
        try:
            if self.collection:
                # 删除集合中的所有数据
                self.collection.delete(expr="chunk_index >= 0")
                self.collection.flush()
                
                logger.info("Milvus集合已清空")
                
        except Exception as e:
            logger.error(f"清空Milvus集合失败: {e}")
            raise
    
    def create_index(self):
        """创建或重建索引"""
        try:
            if self.collection:
                # 释放集合
                self.collection.release()
                
                # 删除现有索引
                self.collection.drop_index()
                
                # 创建新索引
                index_params = {
                    "metric_type": "IP",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                
                self.collection.create_index(
                    field_name="embedding",
                    index_params=index_params
                )
                
                # 重新加载集合
                self.collection.load()
                
                logger.info("Milvus索引重建完成")
                
        except Exception as e:
            logger.error(f"Milvus索引重建失败: {e}")
            raise
    
    def close_connection(self):
        """关闭Milvus连接"""
        try:
            if self.collection:
                self.collection.release()
            connections.disconnect("default")
            logger.info("Milvus连接已关闭")
            
        except Exception as e:
            logger.error(f"关闭Milvus连接失败: {e}")


# 全局Milvus向量存储实例
milvus_vector_store = MilvusVectorStore()
