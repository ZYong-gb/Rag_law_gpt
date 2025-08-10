"""
FAISS向量存储服务
"""
import os
import pickle
import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict, Any
from uuid import UUID
from loguru import logger
from app.config import settings
from app.services.embedding_service import embedding_service


class FAISSVectorStore:
    """FAISS向量存储"""
    
    def __init__(self):
        self.index = None
        self.document_metadata = {}  # 存储文档元数据
        self.dimension = settings.embedding_dimension
        # 创建完整的文件路径
        index_dir = settings.faiss_index_path
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.metadata_path = os.path.join(index_dir, "metadata.pkl")
        self._initialize_index()
    
    def _initialize_index(self):
        """初始化FAISS索引"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # 尝试加载现有索引
            if os.path.exists(self.index_path):
                self.load_index()
            else:
                self.create_new_index()
                
        except Exception as e:
            logger.error(f"FAISS索引初始化失败: {e}")
            self.create_new_index()
    
    def create_new_index(self):
        """创建新的FAISS索引"""
        try:
            # 获取实际的嵌入维度
            self.dimension = embedding_service.get_embedding_dimension()
            
            # 创建FAISS索引 (使用IVF + PQ进行高效搜索)
            quantizer = faiss.IndexFlatIP(self.dimension)  # 内积索引
            self.index = faiss.IndexIVFPQ(
                quantizer, 
                self.dimension, 
                100,  # nlist: 聚类中心数量
                8,    # M: PQ子向量数量
                8     # nbits: 每个子向量的位数
            )
            
            # 如果文档数量较少，使用简单的平面索引
            if not hasattr(self, '_use_simple_index'):
                self.index = faiss.IndexFlatIP(self.dimension)
            
            self.document_metadata = {}
            logger.info(f"创建新的FAISS索引，维度: {self.dimension}")
            
        except Exception as e:
            logger.error(f"创建FAISS索引失败: {e}")
            raise
    
    def add_documents(self, 
                     embeddings: np.ndarray, 
                     document_ids: List[UUID],
                     metadata: List[Dict[str, Any]]):
        """
        添加文档到向量索引
        
        Args:
            embeddings: 文档嵌入向量
            document_ids: 文档ID列表
            metadata: 文档元数据列表
        """
        try:
            # 确保向量是float32类型
            embeddings = embeddings.astype(np.float32)
            
            # 如果是IVF索引且未训练，需要先训练
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                if embeddings.shape[0] >= 100:  # 需要足够的数据进行训练
                    self.index.train(embeddings)
                else:
                    # 数据不够，使用简单索引
                    self.index = faiss.IndexFlatIP(self.dimension)
            
            # 添加向量到索引
            start_id = self.index.ntotal
            self.index.add(embeddings)
            
            # 存储元数据
            for i, (doc_id, meta) in enumerate(zip(document_ids, metadata)):
                self.document_metadata[start_id + i] = {
                    'document_id': str(doc_id),
                    'metadata': meta
                }
            
            logger.info(f"添加了 {len(document_ids)} 个文档到向量索引")
            
        except Exception as e:
            logger.error(f"添加文档到向量索引失败: {e}")
            raise
    
    def search(self, 
               query_embedding: np.ndarray, 
               top_k: int = 5,
               score_threshold: float = 0.0) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        搜索相似文档
        
        Args:
            query_embedding: 查询向量
            top_k: 返回的文档数量
            score_threshold: 相似度阈值
            
        Returns:
            (document_id, score, metadata) 的列表
        """
        try:
            if self.index.ntotal == 0:
                logger.warning("向量索引为空")
                return []
            
            # 确保查询向量格式正确
            query_embedding = query_embedding.astype(np.float32)
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # 执行搜索
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS返回-1表示没有找到足够的结果
                    continue
                    
                if score >= score_threshold:
                    metadata_info = self.document_metadata.get(idx, {})
                    document_id = metadata_info.get('document_id')
                    metadata = metadata_info.get('metadata', {})
                    
                    if document_id:
                        results.append((document_id, float(score), metadata))
            
            logger.info(f"检索到 {len(results)} 个相关文档")
            return results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    def save_index(self):
        """保存索引到磁盘"""
        try:
            # 保存FAISS索引
            faiss.write_index(self.index, self.index_path)
            
            # 保存元数据
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.document_metadata, f)
            
            logger.info(f"向量索引已保存到: {self.index_path}")
            
        except Exception as e:
            logger.error(f"保存向量索引失败: {e}")
            raise
    
    def load_index(self):
        """从磁盘加载索引"""
        try:
            # 加载FAISS索引
            self.index = faiss.read_index(self.index_path)
            
            # 加载元数据
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'rb') as f:
                    self.document_metadata = pickle.load(f)
            else:
                self.document_metadata = {}
            
            logger.info(f"向量索引已加载，包含 {self.index.ntotal} 个向量")
            
        except Exception as e:
            logger.error(f"加载向量索引失败: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__ if self.index else None,
            "metadata_count": len(self.document_metadata)
        }
    
    def clear_index(self):
        """清空索引"""
        try:
            self.create_new_index()
            logger.info("向量索引已清空")
        except Exception as e:
            logger.error(f"清空向量索引失败: {e}")
            raise


# 全局向量存储实例
vector_store = FAISSVectorStore()
