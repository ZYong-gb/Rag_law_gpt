"""
嵌入向量服务
基于Qwen3-Embedding-0.6B模型
"""
import os
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from loguru import logger
from app.config import settings, ModelConfig


class EmbeddingService:
    """嵌入向量服务"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化嵌入模型"""
        try:
            # 使用Sentence Transformers加载Qwen嵌入模型
            model_name = ModelConfig.EMBEDDING_CONFIG["model_name"]
            
            # 如果本地有模型文件，使用本地路径
            if settings.embedding_model_path and os.path.exists(settings.embedding_model_path):
                model_path = settings.embedding_model_path
            else:
                model_path = model_name
            
            self.model = SentenceTransformer(model_path, device=self.device)
            
            # 设置模型为评估模式
            self.model.eval()
            
            logger.info(f"嵌入模型加载成功: {model_path}")
            logger.info(f"使用设备: {self.device}")
            
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            # 备用方案：使用通用的中文嵌入模型
            try:
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                logger.warning("使用备用嵌入模型")
            except Exception as backup_e:
                logger.error(f"备用模型也加载失败: {backup_e}")
                raise
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        将文本编码为向量
        
        Args:
            text: 单个文本或文本列表
            
        Returns:
            numpy数组形式的向量
        """
        try:
            if isinstance(text, str):
                text = [text]
            
            # 使用模型编码
            embeddings = self.model.encode(
                text,
                normalize_embeddings=ModelConfig.EMBEDDING_CONFIG["normalize_embeddings"],
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            raise
    
    def encode_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量编码文档
        
        Args:
            documents: 文档列表
            batch_size: 批处理大小
            
        Returns:
            文档向量矩阵
        """
        try:
            all_embeddings = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_embeddings = self.encode_text(batch)
                all_embeddings.append(batch_embeddings)
                
                logger.info(f"已处理 {min(i + batch_size, len(documents))}/{len(documents)} 个文档")
            
            # 合并所有批次的结果
            if all_embeddings:
                return np.vstack(all_embeddings)
            else:
                return np.array([])
                
        except Exception as e:
            logger.error(f"批量文档编码失败: {e}")
            raise
    
    def compute_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """
        计算查询向量与文档向量的相似度
        
        Args:
            query_embedding: 查询向量
            doc_embeddings: 文档向量矩阵
            
        Returns:
            相似度分数数组
        """
        try:
            # 确保向量是二维的
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # 计算余弦相似度
            # 归一化向量
            query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
            
            query_normalized = query_embedding / (query_norm + 1e-8)
            doc_normalized = doc_embeddings / (doc_norms + 1e-8)
            
            # 计算点积（余弦相似度）
            similarities = np.dot(query_normalized, doc_normalized.T).flatten()
            
            return similarities
            
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        try:
            # 使用一个测试文本获取维度
            test_embedding = self.encode_text("测试文本")
            return test_embedding.shape[-1]
        except Exception as e:
            logger.error(f"获取嵌入维度失败: {e}")
            return settings.embedding_dimension  # 返回配置的默认值


# 全局嵌入服务实例
embedding_service = EmbeddingService()
