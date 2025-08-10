"""
配置管理模块
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置"""
    
    # 应用基础配置
    app_name: str
    app_version: str
    debug: bool
    
    # GPU配置（新增）
    cuda_visible_devices: Optional[str] = None
    cuda_device_order: Optional[str] = None
    
    # 数据库配置
    database_url: str
    
    # Redis配置
    redis_url: str
    redis_cache_ttl: int
    
    # AI模型配置 - VLLM独立部署
    llm_service_url: str
    vllm_model_path: str
    vllm_model_name: str
    embedding_model_path: str
    max_tokens: int
    temperature: float
    
    # FAISS配置
    faiss_index_path: str
    embedding_dimension: int
    
    # 文档处理配置
    chunk_size: int
    chunk_overlap: int
    max_document_size: int
    
    # API配置
    api_host: str
    api_port: int
    api_workers: int
    
    # 日志配置
    log_level: str
    log_file: str
    
    # 安全配置
    secret_key: str
    access_token_expire_minutes: int

    # RAG框架选择
    rag_framework: str
    vector_store_type: str

    # Milvus配置
    milvus_host: str
    milvus_port: int
    milvus_collection_name: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()


class ModelConfig:
    """模型相关配置"""

    @classmethod
    def get_vllm_config(cls):
        """获取VLLM服务配置"""
        return {
            "service_url": settings.llm_service_url,
            "model_path": settings.vllm_model_path,
            "model_name": settings.vllm_model_name,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
        }

    # Qwen3模型配置（保留用于嵌入模型等）
    QWEN_CONFIG = {
        "model_name": "Qwen/Qwen3-8B-Instruct",
        "device_map": "auto",
        "torch_dtype": "float16",
        "trust_remote_code": True,
        "temperature": 0.1,
        "max_tokens": 2048,
    }

    # 嵌入模型配置
    EMBEDDING_CONFIG = {
        "model_name": "Qwen/Qwen3-0.6B-Instruct",
        "device": "cuda",  # 可以改为 "cuda" 如果有GPU
        "normalize_embeddings": True,
    }

    # RAG配置
    RAG_CONFIG = {
        "top_k": 5,  # 检索的文档数量
        "score_threshold": 0.7,  # 相似度阈值
        "max_context_length": 4000,  # 最大上下文长度
    }


class PromptTemplates:
    """提示词模板"""
    
    LEGAL_QA_TEMPLATE = """
你是一个专业的法律助手，请基于以下法律文档内容回答用户的问题。

相关法律文档：
{context}

用户问题：{question}

请注意：
1. 只基于提供的法律文档内容回答问题
2. 如果文档中没有相关信息，请明确说明
3. 提供准确的法条引用
4. 使用专业但易懂的语言
5. 如果涉及复杂法律问题，建议咨询专业律师

回答：
"""

    DOCUMENT_SUMMARY_TEMPLATE = """
请为以下法律文档生成简洁的摘要：

文档内容：
{content}

摘要要求：
1. 突出主要法律条款
2. 包含关键法律概念
3. 长度控制在200字以内

摘要：
"""


def get_rag_config(config_name: str = "production"):
    """
    获取RAG配置

    Args:
        config_name: 配置名称 ("development", "production", "testing")

    Returns:
        RAGConfig实例
    """
    from app.services.advanced_rag_service import RAGConfig, RAGStrategy

    configs = {
        "development": RAGConfig(
            strategy=RAGStrategy.HYBRID,
            top_k=5,
            similarity_threshold=0.6,
            enable_reranking=False,
            enable_query_rewriting=True,
            enable_context_compression=False,
            max_context_length=2000,
            chunk_size=256,
            chunk_overlap=25
        ),
        "production": RAGConfig(
            strategy=RAGStrategy.HYBRID,
            top_k=10,
            similarity_threshold=0.7,
            enable_reranking=True,
            enable_query_rewriting=True,
            enable_context_compression=True,
            max_context_length=4000,
            chunk_size=512,
            chunk_overlap=50
        ),
        "testing": RAGConfig(
            strategy=RAGStrategy.NAIVE,
            top_k=3,
            similarity_threshold=0.5,
            enable_reranking=False,
            enable_query_rewriting=False,
            enable_context_compression=False,
            max_context_length=1000,
            chunk_size=128,
            chunk_overlap=10
        ),
        "self_rag": RAGConfig(
            strategy=RAGStrategy.SELF_RAG,
            top_k=8,
            similarity_threshold=0.7,
            enable_reranking=True,
            enable_query_rewriting=True,
            enable_context_compression=True,
            max_context_length=3500,
            chunk_size=512,
            chunk_overlap=50
        ),
        "corrective": RAGConfig(
            strategy=RAGStrategy.CORRECTIVE,
            top_k=12,
            similarity_threshold=0.6,
            enable_reranking=True,
            enable_query_rewriting=True,
            enable_context_compression=True,
            max_context_length=4500,
            chunk_size=512,
            chunk_overlap=50
        )
    }

    return configs.get(config_name, configs["production"])
