"""
RAG服务工厂
根据配置选择不同的RAG实现
"""
import time
from typing import Union
from loguru import logger

from app.config import settings
from app.services.rag_service import rag_service as langchain_rag_service
from app.services.llamaindex_rag_service import llamaindex_rag_service
from app.services.milvus_rag_service import milvus_rag_service
from app.services.advanced_rag_service import advanced_rag_service, create_advanced_rag_service, RAGStrategy


class RAGServiceFactory:
    """RAG服务工厂"""
    
    @staticmethod
    def get_rag_service():
        """
        根据配置获取RAG服务实例
        
        Returns:
            RAG服务实例
        """
        framework = settings.rag_framework.lower()
        vector_store = settings.vector_store_type.lower()

        logger.info(f"使用RAG框架: {framework}, 向量存储: {vector_store}")

        # 优先使用高级RAG服务
        if framework == "advanced" or framework == "llamaindex_advanced":
            return advanced_rag_service
        elif framework == "llamaindex":
            return llamaindex_rag_service
        elif framework == "langchain" and vector_store == "milvus":
            return milvus_rag_service
        else:
            # 默认使用高级RAG服务
            return advanced_rag_service
    
    @staticmethod
    def get_available_implementations():
        """获取可用的实现列表"""
        return {
            "frameworks": ["langchain", "llamaindex"],
            "vector_stores": ["faiss", "milvus"],
            "combinations": [
                {
                    "name": "LangChain + FAISS",
                    "framework": "langchain",
                    "vector_store": "faiss",
                    "description": "轻量级本地向量搜索，适合中小规模数据",
                    "pros": ["部署简单", "无需额外服务", "性能稳定"],
                    "cons": ["扩展性有限", "不支持分布式"]
                },
                {
                    "name": "LangChain + Milvus",
                    "framework": "langchain", 
                    "vector_store": "milvus",
                    "description": "企业级向量数据库，支持大规模数据和分布式部署",
                    "pros": ["高扩展性", "支持分布式", "丰富的索引类型", "数据持久化"],
                    "cons": ["部署复杂", "资源消耗较大"]
                },
                {
                    "name": "LlamaIndex + FAISS",
                    "framework": "llamaindex",
                    "vector_store": "faiss", 
                    "description": "专为RAG优化的框架，提供更高级的查询能力",
                    "pros": ["RAG专用优化", "丰富的查询类型", "易于扩展"],
                    "cons": ["学习曲线较陡", "社区相对较小"]
                }
            ]
        }


# 全局RAG服务实例
def get_current_rag_service():
    """获取当前配置的RAG服务"""
    return RAGServiceFactory.get_rag_service()


# 比较不同实现的性能
class RAGComparison:
    """RAG实现比较工具"""
    
    @staticmethod
    async def compare_implementations(question: str, db):
        """比较不同RAG实现的性能"""
        from app.models.schemas import QueryRequest
        
        request = QueryRequest(
            question=question,
            user_id="comparison_test"
        )
        
        results = {}
        
        # 测试LangChain + FAISS
        try:
            start_time = time.time()
            langchain_result = await langchain_rag_service.query(request, db)
            langchain_time = time.time() - start_time
            
            results["langchain_faiss"] = {
                "response_time": langchain_time,
                "confidence": langchain_result.confidence_score,
                "retrieved_docs": len(langchain_result.retrieved_documents),
                "answer_length": len(langchain_result.answer)
            }
        except Exception as e:
            results["langchain_faiss"] = {"error": str(e)}
        
        # 测试LlamaIndex
        try:
            start_time = time.time()
            llamaindex_result = await llamaindex_rag_service.query(request, db)
            llamaindex_time = time.time() - start_time
            
            results["llamaindex"] = {
                "response_time": llamaindex_time,
                "confidence": llamaindex_result.confidence_score,
                "retrieved_docs": len(llamaindex_result.retrieved_documents),
                "answer_length": len(llamaindex_result.answer)
            }
        except Exception as e:
            results["llamaindex"] = {"error": str(e)}
        
        # 测试Milvus
        try:
            start_time = time.time()
            milvus_result = await milvus_rag_service.query(request, db)
            milvus_time = time.time() - start_time
            
            results["milvus"] = {
                "response_time": milvus_time,
                "confidence": milvus_result.confidence_score,
                "retrieved_docs": len(milvus_result.retrieved_documents),
                "answer_length": len(milvus_result.answer)
            }
        except Exception as e:
            results["milvus"] = {"error": str(e)}
        
        return results
