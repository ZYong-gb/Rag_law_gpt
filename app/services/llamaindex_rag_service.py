"""
基于LlamaIndex的RAG服务
与LangChain版本的对比实现
"""
import os
import time
import asyncio
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4

from loguru import logger

try:
    # 尝试新版本的导入方式
    from llama_index.core import VectorStoreIndex, Document as LlamaDocument, Settings
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.postprocessor import SimilarityPostprocessor
    from llama_index.core.response.schema import Response

    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.huggingface import HuggingFaceLLM
    from llama_index.vector_stores.faiss import FaissVectorStore

except ImportError:
    # 回退到旧版本的导入方式
    try:
        from llama_index import VectorStoreIndex, Document as LlamaDocument, ServiceContext
        from llama_index.node_parser import SentenceSplitter
        from llama_index.storage.storage_context import StorageContext
        from llama_index.retrievers import VectorIndexRetriever
        from llama_index.query_engine import RetrieverQueryEngine
        from llama_index.postprocessor import SimilarityPostprocessor
        from llama_index.response.schema import Response

        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.huggingface import HuggingFaceLLM
        from llama_index.vector_stores.faiss import FaissVectorStore

        # 兼容性设置
        Settings = None

    except ImportError as e:
        logger.error(f"无法导入LlamaIndex: {e}")
        # 创建占位符类以避免导入错误
        class MockLlamaIndex:
            pass

        VectorStoreIndex = MockLlamaIndex
        LlamaDocument = MockLlamaIndex
        Settings = MockLlamaIndex
        Response = MockLlamaIndex
        SentenceSplitter = MockLlamaIndex
        StorageContext = MockLlamaIndex
        VectorIndexRetriever = MockLlamaIndex
        RetrieverQueryEngine = MockLlamaIndex
        SimilarityPostprocessor = MockLlamaIndex
        HuggingFaceEmbedding = MockLlamaIndex
        HuggingFaceLLM = MockLlamaIndex
        FaissVectorStore = MockLlamaIndex

from sqlalchemy.orm import Session

from app.config import settings, ModelConfig
from app.models.database import Document, DocumentChunk, QueryLog
from app.models.schemas import QueryRequest, QueryResponse, RetrievedDocument
from app.services.cache_service import cache_service


class LlamaIndexRAGService:
    """基于LlamaIndex的RAG服务"""
    
    def __init__(self):
        self.index = None
        self.query_engine = None
        self.embedding_model = None
        self.llm = None
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化LlamaIndex组件"""
        try:
            # 1. 初始化嵌入模型
            self.embedding_model = HuggingFaceEmbedding(
                model_name="Qwen/Qwen2.5-0.6B-Instruct",
                device="cpu",  # 可以改为 "cuda" 如果有GPU
                normalize=True
            )
            
            # 2. 初始化LLM
            self.llm = HuggingFaceLLM(
                model_name="Qwen/Qwen2.5-8B-Instruct",
                tokenizer_name="Qwen/Qwen2.5-8B-Instruct",
                device_map="auto",
                model_kwargs={
                    "torch_dtype": "float16",
                    "trust_remote_code": True
                },
                generate_kwargs={
                    "temperature": settings.temperature,
                    "max_new_tokens": settings.max_tokens,
                    "do_sample": True,
                    "top_p": 0.8
                }
            )
            
            # 3. 设置全局配置
            Settings.embed_model = self.embedding_model
            Settings.llm = self.llm
            Settings.chunk_size = settings.chunk_size
            Settings.chunk_overlap = settings.chunk_overlap
            
            # 4. 初始化向量存储
            self._initialize_vector_store()
            
            logger.info("LlamaIndex组件初始化成功")
            
        except Exception as e:
            logger.error(f"LlamaIndex组件初始化失败: {e}")
            raise
    
    def _initialize_vector_store(self):
        """初始化向量存储"""
        try:
            import faiss
            
            # 创建FAISS索引
            dimension = 768  # Qwen embedding dimension
            faiss_index = faiss.IndexFlatIP(dimension)
            
            # 创建LlamaIndex向量存储
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # 创建空索引
            self.index = VectorStoreIndex([], storage_context=storage_context)
            
            # 创建查询引擎
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=ModelConfig.RAG_CONFIG["top_k"]
            )
            
            # 添加后处理器
            postprocessor = SimilarityPostprocessor(
                similarity_cutoff=ModelConfig.RAG_CONFIG["score_threshold"]
            )
            
            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
                node_postprocessors=[postprocessor]
            )
            
            logger.info("LlamaIndex向量存储初始化成功")
            
        except Exception as e:
            logger.error(f"向量存储初始化失败: {e}")
            raise
    
    def add_documents_to_index(self, documents: List[Dict[str, Any]]):
        """添加文档到索引"""
        try:
            llama_documents = []
            
            for doc_data in documents:
                # 创建LlamaIndex文档对象
                llama_doc = LlamaDocument(
                    text=doc_data['content'],
                    metadata={
                        'document_id': doc_data['document_id'],
                        'title': doc_data['title'],
                        'category': doc_data.get('category'),
                        'source': doc_data.get('source')
                    }
                )
                llama_documents.append(llama_doc)
            
            # 添加到索引
            for doc in llama_documents:
                self.index.insert(doc)
            
            logger.info(f"添加了 {len(llama_documents)} 个文档到LlamaIndex")
            
        except Exception as e:
            logger.error(f"添加文档到LlamaIndex失败: {e}")
            raise
    
    async def query(self, request: QueryRequest, db: Session) -> QueryResponse:
        """
        处理查询请求 - LlamaIndex版本
        
        Args:
            request: 查询请求
            db: 数据库会话
            
        Returns:
            查询响应
        """
        start_time = time.time()
        query_id = uuid4()
        
        try:
            # 检查缓存
            cache_key = f"llamaindex_query:{hash(request.question.strip().lower())}"
            cached_response = await cache_service.get(cache_key)
            if cached_response:
                logger.info(f"LlamaIndex缓存命中: {request.question[:50]}...")
                return QueryResponse.parse_raw(cached_response)
            
            # 使用LlamaIndex查询引擎
            response: Response = self.query_engine.query(request.question)
            
            # 提取检索到的文档
            retrieved_documents = self._extract_retrieved_documents(response)
            
            # 计算置信度
            confidence_score = self._calculate_confidence_llamaindex(response)
            
            # 构建响应
            response_time = time.time() - start_time
            
            query_response = QueryResponse(
                answer=str(response),
                retrieved_documents=retrieved_documents,
                confidence_score=confidence_score,
                response_time=response_time,
                query_id=query_id
            )
            
            # 记录查询日志
            await self._log_query_llamaindex(request, query_response, db)
            
            # 缓存结果
            await cache_service.set(cache_key, query_response.json(), expire=settings.redis_cache_ttl)
            
            return query_response
            
        except Exception as e:
            logger.error(f"LlamaIndex查询处理失败: {e}")
            return QueryResponse(
                answer="抱歉，处理您的查询时出现错误。请稍后重试。",
                retrieved_documents=[],
                confidence_score=0.0,
                response_time=time.time() - start_time,
                query_id=query_id
            )
    
    def _extract_retrieved_documents(self, response: Response) -> List[RetrievedDocument]:
        """从LlamaIndex响应中提取检索到的文档"""
        try:
            retrieved_docs = []
            
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    metadata = node.metadata or {}
                    
                    retrieved_doc = RetrievedDocument(
                        document_id=UUID(metadata.get('document_id', str(uuid4()))),
                        title=metadata.get('title', '未知文档'),
                        content=node.text[:500] + "..." if len(node.text) > 500 else node.text,
                        score=getattr(node, 'score', 0.0),
                        source=metadata.get('source'),
                        category=metadata.get('category')
                    )
                    retrieved_docs.append(retrieved_doc)
            
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"提取检索文档失败: {e}")
            return []
    
    def _calculate_confidence_llamaindex(self, response: Response) -> float:
        """计算LlamaIndex响应的置信度"""
        try:
            if not hasattr(response, 'source_nodes') or not response.source_nodes:
                return 0.0
            
            # 基于检索节点的分数计算置信度
            scores = [getattr(node, 'score', 0.0) for node in response.source_nodes]
            
            if scores:
                avg_score = sum(scores) / len(scores)
                # 归一化到0-1范围
                confidence = min(max(avg_score, 0.0), 1.0)
                return confidence
            
            return 0.0
            
        except Exception as e:
            logger.error(f"计算置信度失败: {e}")
            return 0.0
    
    async def _log_query_llamaindex(self, request: QueryRequest, response: QueryResponse, db: Session):
        """记录LlamaIndex查询日志"""
        try:
            import json
            
            query_log = QueryLog(
                user_id=request.user_id,
                query=request.question,
                response=response.answer,
                retrieved_docs=json.dumps([str(doc.document_id) for doc in response.retrieved_documents]),
                retrieval_score=response.confidence_score,
                response_time=response.response_time,
                token_usage=len(response.answer)  # 简单估算
            )
            
            db.add(query_log)
            db.commit()
            
        except Exception as e:
            logger.error(f"LlamaIndex查询日志记录失败: {e}")
    
    def load_documents_from_database(self, db: Session):
        """从数据库加载文档到LlamaIndex"""
        try:
            # 获取所有活跃文档
            documents = db.query(Document).filter(Document.is_active == True).all()
            
            doc_data_list = []
            for doc in documents:
                doc_data_list.append({
                    'document_id': str(doc.id),
                    'title': doc.title,
                    'content': doc.content,
                    'category': doc.category,
                    'source': doc.source
                })
            
            # 添加到索引
            self.add_documents_to_index(doc_data_list)
            
            logger.info(f"从数据库加载了 {len(doc_data_list)} 个文档到LlamaIndex")
            
        except Exception as e:
            logger.error(f"从数据库加载文档失败: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        try:
            if self.index:
                # LlamaIndex的统计信息
                return {
                    "framework": "LlamaIndex",
                    "index_type": type(self.index).__name__,
                    "embedding_model": "Qwen2.5-0.6B-Instruct",
                    "llm_model": "Qwen2.5-8B-Instruct",
                    "vector_store": "FAISS"
                }
            return {}
            
        except Exception as e:
            logger.error(f"获取LlamaIndex统计失败: {e}")
            return {}


# 全局LlamaIndex RAG服务实例
llamaindex_rag_service = LlamaIndexRAGService()
