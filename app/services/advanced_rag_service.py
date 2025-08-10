"""
Advanced RAG Service with Best Practices
采用现代RAG最佳实践的高级RAG服务
"""
import os
import time
import asyncio
import json
from typing import List, Dict, Any, Optional, Union
from uuid import UUID, uuid4
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from loguru import logger

# Third-party imports that might be used in methods
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS不可用，将使用备选向量存储")
    FAISS_AVAILABLE = False
    faiss = None

# LlamaIndex imports
try:
    # LlamaIndex Core
    from llama_index.core import VectorStoreIndex, Document as LlamaDocument, Settings
    from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.postprocessor import SimilarityPostprocessor, LLMRerank
    from llama_index.core.response.schema import Response
    from llama_index.core.schema import NodeWithScore, QueryBundle
    from llama_index.core.prompts import PromptTemplate

    # Embeddings and LLMs
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.huggingface import HuggingFaceLLM

    # Vector Stores
    from llama_index.vector_stores.faiss import FaissVectorStore

    # BM25 Retriever - 尝试不同的导入路径
    BM25Retriever = None
    try:
        from llama_index.retrievers.bm25 import BM25Retriever
    except ImportError:
        try:
            from llama_index_retrievers_bm25 import BM25Retriever
        except ImportError:
            logger.warning("BM25Retriever导入失败，将禁用BM25检索功能")

    LLAMAINDEX_AVAILABLE = True

except ImportError as e:
    logger.error(f"LlamaIndex导入失败: {e}")
    LLAMAINDEX_AVAILABLE = False

    # 创建占位符以避免导入错误
    class MockClass:
        pass

    VectorStoreIndex = MockClass
    LlamaDocument = MockClass
    Settings = MockClass
    BaseRetriever = MockClass
    VectorIndexRetriever = MockClass
    RetrieverQueryEngine = MockClass
    SimilarityPostprocessor = MockClass
    LLMRerank = MockClass
    Response = MockClass
    NodeWithScore = MockClass
    QueryBundle = MockClass
    PromptTemplate = MockClass
    HuggingFaceEmbedding = MockClass
    HuggingFaceLLM = MockClass
    FaissVectorStore = MockClass
    StorageContext = MockClass
    SentenceSplitter = MockClass
    BM25Retriever = None

from app.config import settings, ModelConfig, get_rag_config
from app.models.database import Document, DocumentChunk, QueryLog
from app.models.schemas import QueryRequest, QueryResponse, RetrievedDocument
from app.services.cache_service import cache_service


class RAGStrategy(Enum):
    """RAG策略枚举"""
    NAIVE = "naive"
    HYBRID = "hybrid"
    SELF_RAG = "self_rag"
    CORRECTIVE = "corrective"


@dataclass
class RAGConfig:
    """RAG配置"""
    strategy: RAGStrategy = RAGStrategy.HYBRID
    top_k: int = 10
    similarity_threshold: float = 0.7
    enable_reranking: bool = True
    enable_query_rewriting: bool = True
    enable_context_compression: bool = True
    max_context_length: int = 4000
    chunk_size: int = 512
    chunk_overlap: int = 50


class HybridRetriever(BaseRetriever):
    """混合检索器：结合向量检索和BM25"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        bm25_retriever: Optional[Any] = None,  # 使用Any类型以避免类型检查问题
        alpha: float = 0.7  # 向量检索权重
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha
        super().__init__()
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """执行混合检索"""
        try:
            # 向量检索
            vector_nodes = self.vector_retriever.retrieve(query_bundle)

            # BM25检索（如果可用）
            if self.bm25_retriever is not None:
                bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
                # 合并和重新评分
                combined_nodes = self._combine_and_rescore(vector_nodes, bm25_nodes)
                return combined_nodes
            else:
                # 只使用向量检索
                return vector_nodes

        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            return []
    
    def _combine_and_rescore(
        self, 
        vector_nodes: List[NodeWithScore], 
        bm25_nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """合并并重新评分检索结果"""
        # 创建节点ID到分数的映射
        vector_scores = {node.node.node_id: node.score for node in vector_nodes}
        bm25_scores = {node.node.node_id: node.score for node in bm25_nodes}
        
        # 获取所有唯一节点
        all_node_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        
        # 重新评分
        rescored_nodes = []
        for node_id in all_node_ids:
            vector_score = vector_scores.get(node_id, 0.0)
            bm25_score = bm25_scores.get(node_id, 0.0)
            
            # 混合评分
            combined_score = self.alpha * vector_score + (1 - self.alpha) * bm25_score
            
            # 找到对应的节点
            node = None
            for vn in vector_nodes:
                if vn.node.node_id == node_id:
                    node = vn.node
                    break
            if not node:
                for bn in bm25_nodes:
                    if bn.node.node_id == node_id:
                        node = bn.node
                        break
            
            if node:
                rescored_nodes.append(NodeWithScore(node=node, score=combined_score))
        
        # 按分数排序
        rescored_nodes.sort(key=lambda x: x.score, reverse=True)
        
        return rescored_nodes


class QueryRewriter:
    """查询重写器"""

    def __init__(self, llm):
        self.llm = llm
        if LLAMAINDEX_AVAILABLE:
            self.rewrite_prompt = PromptTemplate(
                "请将以下用户查询重写为更适合检索的形式，保持原意但使用更准确的法律术语：\n"
                "原查询：{query}\n"
                "重写后的查询："
            )
        else:
            self.rewrite_prompt = None
    
    async def rewrite_query(self, query: str) -> str:
        """重写查询"""
        try:
            if self.rewrite_prompt is None:
                logger.warning("PromptTemplate不可用，返回原查询")
                return query

            prompt = self.rewrite_prompt.format(query=query)
            rewritten = await self.llm.acomplete(prompt)
            return rewritten.text.strip()
        except Exception as e:
            logger.error(f"查询重写失败: {e}")
            return query


class ContextCompressor:
    """上下文压缩器"""

    def __init__(self, llm, max_length: int = 4000):
        self.llm = llm
        self.max_length = max_length
        if LLAMAINDEX_AVAILABLE:
            self.compression_prompt = PromptTemplate(
                "请压缩以下文档内容，保留与查询相关的关键信息：\n"
                "查询：{query}\n"
                "文档内容：{content}\n"
                "压缩后的内容："
            )
        else:
            self.compression_prompt = None
    
    async def compress_context(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """压缩上下文"""
        try:
            if self.compression_prompt is None:
                logger.warning("PromptTemplate不可用，返回前3个文档")
                return documents[:3]

            compressed_docs = []
            total_length = 0

            for doc in documents:
                content = doc['content']

                # 如果内容过长，进行压缩
                if len(content) > 1000:
                    prompt = self.compression_prompt.format(query=query, content=content)
                    compressed = await self.llm.acomplete(prompt)
                    content = compressed.text.strip()

                # 检查总长度
                if total_length + len(content) <= self.max_length:
                    doc_copy = doc.copy()
                    doc_copy['content'] = content
                    compressed_docs.append(doc_copy)
                    total_length += len(content)
                else:
                    break

            return compressed_docs

        except Exception as e:
            logger.error(f"上下文压缩失败: {e}")
            return documents[:3]  # 返回前3个文档作为备选


class AdvancedRAGService:
    """高级RAG服务 - 采用最佳实践"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.index = None
        self.vector_retriever = None
        self.bm25_retriever = None
        self.hybrid_retriever = None
        self.query_engine = None
        self.query_rewriter = None
        self.context_compressor = None
        self.embedding_model = None
        self.llm = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化所有组件"""
        try:
            if not LLAMAINDEX_AVAILABLE:
                logger.error("LlamaIndex不可用，无法初始化高级RAG服务")
                raise ImportError("LlamaIndex不可用")

            # 1. 初始化嵌入模型
            self.embedding_model = HuggingFaceEmbedding(
                model_name=ModelConfig.EMBEDDING_CONFIG["model_name"],
                device=ModelConfig.EMBEDDING_CONFIG["device"],
                normalize=ModelConfig.EMBEDDING_CONFIG["normalize_embeddings"]
            )
            
            # 2. 初始化LLM
            self.llm = HuggingFaceLLM(
                model_name=ModelConfig.QWEN_CONFIG["model_name"],
                tokenizer_name=ModelConfig.QWEN_CONFIG["model_name"],
                device_map="auto",
                model_kwargs={
                    "torch_dtype": "float16",
                    "trust_remote_code": True
                },
                generate_kwargs={
                    "temperature": ModelConfig.QWEN_CONFIG["temperature"],
                    "max_new_tokens": ModelConfig.QWEN_CONFIG["max_tokens"],
                    "do_sample": True,
                    "top_p": 0.8
                }
            )
            
            # 3. 设置全局配置
            Settings.embed_model = self.embedding_model
            Settings.llm = self.llm
            Settings.chunk_size = self.config.chunk_size
            Settings.chunk_overlap = self.config.chunk_overlap
            
            # 4. 初始化查询重写器和上下文压缩器
            self.query_rewriter = QueryRewriter(self.llm)
            self.context_compressor = ContextCompressor(self.llm, self.config.max_context_length)
            
            # 5. 初始化向量存储
            self._initialize_vector_store()
            
            logger.info("高级RAG组件初始化成功")
            
        except Exception as e:
            logger.error(f"高级RAG组件初始化失败: {e}")
            raise
    
    def _initialize_vector_store(self):
        """初始化向量存储和检索器"""
        try:
            if not FAISS_AVAILABLE:
                raise ImportError("FAISS不可用")

            # 创建FAISS索引
            dimension = 768  # Qwen embedding dimension
            faiss_index = faiss.IndexFlatIP(dimension)
            
            # 创建向量存储
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # 创建索引
            self.index = VectorStoreIndex([], storage_context=storage_context)
            
            # 创建向量检索器
            self.vector_retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=self.config.top_k
            )
            
            logger.info("向量存储和检索器初始化成功")
            
        except Exception as e:
            logger.error(f"向量存储初始化失败: {e}")
            raise
    
    def _create_hybrid_retriever(self, documents: List[LlamaDocument]):
        """创建混合检索器"""
        try:
            # 检查BM25检索器是否可用
            if BM25Retriever is None:
                logger.warning("BM25Retriever不可用，使用纯向量检索")
                self.hybrid_retriever = self.vector_retriever
                return

            # 创建BM25检索器
            self.bm25_retriever = BM25Retriever.from_defaults(
                docstore=self.index.docstore,
                similarity_top_k=self.config.top_k
            )

            # 创建混合检索器
            self.hybrid_retriever = HybridRetriever(
                vector_retriever=self.vector_retriever,
                bm25_retriever=self.bm25_retriever,
                alpha=0.7  # 向量检索权重
            )

            logger.info("混合检索器创建成功")

        except Exception as e:
            logger.error(f"混合检索器创建失败: {e}")
            # 回退到向量检索
            self.hybrid_retriever = self.vector_retriever

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
                        'source': doc_data.get('source'),
                        'chunk_id': doc_data.get('chunk_id')
                    }
                )
                llama_documents.append(llama_doc)

            # 使用高级节点解析器
            node_parser = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separator=" "
            )

            # 解析文档为节点
            nodes = node_parser.get_nodes_from_documents(llama_documents)

            # 添加到索引
            self.index.insert_nodes(nodes)

            # 创建混合检索器
            self._create_hybrid_retriever(llama_documents)

            logger.info(f"添加了 {len(llama_documents)} 个文档到高级RAG索引")

        except Exception as e:
            logger.error(f"添加文档到高级RAG索引失败: {e}")
            raise

    async def query(self, request: QueryRequest, db: Session) -> QueryResponse:
        """
        处理查询请求 - 高级RAG版本

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
            cache_key = f"advanced_rag:{hash(request.question.strip().lower())}"
            cached_response = await cache_service.get(cache_key)
            if cached_response:
                logger.info(f"高级RAG缓存命中: {request.question[:50]}...")
                return QueryResponse.parse_raw(cached_response)

            # 根据策略选择处理方式
            if self.config.strategy == RAGStrategy.HYBRID:
                response = await self._hybrid_query(request)
            elif self.config.strategy == RAGStrategy.SELF_RAG:
                response = await self._self_rag_query(request)
            elif self.config.strategy == RAGStrategy.CORRECTIVE:
                response = await self._corrective_rag_query(request)
            else:
                response = await self._naive_query(request)

            # 提取检索到的文档
            retrieved_documents = self._extract_retrieved_documents(response)

            # 计算置信度
            confidence_score = self._calculate_confidence(response)

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
            await self._log_query(request, query_response, db)

            # 缓存结果
            await cache_service.set(cache_key, query_response.json(), expire=settings.redis_cache_ttl)

            return query_response

        except Exception as e:
            logger.error(f"高级RAG查询处理失败: {e}")
            return QueryResponse(
                answer="抱歉，处理您的查询时出现错误。请稍后重试。",
                retrieved_documents=[],
                confidence_score=0.0,
                response_time=time.time() - start_time,
                query_id=query_id
            )

    async def _hybrid_query(self, request: QueryRequest) -> Response:
        """混合RAG查询"""
        try:
            # 1. 查询重写（如果启用）
            query = request.question
            if self.config.enable_query_rewriting:
                query = await self.query_rewriter.rewrite_query(query)
                logger.info(f"查询重写: {request.question} -> {query}")

            # 2. 创建查询引擎
            postprocessors = []

            # 添加相似度后处理器
            postprocessors.append(
                SimilarityPostprocessor(similarity_cutoff=self.config.similarity_threshold)
            )

            # 添加重排序器（如果启用）
            if self.config.enable_reranking:
                postprocessors.append(
                    LLMRerank(
                        llm=self.llm,
                        top_n=self.config.top_k // 2
                    )
                )

            # 创建查询引擎
            query_engine = RetrieverQueryEngine(
                retriever=self.hybrid_retriever,
                node_postprocessors=postprocessors
            )

            # 3. 执行查询
            response = query_engine.query(query)

            return response

        except Exception as e:
            logger.error(f"混合查询失败: {e}")
            raise

    async def _self_rag_query(self, request: QueryRequest) -> Response:
        """Self-RAG查询 - 带自我反思"""
        try:
            # 1. 初始查询
            initial_response = await self._hybrid_query(request)

            # 2. 自我评估
            if LLAMAINDEX_AVAILABLE:
                evaluation_prompt = PromptTemplate(
                    "请评估以下回答的质量和准确性（1-10分）：\n"
                    "问题：{question}\n"
                    "回答：{answer}\n"
                    "评分（1-10）："
                )
                eval_prompt = evaluation_prompt.format(
                    question=request.question,
                    answer=str(initial_response)
                )
            else:
                # 简单的字符串格式化
                eval_prompt = (
                    f"请评估以下回答的质量和准确性（1-10分）：\n"
                    f"问题：{request.question}\n"
                    f"回答：{str(initial_response)}\n"
                    f"评分（1-10）："
                )

            evaluation = await self.llm.acomplete(eval_prompt)

            try:
                score = float(evaluation.text.strip())
                if score < 7.0:  # 如果评分较低，尝试改进
                    logger.info(f"Self-RAG检测到低质量回答，评分: {score}")
                    return await self._improve_response(request, initial_response)
            except ValueError:
                pass

            return initial_response

        except Exception as e:
            logger.error(f"Self-RAG查询失败: {e}")
            return await self._hybrid_query(request)

    async def _corrective_rag_query(self, request: QueryRequest) -> Response:
        """Corrective RAG查询 - 带纠错机制"""
        try:
            # 1. 初始检索
            initial_response = await self._hybrid_query(request)

            # 2. 检查是否需要额外检索
            # 获取检索内容
            retrieved_content = ""
            if hasattr(initial_response, 'source_nodes'):
                retrieved_content = "\n".join([node.text for node in initial_response.source_nodes[:3]])

            if LLAMAINDEX_AVAILABLE:
                relevance_prompt = PromptTemplate(
                    "检索到的文档是否足够回答以下问题？回答'是'或'否'：\n"
                    "问题：{question}\n"
                    "检索到的内容：{content}\n"
                    "判断："
                )
                relevance_check = relevance_prompt.format(
                    question=request.question,
                    content=retrieved_content
                )
            else:
                # 简单的字符串格式化
                relevance_check = (
                    f"检索到的文档是否足够回答以下问题？回答'是'或'否'：\n"
                    f"问题：{request.question}\n"
                    f"检索到的内容：{retrieved_content}\n"
                    f"判断："
                )

            relevance_result = await self.llm.acomplete(relevance_check)

            if "否" in relevance_result.text:
                logger.info("Corrective RAG检测到检索不足，进行额外检索")
                # 使用不同的查询策略重新检索
                expanded_query = f"{request.question} 相关法律条文 法规"
                expanded_request = QueryRequest(
                    question=expanded_query,
                    user_id=request.user_id,
                    top_k=request.top_k
                )
                return await self._hybrid_query(expanded_request)

            return initial_response

        except Exception as e:
            logger.error(f"Corrective RAG查询失败: {e}")
            return await self._hybrid_query(request)

    async def _naive_query(self, request: QueryRequest) -> Response:
        """简单RAG查询"""
        try:
            # 创建简单查询引擎
            query_engine = RetrieverQueryEngine(
                retriever=self.vector_retriever,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=self.config.similarity_threshold)
                ]
            )

            return query_engine.query(request.question)

        except Exception as e:
            logger.error(f"简单查询失败: {e}")
            raise

    async def _improve_response(self, request: QueryRequest, initial_response: Response) -> Response:
        """改进回答质量"""
        try:
            # 使用更宽松的检索条件重新查询
            improved_retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=self.config.top_k * 2  # 检索更多文档
            )

            query_engine = RetrieverQueryEngine(
                retriever=improved_retriever,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=self.config.similarity_threshold * 0.8)
                ]
            )

            return query_engine.query(request.question)

        except Exception as e:
            logger.error(f"改进回答失败: {e}")
            return initial_response

    def _extract_retrieved_documents(self, response: Response) -> List[RetrievedDocument]:
        """从响应中提取检索到的文档"""
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

    def _calculate_confidence(self, response: Response) -> float:
        """计算置信度"""
        try:
            if not hasattr(response, 'source_nodes') or not response.source_nodes:
                return 0.0

            # 基于检索节点的分数和数量计算置信度
            scores = [getattr(node, 'score', 0.0) for node in response.source_nodes]

            if scores:
                avg_score = sum(scores) / len(scores)
                doc_count_factor = min(len(scores) / self.config.top_k, 1.0)
                confidence = avg_score * doc_count_factor
                return min(max(confidence, 0.0), 1.0)

            return 0.0

        except Exception as e:
            logger.error(f"计算置信度失败: {e}")
            return 0.0

    async def _log_query(self, request: QueryRequest, response: QueryResponse, db: Session):
        """记录查询日志"""
        try:
            query_log = QueryLog(
                user_id=request.user_id,
                query=request.question,
                response=response.answer,
                retrieved_docs=json.dumps([str(doc.document_id) for doc in response.retrieved_documents]),
                retrieval_score=response.confidence_score,
                response_time=response.response_time,
                token_usage=len(response.answer)
            )

            db.add(query_log)
            db.commit()

        except Exception as e:
            logger.error(f"查询日志记录失败: {e}")

    def load_documents_from_database(self, db: Session):
        """从数据库加载文档"""
        try:
            # 获取所有活跃文档的分块
            chunks = db.query(DocumentChunk).join(Document).filter(
                Document.is_active == True
            ).all()

            doc_data_list = []
            for chunk in chunks:
                doc_data_list.append({
                    'document_id': str(chunk.document_id),
                    'chunk_id': str(chunk.id),
                    'title': chunk.document.title,
                    'content': chunk.content,
                    'category': chunk.document.category,
                    'source': chunk.document.source
                })

            # 添加到索引
            self.add_documents_to_index(doc_data_list)

            logger.info(f"从数据库加载了 {len(doc_data_list)} 个文档分块到高级RAG")

        except Exception as e:
            logger.error(f"从数据库加载文档失败: {e}")
            raise

    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        try:
            return {
                "framework": "Advanced RAG (LlamaIndex)",
                "strategy": self.config.strategy.value,
                "index_type": type(self.index).__name__ if self.index else "None",
                "embedding_model": ModelConfig.EMBEDDING_CONFIG["model_name"],
                "llm_model": ModelConfig.QWEN_CONFIG["model_name"],
                "vector_store": "FAISS",
                "features": {
                    "hybrid_retrieval": True,
                    "query_rewriting": self.config.enable_query_rewriting,
                    "reranking": self.config.enable_reranking,
                    "context_compression": self.config.enable_context_compression,
                    "self_rag": self.config.strategy == RAGStrategy.SELF_RAG,
                    "corrective_rag": self.config.strategy == RAGStrategy.CORRECTIVE
                }
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}


# 创建不同配置的RAG服务实例
def create_advanced_rag_service(
    strategy: RAGStrategy = RAGStrategy.HYBRID,
    config_name: str = None
) -> AdvancedRAGService:
    """创建高级RAG服务实例"""
    if config_name:
        # 使用预定义配置
        config = get_rag_config(config_name)
    else:
        # 使用自定义策略
        config = get_rag_config("production")
        config.strategy = strategy

    return AdvancedRAGService(config)


# 全局高级RAG服务实例 - 使用生产环境配置
try:
    if LLAMAINDEX_AVAILABLE:
        advanced_rag_service = create_advanced_rag_service(config_name="production")
    else:
        logger.warning("LlamaIndex不可用，高级RAG服务未初始化")
        advanced_rag_service = None
except Exception as e:
    logger.error(f"高级RAG服务初始化失败: {e}")
    advanced_rag_service = None
