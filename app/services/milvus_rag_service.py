"""
基于Milvus向量数据库的RAG服务
与FAISS版本的对比实现
"""
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID, uuid4
from sqlalchemy.orm import Session
from loguru import logger

from app.config import settings, ModelConfig
from app.models.database import Document, DocumentChunk, QueryLog
from app.models.schemas import QueryRequest, QueryResponse, RetrievedDocument
from app.services.embedding_service import embedding_service
from app.services.milvus_vector_store import milvus_vector_store
from app.services.llm_service import llm_service
from app.services.cache_service import cache_service


class MilvusRAGService:
    """基于Milvus的RAG服务"""
    
    def __init__(self):
        self.top_k = ModelConfig.RAG_CONFIG["top_k"]
        self.score_threshold = ModelConfig.RAG_CONFIG["score_threshold"]
        self.max_context_length = ModelConfig.RAG_CONFIG["max_context_length"]
    
    async def query(self, 
                   request: QueryRequest, 
                   db: Session) -> QueryResponse:
        """
        处理用户查询 - Milvus版本
        
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
            cache_key = f"milvus_query:{hash(request.question.strip().lower())}"
            cached_response = await cache_service.get(cache_key)
            if cached_response:
                logger.info(f"Milvus缓存命中: {request.question[:50]}...")
                return QueryResponse.parse_raw(cached_response)
            
            # 1. 生成查询向量
            query_embedding = embedding_service.encode_text(request.question)
            
            # 2. Milvus向量检索
            retrieved_results = milvus_vector_store.search(
                query_embedding=query_embedding,
                top_k=request.top_k or self.top_k,
                score_threshold=self.score_threshold
            )
            
            # 3. 获取文档详细信息
            retrieved_documents = await self._get_document_details_milvus(
                retrieved_results, db, request.category_filter
            )
            
            # 4. 生成回答
            answer = self._generate_answer(request.question, retrieved_documents)
            
            # 5. 计算置信度
            confidence_score = self._calculate_confidence(retrieved_documents)
            
            # 6. 构建响应
            response_time = time.time() - start_time
            
            response = QueryResponse(
                answer=answer,
                retrieved_documents=[
                    RetrievedDocument(
                        document_id=UUID(doc['document_id']),
                        title=doc['title'],
                        content=doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'],
                        score=doc['score'],
                        source=doc.get('source'),
                        category=doc.get('category')
                    ) for doc in retrieved_documents
                ],
                confidence_score=confidence_score,
                response_time=response_time,
                query_id=query_id
            )
            
            # 7. 记录查询日志
            await self._log_query(request, response, db)
            
            # 8. 缓存结果
            await cache_service.set(cache_key, response.json(), expire=settings.redis_cache_ttl)
            
            return response
            
        except Exception as e:
            logger.error(f"Milvus查询处理失败: {e}")
            return QueryResponse(
                answer="抱歉，处理您的查询时出现错误。请稍后重试。",
                retrieved_documents=[],
                confidence_score=0.0,
                response_time=time.time() - start_time,
                query_id=query_id
            )
    
    async def _get_document_details_milvus(self, 
                                         retrieved_results: List[Tuple[str, float, Dict[str, Any]]],
                                         db: Session,
                                         category_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """获取Milvus检索到的文档详细信息"""
        try:
            documents = []
            
            for chunk_id, score, metadata in retrieved_results:
                # Milvus已经返回了完整的元数据，直接使用
                document_id = metadata.get('document_id')
                category = metadata.get('category')
                
                # 应用分类过滤
                if category_filter and category not in category_filter:
                    continue
                
                documents.append({
                    'document_id': document_id,
                    'chunk_id': chunk_id,
                    'title': metadata.get('title', '未知文档'),
                    'content': metadata.get('content', ''),
                    'category': category,
                    'source': metadata.get('source'),
                    'score': score,
                    'chunk_index': metadata.get('chunk_index', 0)
                })
            
            # 按分数排序
            documents.sort(key=lambda x: x['score'], reverse=True)
            
            return documents
            
        except Exception as e:
            logger.error(f"获取Milvus文档详细信息失败: {e}")
            return []
    
    def _generate_answer(self, question: str, documents: List[Dict[str, Any]]) -> str:
        """生成回答"""
        try:
            if not documents:
                return "抱歉，没有找到相关的法律文档来回答您的问题。建议您：\n1. 尝试使用不同的关键词\n2. 咨询专业律师\n3. 查阅相关法律法规"
            
            # 限制上下文长度
            context_documents = self._limit_context(documents)
            
            # 使用LLM生成回答
            answer = llm_service.generate_legal_answer(question, context_documents)
            
            return answer
            
        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return "抱歉，生成回答时出现错误。"
    
    def _limit_context(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """限制上下文长度"""
        limited_docs = []
        total_length = 0
        
        for doc in documents:
            content_length = len(doc['content'])
            if total_length + content_length <= self.max_context_length:
                limited_docs.append(doc)
                total_length += content_length
            else:
                # 截断最后一个文档
                remaining_length = self.max_context_length - total_length
                if remaining_length > 100:  # 至少保留100个字符
                    doc_copy = doc.copy()
                    doc_copy['content'] = doc['content'][:remaining_length] + "..."
                    limited_docs.append(doc_copy)
                break
        
        return limited_docs
    
    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """计算回答的置信度"""
        if not documents:
            return 0.0
        
        # 基于检索分数和文档数量计算置信度
        avg_score = sum(doc['score'] for doc in documents) / len(documents)
        doc_count_factor = min(len(documents) / self.top_k, 1.0)
        
        confidence = avg_score * doc_count_factor
        return min(confidence, 1.0)
    
    async def _log_query(self, 
                        request: QueryRequest, 
                        response: QueryResponse, 
                        db: Session):
        """记录查询日志"""
        try:
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
            logger.error(f"Milvus查询日志记录失败: {e}")
    
    def load_documents_from_database(self, db: Session):
        """从数据库加载文档到Milvus"""
        try:
            # 获取所有活跃文档的分块
            chunks = db.query(DocumentChunk).join(Document).filter(
                Document.is_active == True
            ).all()
            
            if not chunks:
                logger.warning("没有找到文档分块")
                return
            
            logger.info(f"开始加载 {len(chunks)} 个文档分块到Milvus")
            
            # 批量处理
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                # 提取文本内容
                texts = [chunk.content for chunk in batch_chunks]
                
                # 生成嵌入向量
                embeddings = embedding_service.encode_documents(texts)
                
                # 准备元数据
                chunk_ids = [chunk.id for chunk in batch_chunks]
                metadata_list = []
                
                for chunk in batch_chunks:
                    document = db.query(Document).filter(
                        Document.id == chunk.document_id
                    ).first()
                    
                    metadata_list.append({
                        'chunk_id': str(chunk.id),
                        'document_id': str(chunk.document_id),
                        'title': document.title if document else 'Unknown',
                        'content': chunk.content,
                        'category': document.category if document else None,
                        'source': document.source if document else None,
                        'chunk_index': chunk.chunk_index
                    })
                
                # 添加到Milvus
                milvus_vector_store.add_documents(
                    embeddings=embeddings,
                    document_ids=chunk_ids,
                    metadata=metadata_list
                )
                
                logger.info(f"已处理 {min(i + batch_size, len(chunks))}/{len(chunks)} 个分块")
            
            logger.info("文档加载到Milvus完成")
            
        except Exception as e:
            logger.error(f"加载文档到Milvus失败: {e}")
            raise


# 全局Milvus RAG服务实例
milvus_rag_service = MilvusRAGService()
