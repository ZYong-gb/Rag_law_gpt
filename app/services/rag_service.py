"""
RAG (Retrieval-Augmented Generation) 服务
整合检索和生成功能
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
from app.services.vector_store import vector_store
from app.services.llm_service import llm_service
from app.services.cache_service import cache_service


class RAGService:
    """RAG服务"""
    
    def __init__(self):
        self.top_k = ModelConfig.RAG_CONFIG["top_k"]
        self.score_threshold = ModelConfig.RAG_CONFIG["score_threshold"]
        self.max_context_length = ModelConfig.RAG_CONFIG["max_context_length"]
    
    async def query(self, 
                   request: QueryRequest, 
                   db: Session) -> QueryResponse:
        """
        处理用户查询
        
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
            cached_response = await self._check_cache(request.question)
            if cached_response:
                logger.info(f"缓存命中: {request.question[:50]}...")
                return cached_response
            
            # 1. 生成查询向量
            query_embedding = embedding_service.encode_text(request.question)
            
            # 2. 向量检索
            retrieved_results = vector_store.search(
                query_embedding=query_embedding,
                top_k=request.top_k or self.top_k,
                score_threshold=self.score_threshold
            )
            
            # 3. 获取文档详细信息
            retrieved_documents = await self._get_document_details(
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
            await self._cache_response(request.question, response)
            
            return response
            
        except Exception as e:
            logger.error(f"查询处理失败: {e}")
            # 返回错误响应
            return QueryResponse(
                answer="抱歉，处理您的查询时出现错误。请稍后重试。",
                retrieved_documents=[],
                confidence_score=0.0,
                response_time=time.time() - start_time,
                query_id=query_id
            )
    
    async def _check_cache(self, question: str) -> Optional[QueryResponse]:
        """检查缓存中是否有相似问题的答案"""
        try:
            cache_key = f"query:{hash(question.strip().lower())}"
            cached_data = await cache_service.get(cache_key)
            
            if cached_data:
                return QueryResponse.parse_raw(cached_data)
            
            return None
            
        except Exception as e:
            logger.error(f"缓存检查失败: {e}")
            return None
    
    async def _cache_response(self, question: str, response: QueryResponse):
        """缓存查询响应"""
        try:
            cache_key = f"query:{hash(question.strip().lower())}"
            await cache_service.set(
                cache_key, 
                response.json(), 
                expire=settings.redis_cache_ttl
            )
            
        except Exception as e:
            logger.error(f"缓存响应失败: {e}")
    
    async def _get_document_details(self, 
                                  retrieved_results: List[Tuple[str, float, Dict[str, Any]]],
                                  db: Session,
                                  category_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """获取检索到的文档详细信息"""
        try:
            documents = []
            
            for chunk_id, score, metadata in retrieved_results:
                # 获取分块信息
                chunk = db.query(DocumentChunk).filter(
                    DocumentChunk.id == UUID(chunk_id)
                ).first()
                
                if not chunk:
                    continue
                
                # 获取文档信息
                document = db.query(Document).filter(
                    Document.id == chunk.document_id
                ).first()
                
                if not document or not document.is_active:
                    continue
                
                # 应用分类过滤
                if category_filter and document.category not in category_filter:
                    continue
                
                documents.append({
                    'document_id': str(document.id),
                    'chunk_id': str(chunk.id),
                    'title': document.title,
                    'content': chunk.content,
                    'category': document.category,
                    'source': document.source,
                    'score': score,
                    'chunk_index': chunk.chunk_index
                })
            
            # 按分数排序
            documents.sort(key=lambda x: x['score'], reverse=True)
            
            return documents
            
        except Exception as e:
            logger.error(f"获取文档详细信息失败: {e}")
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
            logger.error(f"查询日志记录失败: {e}")


# 全局RAG服务实例
rag_service = RAGService()
