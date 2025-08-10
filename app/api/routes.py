"""
API路由定义
"""
import os
import aiofiles
from datetime import datetime
from typing import List, Dict, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database.connection import get_database, check_database_health, check_redis_health
from app.models.schemas import (
    QueryRequest, QueryResponse, DocumentUploadRequest, DocumentUploadResponse,
    UserFeedback, SystemStats, HealthCheck, DocumentResponse
)
from app.models.database import Document, QueryLog
from app.services.rag_factory import get_current_rag_service, RAGComparison
from app.services.document_processor import document_processor
from app.services.cache_service import cache_service
from app.services.vector_store import vector_store
from app.config import settings
from loguru import logger

# 创建路由器
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_legal_question(
    request: QueryRequest,
    db: Session = Depends(get_database)
):
    """
    法律问题查询接口
    """
    try:
        logger.info(f"收到查询请求: {request.question[:100]}...")

        # 获取当前配置的RAG服务并处理查询
        current_rag_service = get_current_rag_service()
        response = await current_rag_service.query(request, db)
        
        return response
        
    except Exception as e:
        logger.error(f"查询处理失败: {e}")
        raise HTTPException(status_code=500, detail="查询处理失败")


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    category: str = Form(None),
    source: str = Form(None),
    db: Session = Depends(get_database)
):
    """
    文档上传接口
    """
    try:
        # 检查文件类型
        allowed_extensions = {'.pdf', '.docx', '.txt'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件类型: {file_ext}"
            )
        
        # 保存上传的文件
        upload_dir = "./data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # 创建文档数据
        from app.models.schemas import DocumentCreate
        document_data = DocumentCreate(
            title=title,
            content="",  # 将在处理时填充
            category=category,
            source=source,
            file_path=file_path,
            file_type=file_ext
        )
        
        # 后台处理文档
        background_tasks.add_task(
            process_document_background,
            file_path,
            document_data,
            db
        )
        
        return DocumentUploadResponse(
            document_id=UUID('00000000-0000-0000-0000-000000000000'),  # 临时ID
            title=title,
            status="processing",
            chunks_created=0,
            message="文档正在后台处理中"
        )
        
    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        raise HTTPException(status_code=500, detail="文档上传失败")


async def process_document_background(file_path: str, document_data, db: Session):
    """后台处理文档"""
    try:
        document_id = document_processor.process_document(file_path, document_data, db)
        logger.info(f"后台文档处理完成: {document_id}")
        
    except Exception as e:
        logger.error(f"后台文档处理失败: {e}")


@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    category: str = None,
    db: Session = Depends(get_database)
):
    """
    获取文档列表
    """
    try:
        query = db.query(Document).filter(Document.is_active == True)
        
        if category:
            query = query.filter(Document.category == category)
        
        documents = query.offset(skip).limit(limit).all()
        
        return documents
        
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取文档列表失败")


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    db: Session = Depends(get_database)
):
    """
    获取单个文档详情
    """
    try:
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.is_active == True
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="文档不存在")
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档详情失败: {e}")
        raise HTTPException(status_code=500, detail="获取文档详情失败")


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: UUID,
    db: Session = Depends(get_database)
):
    """
    删除文档（软删除）
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="文档不存在")
        
        # 软删除
        document.is_active = False
        db.commit()
        
        return {"message": "文档删除成功"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        raise HTTPException(status_code=500, detail="删除文档失败")


@router.post("/feedback")
async def submit_feedback(
    feedback: UserFeedback,
    db: Session = Depends(get_database)
):
    """
    提交用户反馈
    """
    try:
        # 更新查询日志
        query_log = db.query(QueryLog).filter(QueryLog.id == feedback.query_id).first()
        
        if query_log:
            query_log.user_rating = feedback.rating
            query_log.user_feedback = feedback.feedback
            db.commit()
        
        return {"message": "反馈提交成功"}
        
    except Exception as e:
        logger.error(f"提交反馈失败: {e}")
        raise HTTPException(status_code=500, detail="提交反馈失败")


@router.get("/stats", response_model=SystemStats)
async def get_system_stats(db: Session = Depends(get_database)):
    """
    获取系统统计信息
    """
    try:
        # 数据库统计
        total_documents = db.query(Document).filter(Document.is_active == True).count()
        total_queries = db.query(QueryLog).count()
        
        # 向量存储统计
        vector_stats = vector_store.get_stats()
        
        # 缓存统计
        cache_stats = await cache_service.get_stats()
        
        # 计算平均响应时间
        avg_response_time = db.query(QueryLog.response_time).filter(
            QueryLog.response_time.isnot(None)
        ).all()
        
        avg_time = 0.0
        if avg_response_time:
            avg_time = sum(t[0] for t in avg_response_time) / len(avg_response_time)
        
        # 计算缓存命中率
        cache_hit_rate = 0.0
        if cache_stats:
            hits = cache_stats.get("keyspace_hits", 0)
            misses = cache_stats.get("keyspace_misses", 0)
            if hits + misses > 0:
                cache_hit_rate = hits / (hits + misses)
        
        # 获取活跃分类
        categories = db.query(Document.category).filter(
            Document.is_active == True,
            Document.category.isnot(None)
        ).distinct().all()
        
        active_categories = [cat[0] for cat in categories if cat[0]]
        
        return SystemStats(
            total_documents=total_documents,
            total_chunks=vector_stats.get("total_vectors", 0),
            total_queries=total_queries,
            avg_response_time=avg_time,
            cache_hit_rate=cache_hit_rate,
            active_categories=active_categories
        )
        
    except Exception as e:
        logger.error(f"获取系统统计失败: {e}")
        raise HTTPException(status_code=500, detail="获取系统统计失败")


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """
    健康检查接口
    """
    try:
        # 检查各个组件的健康状态
        db_status = "healthy" if check_database_health() else "unhealthy"
        redis_status = "healthy" if check_redis_health() else "unhealthy"
        
        # 检查模型状态（简单检查）
        model_status = "healthy"  # 可以添加更复杂的模型健康检查
        
        overall_status = "healthy" if all([
            db_status == "healthy",
            redis_status == "healthy",
            model_status == "healthy"
        ]) else "unhealthy"
        
        return HealthCheck(
            status=overall_status,
            timestamp=datetime.utcnow(),
            database_status=db_status,
            redis_status=redis_status,
            model_status=model_status,
            version=settings.app_version
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail="健康检查失败")


@router.post("/cache/clear")
async def clear_cache(pattern: str = None):
    """
    清空缓存
    """
    try:
        success = await cache_service.clear_cache(pattern)

        if success:
            return {"message": "缓存清理成功"}
        else:
            raise HTTPException(status_code=500, detail="缓存清理失败")

    except Exception as e:
        logger.error(f"清空缓存失败: {e}")
        raise HTTPException(status_code=500, detail="清空缓存失败")


@router.post("/compare-rag")
async def compare_rag_implementations(
    question: str,
    db: Session = Depends(get_database)
):
    """
    比较不同RAG实现的性能
    """
    try:
        logger.info(f"开始比较RAG实现: {question[:50]}...")

        comparison_results = await RAGComparison.compare_implementations(question, db)

        return {
            "question": question,
            "comparison_results": comparison_results,
            "available_implementations": RAGComparison.get_available_implementations()
        }

    except Exception as e:
        logger.error(f"RAG实现比较失败: {e}")
        raise HTTPException(status_code=500, detail="RAG实现比较失败")


@router.get("/rag-info")
async def get_rag_info():
    """
    获取当前RAG配置信息
    """
    try:
        from app.services.rag_factory import RAGServiceFactory

        current_service = get_current_rag_service()
        available_implementations = RAGServiceFactory.get_available_implementations()

        return {
            "current_framework": settings.rag_framework,
            "current_vector_store": settings.vector_store_type,
            "current_service_type": type(current_service).__name__,
            "available_implementations": available_implementations
        }

    except Exception as e:
        logger.error(f"获取RAG信息失败: {e}")
        raise HTTPException(status_code=500, detail="获取RAG信息失败")
