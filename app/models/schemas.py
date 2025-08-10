"""
Pydantic数据模型
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID


class DocumentBase(BaseModel):
    """文档基础模型"""
    title: str = Field(..., description="文档标题")
    content: str = Field(..., description="文档内容")
    category: Optional[str] = Field(None, description="法律分类")
    source: Optional[str] = Field(None, description="文档来源")
    summary: Optional[str] = Field(None, description="文档摘要")


class DocumentCreate(DocumentBase):
    """创建文档模型"""
    file_path: Optional[str] = Field(None, description="文件路径")
    file_type: Optional[str] = Field(None, description="文件类型")


class DocumentResponse(DocumentBase):
    """文档响应模型"""
    id: UUID
    file_path: Optional[str]
    file_type: Optional[str]
    created_at: datetime
    updated_at: datetime
    is_active: bool
    
    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    user_id: Optional[str] = Field(None, description="用户ID")
    category_filter: Optional[List[str]] = Field(None, description="分类过滤")
    top_k: Optional[int] = Field(5, description="检索文档数量", ge=1, le=20)
    include_source: bool = Field(True, description="是否包含来源信息")


class RetrievedDocument(BaseModel):
    """检索到的文档"""
    document_id: UUID
    title: str
    content: str
    score: float = Field(..., description="相似度分数")
    source: Optional[str] = None
    category: Optional[str] = None


class QueryResponse(BaseModel):
    """查询响应模型"""
    answer: str = Field(..., description="回答内容")
    retrieved_documents: List[RetrievedDocument] = Field(..., description="检索到的相关文档")
    confidence_score: float = Field(..., description="置信度分数", ge=0.0, le=1.0)
    response_time: float = Field(..., description="响应时间（秒）")
    query_id: UUID = Field(..., description="查询ID")


class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    title: str = Field(..., description="文档标题")
    category: Optional[str] = Field(None, description="法律分类")
    source: Optional[str] = Field(None, description="文档来源")


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    document_id: UUID
    title: str
    status: str = Field(..., description="处理状态")
    chunks_created: int = Field(..., description="创建的分块数量")
    message: str = Field(..., description="处理消息")


class UserFeedback(BaseModel):
    """用户反馈模型"""
    query_id: UUID = Field(..., description="查询ID")
    rating: int = Field(..., description="评分", ge=1, le=5)
    feedback: Optional[str] = Field(None, description="反馈内容", max_length=1000)


class SystemStats(BaseModel):
    """系统统计信息"""
    total_documents: int
    total_chunks: int
    total_queries: int
    avg_response_time: float
    cache_hit_rate: float
    active_categories: List[str]


class HealthCheck(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    timestamp: datetime
    database_status: str
    redis_status: str
    model_status: str
    version: str
