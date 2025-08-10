"""
数据库模型定义
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class Document(Base):
    """法律文档表"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False, comment="文档标题")
    content = Column(Text, nullable=False, comment="文档内容")
    file_path = Column(String(1000), nullable=True, comment="文件路径")
    file_type = Column(String(50), nullable=True, comment="文件类型")
    category = Column(String(100), nullable=True, comment="法律分类")
    source = Column(String(200), nullable=True, comment="文档来源")
    summary = Column(Text, nullable=True, comment="文档摘要")
    
    # 元数据
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    is_active = Column(Boolean, default=True, comment="是否激活")
    
    # 索引
    __table_args__ = (
        Index('idx_documents_category', 'category'),
        Index('idx_documents_source', 'source'),
        Index('idx_documents_created_at', 'created_at'),
    )


class DocumentChunk(Base):
    """文档分块表"""
    __tablename__ = "document_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False, comment="文档ID")
    chunk_index = Column(Integer, nullable=False, comment="分块索引")
    content = Column(Text, nullable=False, comment="分块内容")
    embedding_vector = Column(Text, nullable=True, comment="向量表示（JSON格式）")
    
    # 分块元数据
    start_char = Column(Integer, nullable=True, comment="起始字符位置")
    end_char = Column(Integer, nullable=True, comment="结束字符位置")
    token_count = Column(Integer, nullable=True, comment="token数量")
    
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    
    # 索引
    __table_args__ = (
        Index('idx_chunks_document_id', 'document_id'),
        Index('idx_chunks_document_chunk', 'document_id', 'chunk_index'),
    )


class QueryLog(Base):
    """查询日志表"""
    __tablename__ = "query_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100), nullable=True, comment="用户ID")
    query = Column(Text, nullable=False, comment="用户查询")
    response = Column(Text, nullable=True, comment="系统回答")
    
    # 检索信息
    retrieved_docs = Column(Text, nullable=True, comment="检索到的文档ID列表")
    retrieval_score = Column(Float, nullable=True, comment="检索相关性分数")
    
    # 性能指标
    response_time = Column(Float, nullable=True, comment="响应时间（秒）")
    token_usage = Column(Integer, nullable=True, comment="使用的token数量")
    
    # 用户反馈
    user_rating = Column(Integer, nullable=True, comment="用户评分(1-5)")
    user_feedback = Column(Text, nullable=True, comment="用户反馈")
    
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    
    # 索引
    __table_args__ = (
        Index('idx_query_logs_user_id', 'user_id'),
        Index('idx_query_logs_created_at', 'created_at'),
    )


class LegalCategory(Base):
    """法律分类表"""
    __tablename__ = "legal_categories"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True, comment="分类名称")
    description = Column(Text, nullable=True, comment="分类描述")
    parent_id = Column(UUID(as_uuid=True), nullable=True, comment="父分类ID")
    
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    is_active = Column(Boolean, default=True, comment="是否激活")
    
    # 索引
    __table_args__ = (
        Index('idx_categories_parent_id', 'parent_id'),
        Index('idx_categories_name', 'name'),
    )
