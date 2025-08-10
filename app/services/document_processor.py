"""
文档处理服务
支持PDF、DOCX等格式的法律文档处理
"""
import os
import re
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from loguru import logger
try:
    # 尝试使用pypdf (推荐)
    from pypdf import PdfReader
    PDF_READER_CLASS = PdfReader
except ImportError:
    try:
        # 回退到PyPDF2
        from PyPDF2 import PdfReader
        PDF_READER_CLASS = PdfReader
    except ImportError:
        logger.error("无法导入PDF处理库，请安装pypdf或PyPDF2")
        PDF_READER_CLASS = None

from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session
from loguru import logger

from app.config import settings
from app.models.database import Document, DocumentChunk
from app.models.schemas import DocumentCreate
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store
from app.services.llm_service import llm_service


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "！", "？", " ", ""]
        )
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """从PDF文件提取文本"""
        try:
            if PDF_READER_CLASS is None:
                raise ImportError("PDF处理库不可用")

            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PDF_READER_CLASS(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

            return self._clean_text(text)
            
        except Exception as e:
            logger.error(f"PDF文本提取失败 {file_path}: {e}")
            raise
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """从DOCX文件提取文本"""
        try:
            doc = DocxDocument(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return self._clean_text(text)
            
        except Exception as e:
            logger.error(f"DOCX文本提取失败 {file_path}: {e}")
            raise
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """从TXT文件提取文本"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            return self._clean_text(text)
            
        except Exception as e:
            logger.error(f"TXT文本提取失败 {file_path}: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符但保留中文标点
        text = re.sub(r'[^\u4e00-\u9fff\w\s\.,;:!?()[]{}""''《》【】（）、。，；：！？]', '', text)
        return text.strip()
    
    def process_document(self, 
                        file_path: str, 
                        document_data: DocumentCreate,
                        db: Session) -> UUID:
        """
        处理单个文档
        
        Args:
            file_path: 文件路径
            document_data: 文档数据
            db: 数据库会话
            
        Returns:
            文档ID
        """
        try:
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size > settings.max_document_size:
                raise ValueError(f"文件过大: {file_size} bytes")
            
            # 根据文件类型提取文本
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                content = self.extract_text_from_pdf(file_path)
            elif file_ext == '.docx':
                content = self.extract_text_from_docx(file_path)
            elif file_ext == '.txt':
                content = self.extract_text_from_txt(file_path)
            else:
                raise ValueError(f"不支持的文件类型: {file_ext}")
            
            # 生成文档摘要
            summary = llm_service.generate_document_summary(content)
            
            # 创建文档记录
            document = Document(
                title=document_data.title,
                content=content,
                file_path=file_path,
                file_type=file_ext,
                category=document_data.category,
                source=document_data.source,
                summary=summary
            )
            
            db.add(document)
            db.commit()
            db.refresh(document)
            
            # 处理文档分块
            self._process_document_chunks(document, content, db)
            
            logger.info(f"文档处理完成: {document.title} (ID: {document.id})")
            return document.id
            
        except Exception as e:
            logger.error(f"文档处理失败 {file_path}: {e}")
            db.rollback()
            raise
    
    def _process_document_chunks(self, 
                               document: Document, 
                               content: str, 
                               db: Session):
        """处理文档分块"""
        try:
            # 分割文本
            chunks = self.text_splitter.split_text(content)
            
            # 生成嵌入向量
            embeddings = embedding_service.encode_documents(chunks)
            
            # 准备向量存储的数据
            chunk_ids = []
            metadata_list = []
            
            # 保存分块到数据库
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                chunk = DocumentChunk(
                    document_id=document.id,
                    chunk_index=i,
                    content=chunk_text,
                    embedding_vector=embedding.tolist(),  # 转换为JSON格式存储
                    token_count=self._estimate_tokens(chunk_text)
                )
                
                db.add(chunk)
                db.flush()  # 获取ID但不提交
                
                chunk_ids.append(chunk.id)
                metadata_list.append({
                    'chunk_id': str(chunk.id),
                    'document_id': str(document.id),
                    'title': document.title,
                    'category': document.category,
                    'source': document.source,
                    'chunk_index': i
                })
            
            db.commit()
            
            # 添加到向量索引
            vector_store.add_documents(
                embeddings=embeddings,
                document_ids=chunk_ids,
                metadata=metadata_list
            )
            
            # 保存向量索引
            vector_store.save_index()
            
            logger.info(f"文档分块处理完成: {len(chunks)} 个分块")
            
        except Exception as e:
            logger.error(f"文档分块处理失败: {e}")
            raise
    
    def _estimate_tokens(self, text: str) -> int:
        """估算文本的token数量"""
        # 简单估算：中文字符数 + 英文单词数
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        return chinese_chars + english_words
    
    def batch_process_documents(self, 
                              file_paths: List[str], 
                              document_data_list: List[DocumentCreate],
                              db: Session) -> List[UUID]:
        """
        批量处理文档
        
        Args:
            file_paths: 文件路径列表
            document_data_list: 文档数据列表
            db: 数据库会话
            
        Returns:
            处理成功的文档ID列表
        """
        processed_ids = []
        
        for file_path, doc_data in zip(file_paths, document_data_list):
            try:
                doc_id = self.process_document(file_path, doc_data, db)
                processed_ids.append(doc_id)
                logger.info(f"批量处理成功: {file_path}")
                
            except Exception as e:
                logger.error(f"批量处理失败 {file_path}: {e}")
                continue
        
        logger.info(f"批量处理完成: {len(processed_ids)}/{len(file_paths)} 个文档")
        return processed_ids
    
    def reprocess_document(self, document_id: UUID, db: Session):
        """重新处理文档（重新生成嵌入向量）"""
        try:
            # 获取文档
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ValueError(f"文档不存在: {document_id}")
            
            # 删除旧的分块
            db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
            db.commit()
            
            # 重新处理分块
            self._process_document_chunks(document, document.content, db)
            
            logger.info(f"文档重新处理完成: {document_id}")
            
        except Exception as e:
            logger.error(f"文档重新处理失败 {document_id}: {e}")
            raise


# 全局文档处理器实例
document_processor = DocumentProcessor()
