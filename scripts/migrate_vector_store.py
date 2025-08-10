"""
向量存储迁移脚本
支持在FAISS和Milvus之间迁移数据
"""
import os
import sys
import argparse
import asyncio
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from app.database.connection import db_manager
from app.models.database import Document, DocumentChunk
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store as faiss_store
from app.services.milvus_vector_store import milvus_vector_store
from app.services.llamaindex_rag_service import llamaindex_rag_service
from loguru import logger


class VectorStoreMigrator:
    """向量存储迁移器"""
    
    def __init__(self):
        self.db = None
    
    def init_database(self):
        """初始化数据库连接"""
        try:
            db_manager.init_database()
            self.db = db_manager.get_db_session()
            logger.info("数据库连接初始化成功")
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def migrate_faiss_to_milvus(self):
        """从FAISS迁移到Milvus"""
        try:
            logger.info("开始从FAISS迁移到Milvus...")
            
            # 1. 清空Milvus集合
            milvus_vector_store.clear_collection()
            
            # 2. 从数据库获取所有文档分块
            chunks = self.db.query(DocumentChunk).join(Document).filter(
                Document.is_active == True
            ).all()
            
            if not chunks:
                logger.warning("没有找到文档分块")
                return
            
            logger.info(f"找到 {len(chunks)} 个文档分块")
            
            # 3. 批量处理并迁移
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                # 提取文本内容
                texts = [chunk.content for chunk in batch_chunks]
                
                # 重新生成嵌入向量（确保一致性）
                embeddings = embedding_service.encode_documents(texts)
                
                # 准备元数据
                chunk_ids = [chunk.id for chunk in batch_chunks]
                metadata_list = []
                
                for chunk in batch_chunks:
                    document = self.db.query(Document).filter(
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
                
                logger.info(f"已迁移 {min(i + batch_size, len(chunks))}/{len(chunks)} 个分块")
            
            logger.info("✅ FAISS到Milvus迁移完成")
            
        except Exception as e:
            logger.error(f"FAISS到Milvus迁移失败: {e}")
            raise
    
    def migrate_milvus_to_faiss(self):
        """从Milvus迁移到FAISS"""
        try:
            logger.info("开始从Milvus迁移到FAISS...")
            
            # 1. 清空FAISS索引
            faiss_store.clear_index()
            
            # 2. 从数据库获取所有文档分块
            chunks = self.db.query(DocumentChunk).join(Document).filter(
                Document.is_active == True
            ).all()
            
            if not chunks:
                logger.warning("没有找到文档分块")
                return
            
            logger.info(f"找到 {len(chunks)} 个文档分块")
            
            # 3. 批量处理并迁移
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                # 提取文本内容
                texts = [chunk.content for chunk in batch_chunks]
                
                # 重新生成嵌入向量
                embeddings = embedding_service.encode_documents(texts)
                
                # 准备元数据
                chunk_ids = [chunk.id for chunk in batch_chunks]
                metadata_list = []
                
                for chunk in batch_chunks:
                    document = self.db.query(Document).filter(
                        Document.id == chunk.document_id
                    ).first()
                    
                    metadata_list.append({
                        'chunk_id': str(chunk.id),
                        'document_id': str(chunk.document_id),
                        'title': document.title if document else 'Unknown',
                        'category': document.category if document else None,
                        'source': document.source if document else None,
                        'chunk_index': chunk.chunk_index
                    })
                
                # 添加到FAISS
                faiss_store.add_documents(
                    embeddings=embeddings,
                    document_ids=chunk_ids,
                    metadata=metadata_list
                )
                
                logger.info(f"已迁移 {min(i + batch_size, len(chunks))}/{len(chunks)} 个分块")
            
            # 4. 保存FAISS索引
            faiss_store.save_index()
            
            logger.info("✅ Milvus到FAISS迁移完成")
            
        except Exception as e:
            logger.error(f"Milvus到FAISS迁移失败: {e}")
            raise
    
    def migrate_to_llamaindex(self):
        """迁移数据到LlamaIndex"""
        try:
            logger.info("开始迁移数据到LlamaIndex...")
            
            # 从数据库加载文档到LlamaIndex
            llamaindex_rag_service.load_documents_from_database(self.db)
            
            logger.info("✅ 数据迁移到LlamaIndex完成")
            
        except Exception as e:
            logger.error(f"迁移到LlamaIndex失败: {e}")
            raise
    
    def verify_migration(self, source: str, target: str):
        """验证迁移结果"""
        try:
            logger.info(f"验证从 {source} 到 {target} 的迁移结果...")
            
            # 获取源和目标的统计信息
            if source == "faiss":
                source_stats = faiss_store.get_stats()
            elif source == "milvus":
                source_stats = milvus_vector_store.get_stats()
            
            if target == "faiss":
                target_stats = faiss_store.get_stats()
            elif target == "milvus":
                target_stats = milvus_vector_store.get_stats()
            elif target == "llamaindex":
                target_stats = llamaindex_rag_service.get_index_stats()
            
            logger.info(f"源统计: {source_stats}")
            logger.info(f"目标统计: {target_stats}")
            
            # 简单验证：检查向量数量
            source_count = source_stats.get("total_vectors", 0)
            target_count = target_stats.get("total_vectors", 0)
            
            if abs(source_count - target_count) <= 1:  # 允许1个向量的差异
                logger.info("✅ 迁移验证通过")
                return True
            else:
                logger.warning(f"⚠ 迁移验证失败: 源({source_count}) vs 目标({target_count})")
                return False
                
        except Exception as e:
            logger.error(f"迁移验证失败: {e}")
            return False
    
    def close(self):
        """关闭连接"""
        if self.db:
            self.db.close()


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="向量存储迁移工具")
    parser.add_argument("--from", dest="source", required=True, 
                       choices=["faiss", "milvus"], help="源向量存储")
    parser.add_argument("--to", dest="target", required=True,
                       choices=["faiss", "milvus", "llamaindex"], help="目标向量存储")
    parser.add_argument("--verify", action="store_true", help="验证迁移结果")
    
    args = parser.parse_args()
    
    if args.source == args.target:
        print("❌ 源和目标不能相同")
        return 1
    
    migrator = VectorStoreMigrator()
    migrator.init_database()
    
    try:
        print(f"🔄 开始迁移: {args.source} -> {args.target}")
        
        if args.source == "faiss" and args.target == "milvus":
            migrator.migrate_faiss_to_milvus()
        elif args.source == "milvus" and args.target == "faiss":
            migrator.migrate_milvus_to_faiss()
        elif args.target == "llamaindex":
            migrator.migrate_to_llamaindex()
        else:
            print(f"❌ 不支持的迁移路径: {args.source} -> {args.target}")
            return 1
        
        # 验证迁移结果
        if args.verify:
            success = migrator.verify_migration(args.source, args.target)
            if not success:
                print("⚠ 迁移验证失败，请检查数据完整性")
                return 1
        
        print("✅ 迁移完成！")
        print(f"请更新配置文件以使用新的向量存储: {args.target}")
        
    finally:
        migrator.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
