"""
系统管理脚本
"""
import os
import sys
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from app.database.connection import db_manager
from app.models.database import Document, DocumentChunk, QueryLog
from app.services.vector_store import vector_store
from app.services.cache_service import cache_service
from app.services.embedding_service import embedding_service


class SystemManager:
    """系统管理器"""
    
    def __init__(self):
        self.db = None
    
    def init_database(self):
        """初始化数据库连接"""
        try:
            db_manager.init_database()
            db_manager.init_redis()
            self.db = db_manager.get_db_session()
            print("✓ 数据库连接初始化成功")
        except Exception as e:
            print(f"✗ 数据库连接失败: {e}")
            sys.exit(1)
    
    def show_stats(self):
        """显示系统统计信息"""
        try:
            # 文档统计
            total_docs = self.db.query(Document).count()
            active_docs = self.db.query(Document).filter(Document.is_active == True).count()
            
            # 分块统计
            total_chunks = self.db.query(DocumentChunk).count()
            
            # 查询统计
            total_queries = self.db.query(QueryLog).count()
            
            # 向量索引统计
            vector_stats = vector_store.get_stats()
            
            print("\n📊 系统统计信息:")
            print(f"   文档总数: {total_docs} (活跃: {active_docs})")
            print(f"   文档分块: {total_chunks}")
            print(f"   查询总数: {total_queries}")
            print(f"   向量索引: {vector_stats.get('total_vectors', 0)} 个向量")
            print(f"   索引类型: {vector_stats.get('index_type', 'Unknown')}")
            
            # 分类统计
            categories = self.db.query(Document.category).filter(
                Document.category.isnot(None),
                Document.is_active == True
            ).distinct().all()
            
            if categories:
                print(f"   法律分类: {', '.join([cat[0] for cat in categories])}")
            
        except Exception as e:
            print(f"✗ 获取统计信息失败: {e}")
    
    def list_documents(self, limit: int = 20):
        """列出文档"""
        try:
            documents = self.db.query(Document).filter(
                Document.is_active == True
            ).limit(limit).all()
            
            print(f"\n📚 文档列表 (显示前{limit}个):")
            print("-" * 80)
            
            for doc in documents:
                print(f"ID: {doc.id}")
                print(f"标题: {doc.title}")
                print(f"分类: {doc.category or '未分类'}")
                print(f"来源: {doc.source or '未知'}")
                print(f"创建时间: {doc.created_at}")
                print("-" * 80)
            
        except Exception as e:
            print(f"✗ 获取文档列表失败: {e}")
    
    def delete_document(self, document_id: str):
        """删除文档"""
        try:
            from uuid import UUID
            doc_uuid = UUID(document_id)
            
            document = self.db.query(Document).filter(Document.id == doc_uuid).first()
            
            if not document:
                print(f"✗ 文档不存在: {document_id}")
                return
            
            # 软删除
            document.is_active = False
            self.db.commit()
            
            print(f"✓ 文档已删除: {document.title}")
            
        except Exception as e:
            print(f"✗ 删除文档失败: {e}")
    
    async def clear_cache(self, pattern: str = None):
        """清空缓存"""
        try:
            success = await cache_service.clear_cache(pattern)
            if success:
                print(f"✓ 缓存已清空 (模式: {pattern or 'ALL'})")
            else:
                print("✗ 缓存清空失败")
        except Exception as e:
            print(f"✗ 缓存清空失败: {e}")
    
    def rebuild_vector_index(self):
        """重建向量索引"""
        try:
            print("🔄 开始重建向量索引...")
            
            # 清空现有索引
            vector_store.clear_index()
            
            # 获取所有活跃文档的分块
            chunks = self.db.query(DocumentChunk).join(Document).filter(
                Document.is_active == True
            ).all()
            
            if not chunks:
                print("⚠ 没有找到文档分块")
                return
            
            print(f"找到 {len(chunks)} 个文档分块")
            
            # 重新生成嵌入向量并添加到索引
            
            batch_size = 50
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
                
                # 添加到向量索引
                vector_store.add_documents(
                    embeddings=embeddings,
                    document_ids=chunk_ids,
                    metadata=metadata_list
                )
                
                print(f"已处理 {min(i + batch_size, len(chunks))}/{len(chunks)} 个分块")
            
            # 保存索引
            vector_store.save_index()
            
            print("✅ 向量索引重建完成")
            
        except Exception as e:
            print(f"✗ 向量索引重建失败: {e}")
    
    def export_data(self, output_file: str):
        """导出数据"""
        try:
            import json
            
            # 导出文档数据
            documents = self.db.query(Document).filter(Document.is_active == True).all()
            
            export_data = {
                "documents": [],
                "export_time": str(datetime.utcnow()),
                "total_count": len(documents)
            }
            
            for doc in documents:
                export_data["documents"].append({
                    "id": str(doc.id),
                    "title": doc.title,
                    "content": doc.content,
                    "category": doc.category,
                    "source": doc.source,
                    "summary": doc.summary,
                    "created_at": str(doc.created_at)
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"✓ 数据已导出到: {output_file}")
            
        except Exception as e:
            print(f"✗ 数据导出失败: {e}")
    
    def cleanup_old_logs(self, days: int = 30):
        """清理旧日志"""
        try:
            from datetime import datetime, timedelta
            import glob
            
            cutoff_date = datetime.now() - timedelta(days=days)
            log_pattern = "logs/*.log*"
            
            deleted_count = 0
            for log_file in glob.glob(log_pattern):
                file_time = datetime.fromtimestamp(os.path.getmtime(log_file))
                if file_time < cutoff_date:
                    os.remove(log_file)
                    deleted_count += 1
                    print(f"删除旧日志: {log_file}")
            
            print(f"✓ 清理了 {deleted_count} 个旧日志文件")
            
        except Exception as e:
            print(f"✗ 日志清理失败: {e}")
    
    def close(self):
        """关闭连接"""
        if self.db:
            self.db.close()


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="法律问答系统管理工具")
    parser.add_argument("command", choices=[
        "stats", "list-docs", "delete-doc", "clear-cache", 
        "rebuild-index", "export", "cleanup-logs"
    ], help="管理命令")
    
    parser.add_argument("--doc-id", help="文档ID (用于delete-doc)")
    parser.add_argument("--cache-pattern", help="缓存模式 (用于clear-cache)")
    parser.add_argument("--output", help="输出文件 (用于export)")
    parser.add_argument("--days", type=int, default=30, help="天数 (用于cleanup-logs)")
    parser.add_argument("--limit", type=int, default=20, help="限制数量 (用于list-docs)")
    
    args = parser.parse_args()
    
    manager = SystemManager()
    manager.init_database()
    
    try:
        if args.command == "stats":
            manager.show_stats()
            
        elif args.command == "list-docs":
            manager.list_documents(args.limit)
            
        elif args.command == "delete-doc":
            if not args.doc_id:
                print("✗ 请提供文档ID: --doc-id <uuid>")
                return 1
            manager.delete_document(args.doc_id)
            
        elif args.command == "clear-cache":
            await manager.clear_cache(args.cache_pattern)
            
        elif args.command == "rebuild-index":
            manager.rebuild_vector_index()
            
        elif args.command == "export":
            output_file = args.output or f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            manager.export_data(output_file)

        elif args.command == "cleanup-logs":
            manager.cleanup_old_logs(args.days)

    finally:
        manager.close()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
