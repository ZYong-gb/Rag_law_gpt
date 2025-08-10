#!/usr/bin/env python3
"""
测试所有RAG实现是否正常工作
"""
import asyncio
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database.connection import db_manager
from app.models.schemas import QueryRequest
from app.services.rag_service import rag_service as langchain_rag
from app.services.llamaindex_rag_service import llamaindex_rag_service
from app.services.milvus_rag_service import milvus_rag_service


async def test_implementation(name: str, rag_service, db):
    """测试单个RAG实现"""
    print(f"\n🧪 测试 {name}...")
    
    try:
        # 创建测试请求
        request = QueryRequest(
            question="民法典的立法目的是什么？",
            user_id="test_user"
        )
        
        # 执行查询
        response = await rag_service.query(request, db)
        
        # 检查响应
        if response and response.answer:
            print(f"✅ {name} 测试成功")
            print(f"   回答长度: {len(response.answer)} 字符")
            print(f"   置信度: {response.confidence_score:.2f}")
            print(f"   响应时间: {response.response_time:.2f}秒")
            print(f"   检索文档: {len(response.retrieved_documents)}个")
            return True
        else:
            print(f"❌ {name} 测试失败: 无有效响应")
            return False
            
    except Exception as e:
        print(f"❌ {name} 测试异常: {e}")
        return False


async def main():
    """主测试函数"""
    print("🏛️ 法律问答系统 - 全实现测试")
    print("=" * 50)
    
    # 初始化数据库
    try:
        db_manager.init_database()
        db_manager.init_redis()
        db = db_manager.get_db_session()
        print("✅ 数据库连接成功")
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return 1
    
    try:
        # 测试所有实现
        implementations = [
            ("LangChain + FAISS", langchain_rag),
            ("LlamaIndex + FAISS", llamaindex_rag_service),
            ("LangChain + Milvus", milvus_rag_service)
        ]
        
        results = {}
        
        for name, service in implementations:
            success = await test_implementation(name, service, db)
            results[name] = success
        
        # 打印总结
        print(f"\n📊 测试总结")
        print("=" * 30)
        
        success_count = 0
        for name, success in results.items():
            status = "✅ 通过" if success else "❌ 失败"
            print(f"{name}: {status}")
            if success:
                success_count += 1
        
        print(f"\n总体结果: {success_count}/{len(implementations)} 个实现通过测试")
        
        if success_count == len(implementations):
            print("🎉 所有RAG实现都正常工作！")
            return 0
        else:
            print("⚠ 部分实现存在问题，请检查配置和依赖")
            return 1
            
    except Exception as e:
        print(f"❌ 测试过程中出现异常: {e}")
        return 1
    
    finally:
        if db:
            db.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
