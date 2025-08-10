"""
RAG实现基准测试
比较LangChain、LlamaIndex和Milvus的性能
"""
import asyncio
import time
import statistics
import json
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.connection import db_manager
from app.models.schemas import QueryRequest
from app.services.rag_service import rag_service as langchain_rag
from app.services.llamaindex_rag_service import llamaindex_rag_service
from app.services.milvus_rag_service import milvus_rag_service


class RAGBenchmark:
    """RAG基准测试"""
    
    def __init__(self):
        self.test_questions = [
            "民法典的立法目的是什么？",
            "合同的基本原则有哪些？",
            "劳动合同应当遵循什么原则？",
            "什么是诚信原则？",
            "如何解除劳动合同？",
            "民事主体包括哪些？",
            "合同成立的条件是什么？",
            "什么是违约责任？",
            "如何保护消费者权益？",
            "知识产权包括哪些内容？"
        ]
        self.results = {}
    
    async def benchmark_implementation(self, 
                                     name: str, 
                                     rag_service, 
                                     db,
                                     iterations: int = 3) -> Dict[str, Any]:
        """测试单个RAG实现"""
        print(f"\n🧪 测试 {name}...")
        
        response_times = []
        confidence_scores = []
        retrieved_docs_counts = []
        answer_lengths = []
        errors = 0
        
        for i, question in enumerate(self.test_questions):
            print(f"  问题 {i+1}/{len(self.test_questions)}: {question[:30]}...")
            
            question_times = []
            question_confidences = []
            question_docs = []
            question_lengths = []
            
            # 每个问题测试多次取平均值
            for iteration in range(iterations):
                try:
                    request = QueryRequest(
                        question=question,
                        user_id=f"benchmark_{name}_{i}_{iteration}"
                    )
                    
                    start_time = time.time()
                    response = await rag_service.query(request, db)
                    end_time = time.time()
                    
                    question_times.append(end_time - start_time)
                    question_confidences.append(response.confidence_score)
                    question_docs.append(len(response.retrieved_documents))
                    question_lengths.append(len(response.answer))
                    
                except Exception as e:
                    print(f"    错误: {e}")
                    errors += 1
                    continue
            
            # 计算平均值
            if question_times:
                response_times.append(statistics.mean(question_times))
                confidence_scores.append(statistics.mean(question_confidences))
                retrieved_docs_counts.append(statistics.mean(question_docs))
                answer_lengths.append(statistics.mean(question_lengths))
        
        # 计算总体统计
        if response_times:
            stats = {
                "implementation": name,
                "avg_response_time": statistics.mean(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
                "avg_confidence": statistics.mean(confidence_scores),
                "avg_retrieved_docs": statistics.mean(retrieved_docs_counts),
                "avg_answer_length": statistics.mean(answer_lengths),
                "success_rate": (len(self.test_questions) * iterations - errors) / (len(self.test_questions) * iterations) * 100,
                "total_errors": errors
            }
        else:
            stats = {
                "implementation": name,
                "error": "所有测试都失败了"
            }
        
        return stats
    
    async def run_full_benchmark(self, iterations: int = 3):
        """运行完整基准测试"""
        print("🏁 开始RAG实现基准测试")
        print("=" * 50)
        
        # 初始化数据库
        db_manager.init_database()
        db = db_manager.get_db_session()
        
        try:
            # 测试所有实现
            implementations = [
                ("LangChain + FAISS", langchain_rag),
                ("LlamaIndex + FAISS", llamaindex_rag_service),
                ("LangChain + Milvus", milvus_rag_service)
            ]
            
            for name, service in implementations:
                try:
                    result = await self.benchmark_implementation(name, service, db, iterations)
                    self.results[name] = result
                    
                    if "error" not in result:
                        print(f"✅ {name} 测试完成:")
                        print(f"   平均响应时间: {result['avg_response_time']:.2f}秒")
                        print(f"   平均置信度: {result['avg_confidence']:.2f}")
                        print(f"   成功率: {result['success_rate']:.1f}%")
                    else:
                        print(f"❌ {name} 测试失败: {result['error']}")
                        
                except Exception as e:
                    print(f"❌ {name} 测试异常: {e}")
                    self.results[name] = {"error": str(e)}
        
        finally:
            db.close()
    
    def generate_report(self, output_file: str = "rag_benchmark_report.html"):
        """生成基准测试报告"""
        if not self.results:
            print("没有测试结果可生成报告")
            return
        
        # 过滤成功的结果
        successful_results = {k: v for k, v in self.results.items() if "error" not in v}
        
        if not successful_results:
            print("没有成功的测试结果")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(successful_results).T
        
        # 生成图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 响应时间对比
        implementations = list(successful_results.keys())
        response_times = [successful_results[impl]['avg_response_time'] for impl in implementations]
        
        axes[0, 0].bar(implementations, response_times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 0].set_title('平均响应时间对比')
        axes[0, 0].set_ylabel('响应时间 (秒)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 置信度对比
        confidences = [successful_results[impl]['avg_confidence'] for impl in implementations]
        axes[0, 1].bar(implementations, confidences, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 1].set_title('平均置信度对比')
        axes[0, 1].set_ylabel('置信度')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 成功率对比
        success_rates = [successful_results[impl]['success_rate'] for impl in implementations]
        axes[1, 0].bar(implementations, success_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1, 0].set_title('成功率对比')
        axes[1, 0].set_ylabel('成功率 (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 检索文档数对比
        doc_counts = [successful_results[impl]['avg_retrieved_docs'] for impl in implementations]
        axes[1, 1].bar(implementations, doc_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1, 1].set_title('平均检索文档数对比')
        axes[1, 1].set_ylabel('文档数')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('rag_benchmark_charts.png', dpi=300, bbox_inches='tight')
        
        # 生成HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG实现基准测试报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background: #e7f3ff; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .winner {{ background-color: #d4edda; }}
                .error {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <h1>RAG实现基准测试报告</h1>
            
            <div class="summary">
                <h2>测试摘要</h2>
                <p><strong>测试时间:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>测试问题数:</strong> {len(self.test_questions)}</p>
                <p><strong>每问题迭代次数:</strong> 3</p>
                <p><strong>测试实现数:</strong> {len(self.results)}</p>
            </div>
            
            <h2>性能对比</h2>
            {df.to_html(classes="table", table_id="benchmark-results")}
            
            <h2>详细结果</h2>
            <pre>{json.dumps(self.results, indent=2, ensure_ascii=False)}</pre>
            
            <h2>性能图表</h2>
            <img src="rag_benchmark_charts.png" alt="性能图表" style="max-width: 100%;">
            
            <h2>推荐建议</h2>
            <div class="summary">
                <h3>最佳性能: {min(successful_results.keys(), key=lambda x: successful_results[x]['avg_response_time'])}</h3>
                <h3>最高置信度: {max(successful_results.keys(), key=lambda x: successful_results[x]['avg_confidence'])}</h3>
                <h3>最高成功率: {max(successful_results.keys(), key=lambda x: successful_results[x]['success_rate'])}</h3>
            </div>
            
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ 基准测试报告已生成: {output_file}")
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n📊 基准测试摘要")
        print("=" * 50)
        
        for name, result in self.results.items():
            if "error" in result:
                print(f"❌ {name}: {result['error']}")
            else:
                print(f"✅ {name}:")
                print(f"   响应时间: {result['avg_response_time']:.2f}秒")
                print(f"   置信度: {result['avg_confidence']:.2f}")
                print(f"   成功率: {result['success_rate']:.1f}%")
                print(f"   检索文档: {result['avg_retrieved_docs']:.1f}个")


async def main():
    """主函数"""
    benchmark = RAGBenchmark()
    
    try:
        # 运行基准测试
        await benchmark.run_full_benchmark(iterations=3)
        
        # 打印摘要
        benchmark.print_summary()
        
        # 生成报告
        benchmark.generate_report()
        
    except Exception as e:
        print(f"基准测试失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())
