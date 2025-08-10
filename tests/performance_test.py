"""
性能测试脚本
"""
import asyncio
import time
import statistics
from typing import List, Dict
import httpx
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import pandas as pd


class PerformanceTester:
    """性能测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    async def single_query_test(self, question: str, session: httpx.AsyncClient) -> Dict:
        """单次查询测试"""
        start_time = time.time()
        
        try:
            response = await session.post(
                f"{self.base_url}/api/v1/query",
                json={
                    "question": question,
                    "user_id": "perf_test",
                    "top_k": 5
                },
                timeout=30.0
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "response_time": response_time,
                    "confidence": data.get("confidence_score", 0),
                    "retrieved_docs": len(data.get("retrieved_documents", [])),
                    "answer_length": len(data.get("answer", ""))
                }
            else:
                return {
                    "success": False,
                    "response_time": response_time,
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "response_time": end_time - start_time,
                "error": str(e)
            }
    
    async def concurrent_test(self, questions: List[str], concurrent_users: int = 10):
        """并发测试"""
        print(f"🚀 开始并发测试: {concurrent_users} 个并发用户")
        
        async with httpx.AsyncClient() as session:
            tasks = []
            
            # 创建并发任务
            for i in range(concurrent_users):
                question = questions[i % len(questions)]
                task = self.single_query_test(question, session)
                tasks.append(task)
            
            # 执行并发测试
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # 统计结果
            successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
            failed_results = [r for r in results if isinstance(r, dict) and not r.get("success")]
            
            if successful_results:
                response_times = [r["response_time"] for r in successful_results]
                
                stats = {
                    "concurrent_users": concurrent_users,
                    "total_requests": len(results),
                    "successful_requests": len(successful_results),
                    "failed_requests": len(failed_results),
                    "success_rate": len(successful_results) / len(results) * 100,
                    "total_time": total_time,
                    "requests_per_second": len(results) / total_time,
                    "avg_response_time": statistics.mean(response_times),
                    "min_response_time": min(response_times),
                    "max_response_time": max(response_times),
                    "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
                    "p99_response_time": statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times)
                }
                
                self.results.append(stats)
                return stats
            else:
                return {
                    "concurrent_users": concurrent_users,
                    "total_requests": len(results),
                    "successful_requests": 0,
                    "failed_requests": len(failed_results),
                    "success_rate": 0,
                    "error": "所有请求都失败了"
                }
    
    async def load_test(self, questions: List[str], max_users: int = 50, step: int = 10):
        """负载测试"""
        print(f"📈 开始负载测试: 最大 {max_users} 并发用户")
        
        for concurrent_users in range(step, max_users + 1, step):
            print(f"\n测试 {concurrent_users} 并发用户...")
            
            result = await self.concurrent_test(questions, concurrent_users)
            
            if result:
                print(f"  成功率: {result.get('success_rate', 0):.1f}%")
                print(f"  平均响应时间: {result.get('avg_response_time', 0):.2f}秒")
                print(f"  QPS: {result.get('requests_per_second', 0):.1f}")
                
                # 如果成功率低于80%，停止测试
                if result.get('success_rate', 0) < 80:
                    print("⚠ 成功率过低，停止测试")
                    break
            
            # 等待系统恢复
            await asyncio.sleep(2)
    
    def generate_report(self, output_file: str = "performance_report.html"):
        """生成性能报告"""
        if not self.results:
            print("没有测试结果可生成报告")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(self.results)
        
        # 生成图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 响应时间图
        axes[0, 0].plot(df['concurrent_users'], df['avg_response_time'], 'b-o')
        axes[0, 0].set_title('平均响应时间')
        axes[0, 0].set_xlabel('并发用户数')
        axes[0, 0].set_ylabel('响应时间 (秒)')
        
        # QPS图
        axes[0, 1].plot(df['concurrent_users'], df['requests_per_second'], 'g-o')
        axes[0, 1].set_title('每秒请求数 (QPS)')
        axes[0, 1].set_xlabel('并发用户数')
        axes[0, 1].set_ylabel('QPS')
        
        # 成功率图
        axes[1, 0].plot(df['concurrent_users'], df['success_rate'], 'r-o')
        axes[1, 0].set_title('成功率')
        axes[1, 0].set_xlabel('并发用户数')
        axes[1, 0].set_ylabel('成功率 (%)')
        
        # P95响应时间图
        if 'p95_response_time' in df.columns:
            axes[1, 1].plot(df['concurrent_users'], df['p95_response_time'], 'm-o')
            axes[1, 1].set_title('P95响应时间')
            axes[1, 1].set_xlabel('并发用户数')
            axes[1, 1].set_ylabel('P95响应时间 (秒)')
        
        plt.tight_layout()
        plt.savefig('performance_charts.png', dpi=300, bbox_inches='tight')
        
        # 生成HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>法律问答系统性能测试报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background: #e7f3ff; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>法律问答系统性能测试报告</h1>
            
            <div class="summary">
                <h2>测试摘要</h2>
                <p><strong>测试时间:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>最大并发用户:</strong> {df['concurrent_users'].max()}</p>
                <p><strong>最佳QPS:</strong> {df['requests_per_second'].max():.1f}</p>
                <p><strong>平均成功率:</strong> {df['success_rate'].mean():.1f}%</p>
            </div>
            
            <h2>详细结果</h2>
            {df.to_html(index=False, table_id="results")}
            
            <h2>性能图表</h2>
            <img src="performance_charts.png" alt="性能图表" style="max-width: 100%;">
            
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ 性能报告已生成: {output_file}")


async def main():
    """主测试函数"""
    # 测试问题集
    test_questions = [
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
    
    tester = PerformanceTester()
    
    print("🧪 法律问答系统性能测试")
    print("=" * 50)
    
    # 1. 单次查询测试
    print("\n1. 单次查询测试...")
    async with httpx.AsyncClient() as session:
        result = await tester.single_query_test(test_questions[0], session)
        if result["success"]:
            print(f"   响应时间: {result['response_time']:.2f}秒")
            print(f"   置信度: {result['confidence']:.2f}")
            print(f"   检索文档数: {result['retrieved_docs']}")
        else:
            print(f"   测试失败: {result.get('error', 'Unknown error')}")
    
    # 2. 负载测试
    print("\n2. 负载测试...")
    await tester.load_test(test_questions, max_users=30, step=5)
    
    # 3. 生成报告
    print("\n3. 生成性能报告...")
    tester.generate_report()
    
    print("\n✅ 性能测试完成!")


if __name__ == "__main__":
    asyncio.run(main())
