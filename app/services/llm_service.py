"""
大语言模型服务
基于VLLM独立部署的Qwen3-8B模型
"""
import json
import time
import requests
from typing import List, Dict, Any, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from loguru import logger
from app.config import settings, ModelConfig, PromptTemplates


class VLLMClient(LLM):
    """VLLM服务客户端包装器"""

    service_url: str = ""
    model_path: str = ""
    timeout: int = 60

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vllm_config = ModelConfig.get_vllm_config()
        self.service_url = vllm_config["service_url"]
        self.model_path = vllm_config["model_path"]
        self._test_connection()

    def _test_connection(self):
        """测试VLLM服务连接"""
        try:
            response = requests.get(
                f"{self.service_url}/health",
                timeout=5
            )
            if response.status_code == 200:
                logger.info(f"VLLM服务连接成功: {self.service_url}")
            else:
                logger.warning(f"VLLM服务响应异常: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"VLLM服务连接测试失败: {e}")
            logger.info("将在实际调用时重试连接")
    
    @property
    def _llm_type(self) -> str:
        return "vllm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """调用VLLM服务生成回答"""
        try:
            vllm_config = ModelConfig.get_vllm_config()
            # 构建请求数据
            request_data = {
                "model": self.model_path,
                "prompt": prompt,
                "max_tokens": vllm_config["max_tokens"],
                "temperature": vllm_config["temperature"],
                "top_p": 0.8,
                "stop": stop if stop else []
            }

            # 发送HTTP请求
            response = requests.post(
                f"{self.service_url}/v1/completions",
                headers={"Content-Type": "application/json"},
                json=request_data,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    generated_text = result["choices"][0]["text"]
                    return generated_text.strip()
                else:
                    logger.error(f"VLLM响应格式异常: {result}")
                    return "抱歉，服务响应格式异常。"
            else:
                logger.error(f"VLLM服务请求失败: {response.status_code}, {response.text}")
                return "抱歉，服务请求失败。"

        except requests.exceptions.Timeout:
            logger.error("VLLM服务请求超时")
            return "抱歉，服务请求超时，请稍后重试。"
        except requests.exceptions.RequestException as e:
            logger.error(f"VLLM服务请求异常: {e}")
            return "抱歉，服务连接异常。"
        except Exception as e:
            logger.error(f"调用VLLM服务失败: {e}")
            return "抱歉，生成回答时出现错误。"


class LLMService:
    """大语言模型服务"""

    def __init__(self):
        self.llm = VLLMClient()
    
    def generate_legal_answer(self, 
                            question: str, 
                            context_documents: List[Dict[str, Any]]) -> str:
        """
        生成法律问题的回答
        
        Args:
            question: 用户问题
            context_documents: 检索到的相关文档
            
        Returns:
            生成的回答
        """
        try:
            # 构建上下文
            context = self._build_context(context_documents)
            
            # 构建提示词
            prompt = PromptTemplates.LEGAL_QA_TEMPLATE.format(
                context=context,
                question=question
            )
            
            # 生成回答
            answer = self.llm(prompt)
            
            return answer
            
        except Exception as e:
            logger.error(f"生成法律回答失败: {e}")
            return "抱歉，无法生成回答。请稍后重试。"
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """构建上下文字符串"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', '未知文档')
            content = doc.get('content', '')
            source = doc.get('source', '')
            
            context_part = f"""
文档 {i}: {title}
来源: {source}
内容: {content}
---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def generate_document_summary(self, content: str) -> str:
        """
        生成文档摘要
        
        Args:
            content: 文档内容
            
        Returns:
            文档摘要
        """
        try:
            prompt = PromptTemplates.DOCUMENT_SUMMARY_TEMPLATE.format(
                content=content[:2000]  # 限制输入长度
            )
            
            summary = self.llm(prompt)
            return summary
            
        except Exception as e:
            logger.error(f"生成文档摘要失败: {e}")
            return "无法生成摘要"
    
    def extract_legal_entities(self, text: str) -> List[str]:
        """
        提取法律实体（法条、案例等）
        
        Args:
            text: 输入文本
            
        Returns:
            提取的法律实体列表
        """
        try:
            prompt = f"""
请从以下文本中提取法律相关的实体，包括：
1. 法律条文
2. 案例名称
3. 法律概念
4. 相关法规

文本：{text}

请以列表形式返回，每行一个实体：
"""
            
            result = self.llm(prompt)
            
            # 解析结果
            entities = []
            for line in result.split('\n'):
                line = line.strip()
                if line and not line.startswith('请') and not line.startswith('文本'):
                    # 移除序号和特殊字符
                    entity = line.lstrip('0123456789.- ').strip()
                    if entity:
                        entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"提取法律实体失败: {e}")
            return []

    def test_connection(self) -> Dict[str, Any]:
        """
        测试VLLM服务连接

        Returns:
            测试结果字典
        """
        try:
            test_prompt = "你好，请简单介绍一下法律咨询的重要性"
            start_time = time.time()

            response = self.llm(test_prompt)
            response_time = time.time() - start_time

            vllm_config = ModelConfig.get_vllm_config()
            return {
                "status": "success",
                "response": response,
                "response_time": response_time,
                "service_url": vllm_config["service_url"],
                "model_path": vllm_config["model_path"]
            }

        except Exception as e:
            vllm_config = ModelConfig.get_vllm_config()
            return {
                "status": "error",
                "error": str(e),
                "service_url": vllm_config["service_url"],
                "model_path": vllm_config["model_path"]
            }


# 全局LLM服务实例
llm_service = LLMService()
