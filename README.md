# 法律领域问答系统 (Law GPT)

基于 LangChain + Qwen3-Embedding-0.6B + Qwen3-8B + RAG 的法律领域智能问答系统。

## 🏗️ 系统架构

### 技术栈
- **核心框架**: LangChain + FastAPI
- **大语言模型**: Qwen3-8B (通义千问3-8B)
- **嵌入模型**: Qwen3-Embedding-0.6B
- **向量数据库**: FAISS (高性能本地搜索)
- **结构化数据库**: PostgreSQL
- **缓存层**: Redis
- **文档处理**: PyPDF2, python-docx

### 系统特性
- 🔍 **智能检索**: 基于语义相似度的文档检索
- 🧠 **知识增强**: RAG架构结合法律知识库
- ⚡ **高性能**: FAISS向量搜索 + Redis缓存
- 📚 **多格式支持**: PDF、DOCX、TXT文档处理
- 🔄 **实时更新**: 支持动态添加法律文档
- 📊 **完整监控**: 查询日志、性能统计、用户反馈

## 🚀 快速开始

### 方式一：一键启动（推荐）

```bash
# 1. 克隆项目
git clone <repository-url>
cd law-gpt

# 2. 一键启动（包含环境设置、依赖安装、服务启动）
python run.py --dev --load-data

# 3. 访问系统
# API文档: http://localhost:8000/docs
# Web界面: http://localhost:8000/static/index.html
```

### 方式二：Docker部署

```bash
# 1. 启动所有服务
docker-compose up -d

# 2. 加载示例数据
docker-compose exec law-gpt-app python scripts/load_sample_data.py

# 3. 查看日志
docker-compose logs -f law-gpt-app
```

### 方式三：手动部署

```bash
# 1. 环境准备
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# 2. 启动依赖服务
docker-compose up -d postgres redis

# 3. 配置环境
cp .env .env
# 编辑 .env 文件

# 4. 启动应用
python -m app.main

# 5. 加载示例数据
python scripts/load_sample_data.py
```

## 📖 使用方式

### 1. Web界面使用

访问 http://localhost:8000/static/index.html 使用图形界面：

- **法律咨询**: 直接输入问题获得专业回答
- **文档上传**: 上传PDF、DOCX、TXT格式的法律文档
- **实时反馈**: 查看置信度、响应时间、相关文档

### 2. API接口使用

#### 查询法律问题

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "民法典的立法目的是什么？",
    "user_id": "user123",
    "top_k": 5,
    "include_source": true
  }'
```

#### 上传法律文档

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@document.pdf" \
  -F "title=新法律文档" \
  -F "category=民法" \
  -F "source=最高人民法院"
```

#### 获取系统统计

```bash
curl "http://localhost:8000/api/v1/stats"
```

### 3. Python客户端使用

```python
# 使用提供的客户端
python client_example.py

# 或交互模式
python client_example.py interactive
```

### 4. 管理工具使用

```bash
# 查看系统统计
python scripts/manage.py stats

# 列出所有文档
python scripts/manage.py list-docs

# 重建向量索引
python scripts/manage.py rebuild-index

# 清空缓存
python scripts/manage.py clear-cache
```

### 查看完整API文档

访问 http://localhost:8000/docs 查看Swagger API文档。

## 🏛️ 项目结构

```
law-gpt/
├── app/                        # 应用核心代码
│   ├── api/                    # API路由
│   ├── database/               # 数据库连接和配置
│   ├── models/                 # 数据模型
│   ├── services/               # 业务逻辑服务
│   ├── utils/                  # 工具函数
│   ├── config.py              # 配置管理
│   └── main.py                # 应用入口
├── data/                       # 数据文件
│   ├── uploads/               # 上传的文档
│   ├── faiss_index/           # FAISS索引文件
│   └── documents/             # 示例文档
├── tests/                      # 测试文件
├── scripts/                    # 脚本工具
├── logs/                       # 日志文件
├── models/                     # AI模型文件
├── docker-compose.yml          # Docker编排
├── requirements.txt            # Python依赖
└── README.md                   # 项目说明
```

## 🔧 核心服务

### 1. 嵌入服务 (EmbeddingService)
- 基于Qwen3-Embedding-0.6B模型
- 支持批量文档向量化
- 自动维度检测和归一化

### 2. 向量存储 (FAISSVectorStore)
- 高性能FAISS索引
- 支持IVF+PQ压缩
- 持久化存储和加载

### 3. LLM服务 (LLMService)
- Qwen3-8B模型集成
- 专业法律提示词模板
- 支持文档摘要生成

### 4. RAG服务 (RAGService)
- 端到端查询处理
- 智能上下文管理
- 置信度评估

### 5. 缓存服务 (CacheService)
- Redis缓存层
- 查询结果缓存
- 性能统计

## 📊 监控和管理

### 系统统计
```bash
curl http://localhost:8000/api/v1/stats
```

### 健康检查
```bash
curl http://localhost:8000/api/v1/health
```

### 清空缓存
```bash
curl -X POST http://localhost:8000/api/v1/cache/clear
```

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_rag_service.py

# 生成测试覆盖率报告
pytest --cov=app tests/
```

## 🔒 生产部署

### 1. 安全配置
- 修改 `.env` 中的 `SECRET_KEY`
- 配置数据库访问权限
- 设置防火墙规则

### 2. 性能优化
- 使用GPU加速模型推理
- 调整FAISS索引参数
- 配置Redis集群

### 3. 监控告警
- 集成Prometheus监控
- 设置日志告警
- 配置健康检查

## 📝 最佳实践

### 1. 文档管理
- 定期更新法律文档
- 维护文档分类体系
- 监控文档质量

### 2. 性能优化
- 合理设置缓存策略
- 优化向量索引参数
- 监控系统资源使用

### 3. 用户体验
- 收集用户反馈
- 持续优化回答质量
- 提供多样化查询方式

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

MIT License

## 📞 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。
