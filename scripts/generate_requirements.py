#!/usr/bin/env python3
"""
生成精确的requirements文件
"""
import subprocess
import sys

def get_installed_version(package_name):
    """获取已安装包的版本"""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "show", package_name], 
                              capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split(':')[1].strip()
    except:
        return None

# 项目核心包列表
core_packages = [
    'fastapi', 'uvicorn', 'pydantic', 'pydantic-settings',
    'langchain', 'langchain-community', 'langchain-core',
    'llama-index', 'llama-index-core', 'transformers', 
    'sentence-transformers', 'accelerate', 'torch', 'torchvision', 'torchaudio',
    'pymilvus', 'numpy', 'xformers', 'faiss-cpu',
    'psycopg2-binary', 'sqlalchemy', 'alembic', 'redis', 'hiredis',
    'pypdf', 'python-docx', 'python-multipart',
    'python-dotenv', 'loguru', 'httpx', 'requests', 'aiofiles',
    'pytest', 'pytest-asyncio', 'black', 'isort', 'flake8',
    'prometheus-client', 'beautifulsoup4', 'scikit-learn', 'pandas', 'nltk'
]

print("# 法律问答系统依赖文件")
print("# 生成时间:", subprocess.run(['date'], capture_output=True, text=True).stdout.strip())
print()

for package in core_packages:
    version = get_installed_version(package)
    if version:
        print(f"{package}=={version}")
    else:
        print(f"# {package} - 未安装")