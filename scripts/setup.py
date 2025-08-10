"""
项目设置脚本
"""
import os
import sys
import subprocess
from pathlib import Path


def create_directories():
    """创建必要的目录"""
    directories = [
        "data/uploads",
        "data/faiss_index", 
        "logs",
        "models",
        "data/documents"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {directory}")


def setup_environment():
    """设置环境文件"""
    if not os.path.exists(".env"):
        if os.path.exists(".env"):
            import shutil
            shutil.copy(".env", ".env")
            print("✓ 创建 .env 文件")
        else:
            print("⚠ 未找到 .env 文件")


def install_dependencies():
    """安装依赖"""
    try:
        print("正在安装Python依赖...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✓ Python依赖安装完成")
    except subprocess.CalledProcessError as e:
        print(f"✗ 依赖安装失败: {e}")
        return False
    return True


def download_models():
    """下载模型（可选）"""
    print("\n模型下载说明:")
    print("1. Qwen3-8B模型: 请从 https://huggingface.co/Qwen/Qwen2.5-8B-Instruct 下载")
    print("2. Qwen3-Embedding-0.6B模型: 请从 https://huggingface.co/Qwen/Qwen2.5-0.6B-Instruct 下载")
    print("3. 将模型文件放置在 ./models/ 目录下")
    print("4. 或者在首次运行时自动下载（需要网络连接）")


def setup_database():
    """设置数据库"""
    print("\n数据库设置说明:")
    print("1. 确保PostgreSQL已安装并运行")
    print("2. 创建数据库: createdb law_gpt")
    print("3. 或者使用Docker: docker-compose up -d postgres")
    print("4. 数据库表将在首次运行时自动创建")


def main():
    """主设置函数"""
    print("🚀 开始设置法律问答系统...")
    
    # 创建目录
    create_directories()
    
    # 设置环境文件
    setup_environment()
    
    # 安装依赖
    if not install_dependencies():
        return
    
    # 模型下载说明
    download_models()
    
    # 数据库设置说明
    setup_database()
    
    print("\n✅ 项目设置完成!")
    print("\n下一步:")
    print("1. 编辑 .env 文件配置数据库连接")
    print("2. 启动数据库和Redis: docker-compose up -d")
    print("3. 运行应用: python -m app.main")
    print("4. 访问API文档: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
