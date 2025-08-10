#!/usr/bin/env python3
"""
法律问答系统启动脚本
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path




def check_environment():
    """检查环境配置"""
    if not os.path.exists(".env"):
        print("⚠ 未找到 .env 文件")
        if os.path.exists(".env"):
            import shutil
            shutil.copy(".env", ".env")
            print("✓ 已创建 .env 文件，请根据需要修改配置")
        else:
            print("✗ 请创建 .env 配置文件")
            return False

    # 检查必要目录
    directories = ["data/uploads", "data/faiss_index", "logs"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("✓ 环境配置检查完成")
    return True


def start_app(dev_mode=False):
    """启动应用"""
    print("🏛️ 启动法律问答系统...")

    if dev_mode:
        # 开发模式：使用uvicorn的reload功能
        cmd = [
            "uvicorn",
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8001",
            "--reload",
            "--log-level", "info"
        ]
    else:
        # 生产模式
        cmd = [sys.executable, "-m", "app.main"]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 系统已停止")


def load_sample_data():
    """加载示例数据"""
    print("📚 加载示例法律数据...")

    try:
        subprocess.run([sys.executable, "scripts/load_sample_data.py"], check=True)
        print("✓ 示例数据加载完成")
    except subprocess.CalledProcessError as e:
        print(f"✗ 示例数据加载失败: {e}")


def run_tests():
    """运行测试"""
    print("🧪 运行测试...")

    try:
        subprocess.run(["pytest", "tests/", "-v"], check=True)
        print("✓ 所有测试通过")
    except subprocess.CalledProcessError:
        print("✗ 测试失败")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="法律问答系统启动脚本")
    parser.add_argument("--dev", action="store_true", help="开发模式")
    parser.add_argument("--no-services", action="store_true", help="不启动依赖服务")
    parser.add_argument("--load-data", action="store_true", help="加载示例数据")
    parser.add_argument("--test", action="store_true", help="运行测试")
    parser.add_argument("--setup", action="store_true", help="仅进行设置")

    args = parser.parse_args()

    print("🏛️ 法律问答系统启动器\n")


    # 检查环境
    if not check_environment():
        return 1

    # 如果只是设置，则退出
    if args.setup:
        print("✅ 设置完成")
        return 0

    # 运行测试
    if args.test:
        run_tests()
        return 0

    # 加载示例数据
    if args.load_data:
        load_sample_data()

    # 启动应用
    start_app(dev_mode=args.dev)

    return 0


if __name__ == "__main__":
    sys.exit(main())
