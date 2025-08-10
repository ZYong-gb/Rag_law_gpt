"""
日志配置
"""
import os
import sys
from loguru import logger
from app.config import settings


def setup_logging():
    """设置日志配置"""
    
    # 移除默认的日志处理器
    logger.remove()
    
    # 确保日志目录存在
    log_dir = os.path.dirname(settings.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # 控制台日志
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True
    )
    
    # 文件日志
    logger.add(
        settings.log_file,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8"
    )
    
    # 错误日志单独文件
    error_log_file = settings.log_file.replace('.log', '_error.log')
    logger.add(
        error_log_file,
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="50 MB",
        retention="60 days",
        compression="zip",
        encoding="utf-8"
    )
    
    logger.info("日志系统初始化完成")
