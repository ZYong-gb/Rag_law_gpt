"""
数据库连接管理
"""
import redis
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from app.config import settings
from app.models.database import Base
from loguru import logger


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self.engine = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None
        self.redis_client = None
    
    def init_database(self):
        """初始化数据库连接"""
        try:
            # 同步数据库引擎
            self.engine = create_engine(
                settings.database_url,
                poolclass=StaticPool,
                pool_pre_ping=True,
                echo=settings.debug
            )
            
            # 异步数据库引擎
            async_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")
            self.async_engine = create_async_engine(
                async_url,
                pool_pre_ping=True,
                echo=settings.debug
            )
            
            # 会话工厂
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            self.AsyncSessionLocal = sessionmaker(
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
                bind=self.async_engine
            )
            
            # 创建表
            Base.metadata.create_all(bind=self.engine)
            logger.info("数据库初始化成功")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def init_redis(self):
        """初始化Redis连接"""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # 测试连接
            self.redis_client.ping()
            logger.info("Redis连接初始化成功")
            
        except Exception as e:
            logger.error(f"Redis连接初始化失败: {e}")
            raise
    
    def get_db_session(self) -> Session:
        """获取数据库会话"""
        return self.SessionLocal()
    
    def get_async_db_session(self) -> AsyncSession:
        """获取异步数据库会话"""
        return self.AsyncSessionLocal()
    
    def get_redis_client(self) -> redis.Redis:
        """获取Redis客户端"""
        return self.redis_client
    
    def close_connections(self):
        """关闭所有连接"""
        if self.engine:
            self.engine.dispose()
        if self.async_engine:
            self.async_engine.dispose()
        if self.redis_client:
            self.redis_client.close()
        logger.info("数据库连接已关闭")


# 全局数据库管理器实例
db_manager = DatabaseManager()


def get_database():
    """依赖注入：获取数据库会话"""
    db = db_manager.get_db_session()
    try:
        yield db
    finally:
        db.close()


async def get_async_database():
    """依赖注入：获取异步数据库会话"""
    async with db_manager.get_async_db_session() as session:
        yield session


def get_redis():
    """依赖注入：获取Redis客户端"""
    return db_manager.get_redis_client()


# 数据库健康检查
def check_database_health() -> bool:
    """检查数据库连接健康状态"""
    try:
        with db_manager.get_db_session() as db:
            db.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"数据库健康检查失败: {e}")
        return False


def check_redis_health() -> bool:
    """检查Redis连接健康状态"""
    try:
        redis_client = db_manager.get_redis_client()
        redis_client.ping()
        return True
    except Exception as e:
        logger.error(f"Redis健康检查失败: {e}")
        return False
