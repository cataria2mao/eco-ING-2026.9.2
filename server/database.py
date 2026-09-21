"""PostgreSQL 数据层：用户账号、会话、消息。

连接串通过 .env 的 PG_DSN 配置，例如：
    PG_DSN=postgresql+psycopg://animal_agent:010405@localhost:5432/agent_platform
"""
import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv(override=True)

PG_DSN = os.getenv(
    "PG_DSN",
    "postgresql+psycopg://animal_agent:010405@localhost:5432/agent_platform",
)

engine = create_engine(PG_DSN, pool_pre_ping=True, pool_size=5, max_overflow=10, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(Text, nullable=False)
    display_name = Column(String(64))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    thread_id = Column(String(64), unique=True, nullable=False, index=True)
    title = Column(String(120), nullable=False, default="新会话")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("conversation_id", "message_id", name="uq_messages_conversation_message"),
    )

    id = Column(Integer, primary_key=True)
    conversation_id = Column(
        Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    message_id = Column(String(80), nullable=False)   # LangChain message.id，用于去重
    role = Column(String(16), nullable=False)         # user / assistant / tool / system
    content = Column(Text, nullable=False, default="")
    tool_calls = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


def init_tables():
    """在已存在的数据库中创建业务表（不负责创建数据库本身）。"""
    Base.metadata.create_all(engine)
