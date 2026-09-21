"""初始化 PostgreSQL：创建应用数据库 + 业务表。

使用前请在 .env 中配置：
    PG_DSN=postgresql+psycopg://animal_agent:010405@localhost:5432/agent_platform
    PG_ADMIN_DSN=postgresql://postgres:<postgres超级用户密码>@localhost:5432/postgres
    PG_DB_NAME=agent_platform
    PG_DB_OWNER=animal_agent

运行：
    python init_db.py
或临时指定管理员连接串：
    python init_db.py --admin-dsn "postgresql://postgres:密码@localhost:5432/postgres"
"""
import argparse
import os
import sys

# Windows 控制台可能是 GBK，避免错误信息中的不可映射字符导致崩溃
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

import psycopg
from psycopg import sql
from dotenv import load_dotenv
from sqlalchemy.engine.url import make_url

load_dotenv(override=True)

PG_DSN = os.getenv(
    "PG_DSN",
    "postgresql+psycopg://animal_agent:010405@localhost:5432/agent_platform",
)
PG_ADMIN_DSN = os.getenv("PG_ADMIN_DSN", "")

_app_url = make_url(PG_DSN)
DB_NAME = os.getenv("PG_DB_NAME") or _app_url.database
DB_OWNER = os.getenv("PG_DB_OWNER") or _app_url.username


def create_database_if_missing(admin_dsn: str) -> bool:
    """用超级用户连接创建数据库（若不存在）。返回是否新建。"""
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,)
        ).fetchone()
        if exists:
            print(f"[DB] 数据库 '{DB_NAME}' 已存在，跳过创建。")
            return False

        conn.execute(
            sql.SQL("CREATE DATABASE {} OWNER {}").format(
                sql.Identifier(DB_NAME), sql.Identifier(DB_OWNER)
            )
        )
        print(f"[DB] 已创建数据库 '{DB_NAME}'，所有者 '{DB_OWNER}'。")
        return True


def setup_checkpoints():
    """创建 LangGraph PostgresSaver 所需的 checkpoint 表（幂等）。"""
    dsn = os.getenv("PG_CHECKPOINT_DSN") or (
        PG_DSN.replace("postgresql+psycopg://", "postgresql://")
              .replace("postgresql+psycopg2://", "postgresql://")
    )
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
    from langgraph.checkpoint.postgres import PostgresSaver

    pool = ConnectionPool(
        conninfo=dsn,
        min_size=1,
        max_size=5,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    pool.open(wait=True, timeout=10)
    try:
        PostgresSaver(pool).setup()
        print("[DB] LangGraph checkpoint 表已就绪（checkpoints / checkpoint_blobs / checkpoint_writes）")
    finally:
        pool.close()


def main():
    parser = argparse.ArgumentParser(description="初始化用户/对话 PostgreSQL 数据库")
    parser.add_argument("--admin-dsn", default=PG_ADMIN_DSN,
                        help="postgres 超级用户连接串，缺省读取 PG_ADMIN_DSN")
    parser.add_argument("--skip-create", action="store_true",
                        help="跳过创建数据库，仅创建表")
    args = parser.parse_args()

    if not args.skip_create:
        if not args.admin_dsn:
            print("[DB] 未提供 PG_ADMIN_DSN，无法创建数据库。")
            print("     请设置 .env 中的 PG_ADMIN_DSN，或执行：")
            print(f'     python init_db.py --admin-dsn "postgresql://postgres:密码@localhost:5432/postgres"')
            print("     （也可先用 psql 手动创建数据库，再运行 python init_db.py --skip-create）")
            sys.exit(2)
        try:
            create_database_if_missing(args.admin_dsn)
        except Exception as e:
            print(f"[DB] 创建数据库失败：{e}")
            sys.exit(1)

    # 创建业务表
    from database import init_tables, engine
    try:
        init_tables()
        print(f"[DB] 业务表已就绪：users / conversations / messages")
        print(f"[DB] 连接：{engine.url.render_as_string(hide_password=True)}")
    except Exception as e:
        print(f"[DB] 创建业务表失败：{e}")
        sys.exit(1)

    # 创建 LangGraph 持久化 checkpoint 表
    try:
        setup_checkpoints()
    except Exception as e:
        print(f"[DB] 创建 checkpoint 表失败：{e}")
        print("     可稍后重试；服务启动时也会自动尝试（失败则回退内存 checkpointer）。")


if __name__ == "__main__":
    main()
