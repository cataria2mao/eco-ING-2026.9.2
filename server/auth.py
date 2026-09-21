"""注册 / 登录 / 令牌校验。密码使用 bcrypt 哈希，令牌使用 PyJWT。"""
import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from database import SessionLocal, User

load_dotenv(override=True)

AUTH_SECRET = os.getenv("AUTH_SECRET", "dev-insecure-change-me")
AUTH_ALGORITHM = "HS256"
TOKEN_TTL_DAYS = int(os.getenv("TOKEN_TTL_DAYS", "30"))

_bearer = HTTPBearer(auto_error=False)


# ---------- 密码 ----------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


# ---------- 令牌 ----------
def create_token(user: User) -> str:
    payload = {
        "sub": str(user.id),
        "username": user.username,
        "exp": datetime.now(tz=timezone.utc) + timedelta(days=TOKEN_TTL_DAYS),
        "iat": datetime.now(tz=timezone.utc),
    }
    return jwt.encode(payload, AUTH_SECRET, algorithm=AUTH_ALGORITHM)


def decode_token(token: str) -> dict:
    """解码并校验令牌；失败返回空 dict。"""
    if not token:
        return {}
    try:
        return jwt.decode(token, AUTH_SECRET, algorithms=[AUTH_ALGORITHM])
    except jwt.PyJWTError:
        return {}


# ---------- FastAPI 依赖 ----------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录")
    payload = decode_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录已失效")
    user = db.get(User, int(user_id))
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在")
    return user


def authenticate_ws_token(token: str):
    """WebSocket 使用：返回 (User, Session) 或 (None, Session)。"""
    db = SessionLocal()
    payload = decode_token(token)
    user_id = payload.get("sub")
    if not user_id:
        return None, db
    user = db.get(User, int(user_id))
    return user, db
