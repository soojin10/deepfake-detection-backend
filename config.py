from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum
from typing import List

class ServerType(Enum):
    MAIN = "main"
    AI = "ai"
    EDGE = "edge"

class Settings(BaseSettings):
    # 공통 설정
    PROJECT_NAME: str = "Deepfake Detection"
    API_V1_STR: str = "/api/v1"
    SERVER_TYPE: str
    
    # 각 서버 엔드포인트 설정
    MAIN_SERVER_HOST: str
    MAIN_SERVER_PORT: int
    AI_SERVER_HOST: str
    AI_SERVER_PORT: int
    EDGE_SERVER_HOST: str
    EDGE_SERVER_PORT: int

    # 서버 간 통신을 위한 URL 설정
    MAIN_SERVER_URL: str
    AI_SERVER_URL: str
    SERVER_URL: str
    
    # WebSocket URL 설정
    MAIN_WS_URL: str
    AI_WS_URL: str
    EDGE_WS_URL: str
    
    # DB 설정 - 스키마 분리
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str
    MAIN_SCHEMA: str
    AI_SCHEMA: str
    
    # 데이터베이스 연결 문자열 (스키마 구분)
    DATABASE_URL: str
    
    # 데이터베이스 풀 설정
    DB_POOL_SIZE: int
    DB_MAX_OVERFLOW: int
    DB_POOL_RECYCLE: int
    
    # AI 서버 특정 설정
    MODEL_PATH: str = "models/deepfake_detection_model.h5"
    
    # 엣지 서버 특정 설정
    EDGE_BUFFER_SIZE: int
    EDGE_SYNC_INTERVAL: int
    
    # CORS 설정
    CORS_ORIGINS: List[str]
    
    # 기존 설정 유지
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024
    ALLOWED_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".jpg", ".jpeg", ".png"]
    
    # Naver Cloud SMS Settings
    NCP_ACCESS_KEY: str
    NCP_SECRET_KEY: str
    NCP_SERVICE_ID: str
    NCP_SENDER_PHONE: str

    # 이메일 설정
    EMAIL_HOST: str
    EMAIL_PORT: int
    EMAIL_HOST_USER: str
    EMAIL_HOST_PASSWORD: str

    # OAuth Settings
    GOOGLE_WEB_CLIENT_ID: str
    GOOGLE_WEB_CLIENT_SECRET: str
    GOOGLE_ANDROID_CLIENT_ID: str
    GOOGLE_ANDROID_SHA1: str
    GOOGLE_REDIRECT_URI: str

    NAVER_CLIENT_ID: str
    NAVER_CLIENT_SECRET: str
    NAVER_CLIENT_NAME: str
    NAVER_REDIRECT_URI: str

    KAKAO_CLIENT_ID: str
    KAKAO_NATIVE_APP_KEY: str
    KAKAO_CLIENT_SECRET: str
    KAKAO_REDIRECT_URI: str

    # WebSocket 관련 설정
    WEBSOCKET_MAX_CONNECTIONS: int = 1000
    WEBSOCKET_PING_INTERVAL: int = 30
    WEBSOCKET_PING_TIMEOUT: int = 10
    WEBSOCKET_MAX_MESSAGE_SIZE: int = 10 * 1024 * 1024
    WEBSOCKET_CONNECTION_TIMEOUT: int = 300
    WEBSOCKET_MESSAGE_TIMEOUT: int = 300
    WEBSOCKET_KEEP_ALIVE_TIMEOUT: int = 300
    WEBSOCKET_STATS_CLEANUP_INTERVAL: int = 3600

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        extra="ignore"
    )

# 환경 변수에 따라 서버 타입 설정
settings = Settings()
