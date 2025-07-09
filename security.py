import os
import re
import smtplib
import random
import string
import logging
import hmac
import base64
import hashlib
import time
import requests
import json
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import Dict, Optional, Any
from jose import jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from fastapi import HTTPException
from email.utils import formataddr
from core.config import settings
import httpx

from models.tables import User

# passlib의 로깅 레벨을 ERROR로 설정
logging.getLogger('passlib').setLevel(logging.ERROR)

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # ERROR 레벨만 로깅

# 비밀번호 해싱 설정
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,  # bcrypt 라운드 수 설정
    bcrypt__ident="2b"  # bcrypt 버전 식별자 명시적 설정
)

# JWT 설정
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="토큰이 만료되었습니다")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다")

async def send_verification_email(email: str, verification_code: str) -> None:
    try:
        message = MIMEText(f"""
        <div style="white-space: pre-line;">
안녕하세요, CAKE입니다.
회원가입을 완료하기 위해 아래의 인증 코드를 입력해주세요:

인증 코드: <span style="font-size: 1.2em;"><b>{verification_code}</b></span>

* 인증 코드는 5분간 유효합니다.
        </div>
        """, 'html')
        
        message["Subject"] = "CAKE 이메일 인증"
        message["From"] = formataddr(("CAKE", settings.EMAIL_HOST_USER))  
        message["To"] = email

        with smtplib.SMTP(settings.EMAIL_HOST, settings.EMAIL_PORT) as server:
            server.starttls()
            server.login(settings.EMAIL_HOST_USER, settings.EMAIL_HOST_PASSWORD)
            server.send_message(message)
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"이메일 전송에 실패했습니다: {str(e)}"
        )

def generate_verification_code() -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def generate_phone_verification_code() -> str:
    """6자리 숫자 인증코드 생성"""
    return ''.join(random.choices('0123456789', k=6))

def make_signature(secret_key: str, timestamp: str) -> str:
    """NCloud API 서명 생성"""
    method = "POST"
    uri = "/sms/v2/services/" + settings.NCLOUD_SMS_SERVICE_ID + "/messages"
    message = method + " " + uri + "\n" + timestamp + "\n" + settings.NCLOUD_ACCESS_KEY
    signing_key = bytes(secret_key, 'UTF-8')
    message = bytes(message, 'UTF-8')
    return base64.b64encode(hmac.new(signing_key, message, digestmod=hashlib.sha256).digest()).decode('UTF-8')

async def send_sms(phone_number: str, verification_code: str) -> bool:
    """네이버 클라우드 SMS 발송"""
    try:
        # 전화번호 형식 변환 (수신번호)
        phone_number = phone_number.replace("-", "")
        if phone_number.startswith("82"):
            phone_number = "0" + phone_number[2:]  # 82를 0으로 변환
        elif not phone_number.startswith("0"):
            phone_number = "0" + phone_number
            
        # 현재 시간을 timestamp로 변환
        timestamp = str(int(time.time() * 1000))
        
        # URI 및 API URL 설정
        service_id = settings.NCP_SERVICE_ID
        uri = f"/sms/v2/services/{service_id}/messages"
        api_url = f"https://sens.apigw.ntruss.com{uri}"
        
        # 서명 생성
        message = f"POST {uri}\n{timestamp}\n{settings.NCP_ACCESS_KEY}"
        signature = base64.b64encode(
            hmac.new(settings.NCP_SECRET_KEY.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).digest()
        ).decode("utf-8")
        
        # 발신번호 형식 변환
        from_number = settings.NCP_SENDER_PHONE.replace("-", "")
        if from_number.startswith("82"):
            from_number = "0" + from_number[2:]  # 82를 0으로 변환
        elif not from_number.startswith("0"):
            from_number = "0" + from_number
        
        # 요청 본문
        body = {
            "type": "SMS",
            "contentType": "COMM",
            "countryCode": "82",
            "from": from_number,
            "content": f"[CAKE] 인증번호 [{verification_code}]를 입력해주세요.",
            "messages": [
                {
                    "to": phone_number
                }
            ]
        }
        
        # 요청 헤더
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "x-ncp-apigw-timestamp": timestamp,
            "x-ncp-iam-access-key": settings.NCP_ACCESS_KEY,
            "x-ncp-apigw-signature-v2": signature
        }
        
        # SMS 발송 요청
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.post(api_url, headers=headers, json=body)
            
            if response.status_code == 202:
                return True
            else:
                logger.error(f"SMS 발송 실패 - 상태 코드: {response.status_code}")
                return False
                
    except Exception as e:
        logger.error(f"SMS 발송 중 오류 발생: {str(e)}")
        return False

async def get_current_user(db: Session, token: str) -> User:
    """
    JWT 토큰을 검증하고 현재 사용자를 반환하는 함수
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다")
            
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")
            
        return user
        
    except jwt.JWTError:
        raise HTTPException(
            status_code=401,
            detail="인증할 수 없습니다",
            headers={"WWW-Authenticate": "Bearer"}
        )

async def verify_websocket_token(token: str) -> dict:
    try:
        payload = verify_token(token)
        return payload
    except Exception as e:
        logger.error(f"WebSocket 토큰 검증 실패: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="유효하지 않은 토큰"
        )

def create_websocket_token(data: dict) -> str:
    """
    웹소켓 연결용 토큰을 생성합니다.
    만료 시간은 24시간으로 설정됩니다.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)  # 24시간
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)