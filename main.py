from fastapi import FastAPI, Depends, HTTPException, Body, Request, UploadFile, File, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import re
import logging
from typing import Dict, Any, Optional, List
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
import os
import shutil
from pathlib import Path
import httpx
import json
import functools
import time
import pytz
import asyncio

from core.database import get_db
from core.security import (
    verify_password, 
    get_password_hash, 
    create_access_token, 
    generate_verification_code, 
    send_verification_email, 
    get_current_user
)
from core.config import settings
from models.tables import (
    User, 
    Analysis, 
    DetectionDetail, 
    UserTermsAgreement,
    PhoneVerification,
    get_korean_time
)
from schemas.user import (
    UserCreate, 
    Token, 
    EmailVerification, 
    UserResponse, 
    LoginRequest, 
    SocialSignUpRequest, 
    SendPhoneVerificationRequest, 
    SendPhoneVerificationResponse, 
    VerifyPhoneRequest, 
    VerifyPhoneResponse, 
    AnalysisResponse, 
    AIAnalysisRequest, 
    AIAnalysisResponse, 
    AnalysisResultResponse, 
    UploadResponse, 
    DeepfakeResult, 
    UploadRequest,
    RiskLevel,
    DetectionType,
    RiskLevelInfo,
    FileType
)
from api.endpoints import auth, upload, realtime
from utils import detection

# 로깅 설정 개선
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# python_multipart의 로깅 레벨을 WARNING으로 설정
logging.getLogger('python_multipart.multipart').setLevel(logging.WARNING)

# passlib의 로깅 레벨을 WARNING으로 설정
logging.getLogger('passlib').setLevel(logging.WARNING)

# httpx의 로깅 레벨을 WARNING으로 설정
logging.getLogger('httpx').setLevel(logging.WARNING)

app = FastAPI()

# 정적 파일 서빙 설정
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/thumbnails", StaticFiles(directory="thumbnails"), name="thumbnails")

# CORS 설정을 settings에서 가져오도록 수정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*", "Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
    expose_headers=["*"],
    max_age=3600
)

# 라우터 등록
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(upload.router, prefix="/up", tags=["upload"])
app.include_router(realtime.router, prefix="", tags=["realtime"])
app.include_router(detection.router, prefix="/detection", tags=["detection"])

class EmailVerificationRequest(BaseModel):
    email: str

# AI 서버 통신 로깅 데코레이터
def log_ai_communication(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"AI 서버 통신 성공 - 함수: {func.__name__}, 소요시간: {duration:.2f}초")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"AI 서버 통신 실패 - 함수: {func.__name__}, 소요시간: {duration:.2f}초, 오류: {str(e)}")
            raise
    return wrapper

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")  # 루트 경로 접속 시 자동으로 /docs로 리다이렉트

@app.get("/test")
async def test_connection():
    return {"message": "Connection successful!"}

@app.post("/login", response_model=Token)
async def login(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"로그인 시도 - 이메일: {login_data.email}")
        
        # social_type이 'normal'일반 계정만 검색
        user = db.query(User).filter(
            User.email == login_data.email,
            (User.social_type == 'normal') | (User.social_type.is_(None))
        ).first()
        
        if not user:
            logger.warning(f"사용자를 찾을 수 없음 - 이메일: {login_data.email}")
            raise HTTPException(
                status_code=401,
                detail="이메일 또는 비밀번호가 일치하지 않습니다."
            )
            
        # 비밀번호 검증
        if not verify_password(login_data.password, user.hashed_password):
            logger.warning(f"비밀번호 불일치 - 이메일: {login_data.email}")
            raise HTTPException(
                status_code=401,
                detail="이메일 또는 비밀번호가 일치하지 않습니다."
            )
            
        # 인증되지 않은 계정인 경우
        if not user.is_verified:
            raise HTTPException(
                status_code=400,
                detail="BAD_REQUEST"
            )           
        # 액세스 토큰 생성
        access_token = create_access_token(
            data={"sub": user.email}
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user.id,
                "email": user.email,
                "username": user.username,
                "is_verified": user.is_verified,
                "phone_number": user.phone_number,
                "social_type": user.social_type,
                "account_type": user.account_type
            }
        } 
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="SERVER_ERROR"
        )

@app.post("/register", response_model=Dict[str, Any])
async def register(user: UserCreate, request: Request, db: Session = Depends(get_db)):
    try:
        logger.info(f"회원가입 시도 - 이메일: {user.email}, 사용자명: {user.username}")
        
        # 이메일과 account_type으로 중복 체크
        existing_user = db.query(User).filter(
            User.email == user.email,
            User.account_type == "normal"  # 일반 계정만 체크
        ).first()
        
        if existing_user:
            if existing_user.is_verified:
                logger.info(f"이미 존재하는 일반 계정 - 이메일: {existing_user.email}")
                raise HTTPException(
                    status_code=400,
                    detail="이미 가입된 이메일입니다."
                )
            else:
                # 미인증 계정인 경우 연관된 약관 동의 정보도 함께 삭제
                logger.info(f"미인증 계정 및 관련 데이터 삭제: {existing_user.email}")
                # 전화번호 인증 정보 삭제
                db.query(PhoneVerification).filter(
                    PhoneVerification.user_id == existing_user.id
                ).delete()
                # 약관 동의 정보 삭제
                db.query(UserTermsAgreement).filter(
                    UserTermsAgreement.user_id == existing_user.id
                ).delete()
                # 사용자 계정 삭제
                db.delete(existing_user)
                db.commit()
                logger.info(f"미인증 계정 삭제 완료: {existing_user.email}")
        
        # 비밀번호 확인
        if user.password != user.confirm_password:
            raise HTTPException(
                status_code=400,
                detail="비밀번호와 비밀번호 확인이 일치하지 않습니다."
            )
        
        # 비밀번호 유효성 검사
        if len(user.password) < 8:
            raise HTTPException(
                status_code=400,
                detail="비밀번호는 8자 이상이어야 합니다."
            )
        
        if not re.search(r"\d", user.password):
            raise HTTPException(
                status_code=400,
                detail="비밀번호에 숫자가 포함되어야 합니다."
            )
        
        # 특수문자 검사 - 허용된 특수문자만 사용 가능
        special_chars = r"[!@#$%^&*(),.?\":{}|<>]"
        if not re.search(special_chars, user.password):
            raise HTTPException(
                status_code=400,
                detail="비밀번호에 특수문자가 포함되어야 합니다."
            )
            
        # 허용되지 않는 특수문자가 있는지 확인
        invalid_chars = re.search(r'[^a-zA-Z0-9!@#$%^&*(),.?":{}|<>]', user.password)
        if invalid_chars:
            raise HTTPException(
                status_code=400,
                detail="비밀번호에 허용되지 않는 문자가 포함되어 있습니다."
            )
        
        # 사용자 생성
        hashed_password = get_password_hash(user.password)
        
        new_user = User(
            email=user.email,
            username=user.username,
            hashed_password=hashed_password,
            device_id=user.device_id,
            is_verified=False,  # 일반 회원가입은 항상 False로 시작
            social_type="normal",  # NULL 대신 'normal'로 설정
            account_type="normal",  # 기존과 동일
            phone_number=user.phone_number,
            verification_code=None,  # 초기값은 None
            verification_code_expires_at=None  # 초기값은 None
        )
        
        try:
            # 사용자 저장
            db.add(new_user)
            db.flush()  # ID 생성을 위해 flush
            
            # 약관 동의 정보 저장
            terms_agreements = [
                UserTermsAgreement(
                    user_id=new_user.id,
                    agreed_at=datetime.utcnow(),
                    terms_type="service",  # 서비스 이용약관
                    is_required=True,
                    device_id=user.device_id
                ),
                UserTermsAgreement(
                    user_id=new_user.id,
                    agreed_at=datetime.utcnow(),
                    terms_type="privacy",  # 개인정보 처리방침
                    is_required=True,
                    device_id=user.device_id
                ),
                UserTermsAgreement(
                    user_id=new_user.id,
                    agreed_at=datetime.utcnow(),
                    terms_type="media_learning",  # 미디어 컨텐츠 학습 동의
                    is_required=False,  # 선택적 동의
                    device_id=user.device_id
                )
            ]
            db.add_all(terms_agreements)
            
            # 최종 커밋
            db.commit()
            db.refresh(new_user)
            
            logger.info(f"일반 회원가입 성공 - 이메일: {user.email}")
            return {
                "success": True,
                "message": "이메일 인증을 진행해주세요.",
                "user": {
                    "id": new_user.id,
                    "email": new_user.email,
                    "username": new_user.username,
                    "account_type": new_user.account_type,
                    "phone_number": new_user.phone_number,
                    "is_verified": new_user.is_verified  # 인증 상태도 반환
                }
            }
        except Exception as e:
            logger.error(f"데이터베이스 저장 중 오류: {str(e)}")
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail="회원가입 처리 중 오류가 발생했습니다."
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"회원가입 오류 발생: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="서버 오류가 발생했습니다."
        )

@app.post("/verify-email")
async def verify_email(
    verification_data: EmailVerification,
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"이메일 인증 시도 - 이메일: {verification_data.email}, 코드: {verification_data.verification_code}")
        
        # 트랜잭션 시작
        try:
            # 일반 계정만 조회 (social_type이 'normal'인 계정)
            user = db.query(User).filter(
                User.email == verification_data.email,
                User.social_type == 'normal'  # 일반 계정만 처리
            ).with_for_update().first()
            
            if not user:
                logger.warning(f"일반 계정을 찾을 수 없음 - 이메일: {verification_data.email}")
                raise HTTPException(
                    status_code=404,
                    detail="등록되지 않은 이메일이거나 소셜 계정입니다."
                )
            
            # 인증 코드 만료 확인 (타임존 정보 제거 후 비교)
            current_time = get_korean_time().replace(tzinfo=None)
            expires_at = user.verification_code_expires_at.replace(tzinfo=None) if user.verification_code_expires_at else None
            
            logger.info(f"인증 코드 만료 시간 확인 - 현재: {current_time}, 만료: {expires_at}")
            
            if not expires_at or current_time > expires_at:
                logger.warning(f"인증 코드 만료 - 이메일: {verification_data.email}")
                raise HTTPException(
                    status_code=400,
                    detail="BAD_REQUEST"
                )
                
            # 인증 코드 확인
            logger.info(f"인증 코드 비교 - 입력: {verification_data.verification_code}, DB: {user.verification_code}")
            if user.verification_code != verification_data.verification_code:
                logger.warning(f"인증 코드 불일치 - 이메일: {verification_data.email}")
                raise HTTPException(
                    status_code=400,
                    detail="BAD_REQUEST"
                )
                
            # 이메일 인증 상태 업데이트
            logger.info(f"인증 상태 업데이트 전 - ID: {user.id}, is_verified: {user.is_verified}")
            user.is_verified = True
            user.verification_code = None  # 인증 완료 후 코드 삭제
            
            # 변경사항 커밋
            db.commit()
            
            # 실제로 업데이트되었는지 확인
            db.refresh(user)
            logger.info(f"인증 상태 업데이트 후 - ID: {user.id}, is_verified: {user.is_verified}")
            
            if not user.is_verified:
                logger.error(f"인증 상태 업데이트 실패 - 이메일: {verification_data.email}")
                raise HTTPException(
                    status_code=500,
                    detail="인증 상태 업데이트 실패"
                )
            
            return {"message": "이메일 인증이 완료되었습니다."}
            
        except Exception as e:
            db.rollback()
            logger.error(f"이메일 인증 처리 중 오류 발생: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="인증 처리 중 오류가 발생했습니다."
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Email verification error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="SERVER_ERROR"
        )

@app.post("/send-verification")
async def send_verification(data: EmailVerificationRequest, db: Session = Depends(get_db)):
    try:
        logger.info(f"이메일 인증 요청 시작 - 이메일: {data.email}")
        
        # 트랜잭션 시작
        try:
            # 일반 계정만 조회 (social_type이 'normal'인 계정)
            user = db.query(User).filter(
                User.email == data.email,
                User.social_type == 'normal'  # 일반 계정만 처리
            ).with_for_update().first()
            
            if not user:
                logger.warning(f"일반 계정을 찾을 수 없음 - 이메일: {data.email}")
                raise HTTPException(
                    status_code=404,
                    detail="등록되지 않은 이메일이거나 소셜 계정입니다."
                )
            
            # 현재 상태 로깅
            logger.info(f"현재 사용자 상태 - ID: {user.id}, is_verified: {user.is_verified}, social_type: {user.social_type}, verification_code: {user.verification_code}")
                
            # 새로운 인증 코드 생성 및 만료 시간 설정
            verification_code = generate_verification_code()
            logger.info(f"새 인증 코드 생성 - 코드: {verification_code}")
            
            # 사용자 정보 업데이트
            user.verification_code = verification_code
            user.verification_code_expires_at = get_korean_time() + timedelta(minutes=5)
            
            # 변경사항 저장
            db.commit()
            
            # 실제로 저장되었는지 확인
            db.refresh(user)
            logger.info(f"인증 코드 저장 후 상태 - ID: {user.id}, verification_code: {user.verification_code}, expires_at: {user.verification_code_expires_at}")
            
            if user.verification_code != verification_code:
                logger.error(f"인증 코드 저장 실패 - 예상: {verification_code}, 실제: {user.verification_code}")
                raise HTTPException(
                    status_code=500,
                    detail="인증 코드 저장 실패"
                )
            
            # 이메일 전송
            await send_verification_email(data.email, verification_code)
            logger.info(f"인증 이메일 전송 완료 - 이메일: {data.email}")
            
            return {
                "message": "인증 코드가 전송되었습니다."
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"인증 코드 저장 중 오류 발생: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="인증 코드 저장 중 오류가 발생했습니다."
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Verification email sending error: {str(e)}")
        logger.exception("상세 에러 스택 트레이스:")
        raise HTTPException(
            status_code=500,
            detail="인증 코드 전송 중 오류가 발생했습니다."
        )

@app.post("/send-detection", response_model=AIAnalysisRequest)
@log_ai_communication
async def send_to_ai(
    analysis_id: str,
    detection_type: DetectionType,
    social_type: str,
    frame_data: Optional[bytes] = File(None),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"===== 분석 요청 시작 =====")
        logger.info(f"분석 ID: {analysis_id}, 탐지 유형: {detection_type}, 소셜타입: {social_type}")
        
        # 분석 기록 조회
        analysis = db.query(Analysis).filter(
            Analysis.id == analysis_id,
            Analysis.social_type == social_type
        ).first()
        
        if not analysis:
            logger.error(f"분석 기록을 찾을 수 없음 - ID: {analysis_id}, 소셜타입: {social_type}")
            raise HTTPException(
                status_code=404,
                detail="분석 기록을 찾을 수 없습니다."
            )
        
        logger.info(f"분석 기록 조회 성공 - ID: {analysis.id}, 사용자: {analysis.user_email}")
        
        # 1. AI 서버로 요청 전송
        async with httpx.AsyncClient() as client:
            try:
                ai_request = AIAnalysisRequest(
                    analysis_id=analysis.id,
                    user_email=analysis.user_email,
                    device_id=analysis.device_id,
                    file_url=f"{settings.SERVER_URL}{analysis.file_url}",
                    detection_type=detection_type,
                    social_type=social_type
                )
                
                response = await client.post(
                    f"{settings.AI_SERVER_URL}/send-detection",
                    json=ai_request.model_dump(),
                    timeout=1800.0
                )
                
                if response.status_code != 200:
                    error_detail = response.json().get("detail", "알 수 없는 오류")
                    logger.error(f"AI 서버 응답 오류 - 상태 코드: {response.status_code}, 상세: {error_detail}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"AI 서버 오류: {error_detail}"
                    )
                
                # 2. AI 서버 응답 후 DB에서 최신 상태 확인
                db.refresh(analysis)  # DB 상태 새로고침
                
                # 3. 분석이 완료되었는지 확인
                if analysis.status == "completed":
                    logger.info(f"분석이 완료된 상태입니다 - ID: {analysis.id}")
                    return ai_request
                
                logger.info("AI 서버 요청 성공")
                return ai_request
                
            except httpx.TimeoutException:
                logger.error("AI 서버 요청 시간 초과")
                analysis.status = "failed"
                db.commit()
                raise HTTPException(
                    status_code=504,
                    detail="AI 서버 응답 시간 초과"
                )
            except httpx.RequestError as e:
                logger.error(f"AI 서버 연결 실패: {str(e)}")
                analysis.status = "failed"
                db.commit()
                raise HTTPException(
                    status_code=503,
                    detail="AI 서버 연결 실패"
                )
                
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"분석 요청 중 오류 발생: {str(e)}")
        logger.exception("상세 에러:")
        raise HTTPException(
            status_code=500,
            detail="분석 요청 처리 중 오류가 발생했습니다."
        )

@app.post("/receive-result", response_model=AnalysisResultResponse)
@log_ai_communication
async def receive_ai_result(request: Request, db: Session = Depends(get_db)):
    try:
        data = await request.json()
        analysis_id = data["analysis_id"]
        
        # 이미 처리된 결과인지 확인
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
            logger.error(f"분석 기록을 찾을 수 없음 - ID: {analysis_id}")
            raise HTTPException(
                status_code=404,
                detail="분석 기록을 찾을 수 없습니다."
            )

        if analysis.status == "completed":
            logger.info(f"이미 처리된 결과입니다 - ID: {analysis_id}")
            return {"message": "이미 처리된 결과입니다."}
            
        # 분석 결과가 실패인 경우
        if data.get("error") or data.get("confidence", 0.0) == 0.0:
            logger.error(f"분석 실패 - ID: {analysis_id}")
            analysis.status = "failed"
            analysis.updated_at = get_korean_time()
            db.commit()
            raise HTTPException(
                status_code=500,
                detail="분석 처리 실패"
            )

        # deepfake_score 처리 - 분석 서버에서 받은 원본 값 사용
        deepfake_score = data.get("deepfake_score", 0.0)
        if not deepfake_score and "details" in data:
            deepfake_score = data["details"].get("deepfake_score", 0.0)
            
        logger.info(f"분석 서버로부터 받은 deepfake_score: {deepfake_score}")
        
        # 위험도 점수 계산 (0-100 범위)
        score = round(deepfake_score * 100, 2)
        logger.info(f"계산된 위험도 점수: {score}")
        
        # 위험도 단계 계산 함수
        def calculate_risk_level(score: float) -> RiskLevel:
            if 0 <= score <= 25:
                return RiskLevel.VERY_SAFE
            elif 26 <= score <= 50:
                return RiskLevel.SAFE
            elif 51 <= score <= 70:
                return RiskLevel.NORMAL
            elif 71 <= score <= 80:
                return RiskLevel.DANGEROUS
            else:  # 81 <= score <= 100
                return RiskLevel.VERY_DANGEROUS

        # 위험도 단계 계산
        risk_level = calculate_risk_level(score)
        risk_level_info = RiskLevelInfo(
            level=risk_level,
            score=int(score)
        )
        
        # 1. AI 서버에서 받은 데이터를 detection_details 테이블에 저장
        details = data.get("details", {})
        
        # 기존 detection_details 삭제 (새로운 분석 결과로 대체)
        db.query(DetectionDetail).filter(
            DetectionDetail.analysis_id == analysis_id
        ).delete()
        
        # detection_type 설정 - file_type 문자열 비교로 변경
        detection_type = "youtube" if str(analysis.file_type) == "youtube" else "upload"
        logger.info(f"Detection type 설정 - file_type: {analysis.file_type}, detection_type: {detection_type}")
        
        try:
            # 새로운 detection_details 저장 - 원본 deepfake_score 사용
            detection_detail = DetectionDetail(
                analysis_id=analysis_id,
                frame_number=0,  # 전체 분석 결과는 frame_number 0으로 저장
                detection_type=detection_type,
                confidence_score=data.get("confidence", 0.0),
                detected_artifacts=deepfake_score,  # 원본 deepfake_score 사용
                timestamp=get_korean_time()
            )
            db.add(detection_detail)
            
            # 프레임별 데이터가 있는 경우 저장
            if "frames" in details:
                for frame_num, frame_data in details["frames"].items():
                    frame_detail = DetectionDetail(
                        analysis_id=analysis_id,
                        frame_number=int(frame_num),
                        detection_type=detection_type,
                        confidence_score=frame_data.get("confidence", 0.0),
                        detected_artifacts=frame_data.get("artifacts", deepfake_score),  # 프레임별 artifacts가 없으면 전체 deepfake_score 사용
                        timestamp=get_korean_time()
                    )
                    db.add(frame_detail)
            
            # DB에 저장
            db.commit()
            logger.info(f"detection_details 저장 완료 - ID: {analysis_id}, detection_type: {detection_type}, deepfake_score: {deepfake_score}")
            
        except Exception as e:
            db.rollback()
            logger.error(f"detection_details 저장 중 오류 발생: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"분석 결과 저장 중 오류가 발생했습니다: {str(e)}"
            )

        # 2. 저장된 detection_details 데이터를 조회
        detection_details = db.query(DetectionDetail).filter(
            DetectionDetail.analysis_id == analysis_id
        ).order_by(DetectionDetail.frame_number).all()

        # 3. 조회한 데이터로 frames_data 생성
        frames_data = {
            str(detail.frame_number): {
                "confidence": detail.confidence_score,
                "artifacts": detail.detected_artifacts
            }
            for detail in detection_details
        }

        # 4. analyses 테이블 업데이트
        current_time = get_korean_time()
        analysis.confidence = data.get("confidence", 0.0)
        analysis.analysis_time = data.get("analysis_time", 0.0)
        analysis.status = "completed"
        analysis.updated_at = current_time
        analysis.report_url = data.get("report_url")  # 리포트 URL 저장
        
        # DetectionDetail 테이블의 데이터를 기반으로 details 컬럼 업데이트
        analysis.details = json.dumps({
            "score": score,
            "risk_level": risk_level.value,
            "details": {
                "frames": frames_data,
                "detection_details": [
                    {
                        "frame_number": detail.frame_number,
                        "confidence_score": detail.confidence_score,
                        "detected_artifacts": detail.detected_artifacts,
                        "timestamp": detail.timestamp.isoformat()
                    }
                    for detail in detection_details
                ]
            }
        }, ensure_ascii=False)
        
        # 5. 썸네일 생성
        file_path = Path(analysis.file_path)
        if file_path.exists():
            thumbnail_dir = Path("thumbnails")
            thumbnail_dir.mkdir(exist_ok=True)
            thumbnail_filename = f"{analysis_id}.webp"
            thumbnail_path = thumbnail_dir / thumbnail_filename
            
            if not thumbnail_path.exists():
                logger.info("썸네일 생성 시도")
                from api.endpoints.upload import generate_thumbnail
                thumbnail_created = await generate_thumbnail(file_path, thumbnail_path)
                if thumbnail_created:
                    analysis.thumbnail_url = f"/thumbnails/{thumbnail_filename}"
                    logger.info(f"썸네일 생성 및 URL 저장 완료: {analysis.thumbnail_url}")
                else:
                    logger.error("썸네일 생성 실패")
        
        # DB에 저장하고 새로고침
        try:
            db.commit()
            db.refresh(analysis)
            logger.info(f"분석 결과 및 상세 데이터 저장 완료 - ID: {analysis_id}, 상태: {analysis.status}, 시간: {current_time}")
            
        except Exception as e:
            logger.error(f"DB 커밋 중 오류 발생: {str(e)}")
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail="분석 결과 저장 중 오류가 발생했습니다."
            )
        
        # 응답 데이터 구성
        response_data = {
            "id": analysis.id,
            "user_email": analysis.user_email,
            "device_id": analysis.device_id,
            "confidence": analysis.confidence,
            "confidence_percent": round(analysis.confidence * 100, 2),
            "risk_level_info": risk_level_info,
            "analysis_time": analysis.analysis_time,
            "details": analysis.details,
            "file_url": analysis.file_url,
            "thumbnail_url": analysis.thumbnail_url,
            "report_url": analysis.report_url,  # 리포트 URL 응답에 포함
            "created_at": analysis.created_at,
            "status": analysis.status
        }
        
        logger.info("===== AI 분석 결과 수신 완료 =====")
        return response_data
        
    except Exception as e:
        logger.error(f"분석 결과 처리 중 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="분석 결과 처리 중 오류가 발생했습니다."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)