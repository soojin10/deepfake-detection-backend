from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from loguru import logger
import re, httpx
from fastapi.responses import RedirectResponse
from core.config import settings
import random
from datetime import datetime, timedelta
import os
import traceback

from core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    send_verification_email,
    generate_verification_code,
    send_sms as security_send_sms
)
from schemas.user import (
    UserCreate, 
    EmailVerification, 
    SocialSignUpRequest,
    CheckUserResponse,
    SendPhoneVerificationRequest, 
    SendPhoneVerificationResponse,
    VerifyPhoneRequest, 
    VerifyPhoneResponse,
    EmailCheck,
    SocialLoginRequest,
    Token,
    UpdateUsernameRequest,
    UpdateUsernameResponse,
    UpdatePhoneRequest,
    UpdatePhoneResponse,
    UpdatePasswordRequest,
    UpdatePasswordResponse
)
from core.database import get_db
from models.tables import User, PhoneVerification, UserTermsAgreement, Analysis, DetectionDetail

router = APIRouter()

# 유효한 소셜 타입 정의
VALID_SOCIAL_TYPES = {"kakao", "naver", "google"}

def check_email_exists(db: Session, email: str) -> bool:
    existing_user = db.query(User).filter(User.email == email).first()
    return existing_user is not None

def validate_password(password: str) -> bool:
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="BAD_REQUEST")
    
    if not re.search(r"\d", password):
        raise HTTPException(status_code=400, detail="BAD_REQUEST")
    
    # 특수문자 검사 - 허용된 특수문자만 사용 가능
    special_chars = r"[!@#$%^&*(),.?\":{}|<>]"
    if not re.search(special_chars, password):
        raise HTTPException(status_code=400, detail="BAD_REQUEST")
        
    # 허용되지 않는 특수문자가 있는지 확인
    invalid_chars = re.search(r'[^a-zA-Z0-9!@#$%^&*(),.?":{}|<>]', password)
    if invalid_chars:
        raise HTTPException(status_code=400, detail="BAD_REQUEST")
    
    return True

@router.get("/check-user", response_model=CheckUserResponse)
async def check_user(
    email: str,
    account_type: str,  # "normal" 또는 "social"
    db: Session = Depends(get_db)
):
    try:
        # account_type에 따라 다른 조건으로 조회
        if account_type == "normal":
            # 일반 계정에서만 체크 (social_type이 'normal'인 경우)
            user = db.query(User).filter(
                User.email == email,
                User.social_type == 'normal'
            ).first()
        else:  # social
            # 소셜 계정에서만 체크 (social_type이 'google', 'naver', 'kakao'인 경우)
            user = db.query(User).filter(
                User.email == email,
                User.social_type.in_(['google', 'naver', 'kakao'])
            ).first()
            
        if user:
            return {
                "exists": True,
                "message": "이미 가입된 사용자입니다."
            }
        return {
            "exists": False,
            "message": "가입되지 않은 사용자입니다."
        }
    except Exception as e:
        logger.error(f"사용자 확인 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="SERVER_ERROR"
        )

@router.post("/social-signup")
async def social_signup(user_data: SocialSignUpRequest, db: Session = Depends(get_db)):
    try:
        logger.info(f"소셜 로그인 시도 - 이메일: {user_data.email}, 소셜 타입: {user_data.social_type}")
        
        # 이메일과 account_type으로 중복 체크
        existing_user = db.query(User).filter(
            User.email == user_data.email,
            User.account_type == "social",  # 소셜 계정만 체크
            User.social_type == user_data.social_type,  # 동일한 소셜 타입만 체크
            User.social_id == user_data.social_id  # 동일한 소셜 ID만 체크
        ).first()
        
        if existing_user:
            logger.info(f"기존 소셜 계정 발견 - 이메일: {existing_user.email}")
            if existing_user.is_verified:
                # 이미 인증된 계정인 경우, 로그인 처리
                logger.info(f"동일 소셜 계정으로 로그인 - 이메일: {existing_user.email}")
                access_token = create_access_token({"sub": existing_user.email})
                return {
                    "success": True,
                    "access_token": access_token,
                    "token_type": "bearer",
                    "user": {
                        "id": existing_user.id,
                        "email": existing_user.email,
                        "username": existing_user.username,
                        "social_type": existing_user.social_type,
                        "account_type": existing_user.account_type,
                        "phone_number": existing_user.phone_number
                    }
                }
            else:
                # 미인증 계정인 경우 삭제 후 다시 생성
                logger.info(f"미인증 계정 삭제: {existing_user.email}")
                db.delete(existing_user)
                db.commit()

        # 새로운 사용자 생성
        logger.info(f"새로운 소셜 사용자 생성 - 이메일: {user_data.email}, 소셜 타입: {user_data.social_type}")
        new_user = User(
            email=user_data.email,
            username=user_data.username or user_data.email.split('@')[0],  # username이 없으면 이메일에서 추출
            social_type=user_data.social_type,
            social_id=user_data.social_id,
            device_id=user_data.device_id,
            is_verified=True,  # 소셜 로그인은 이메일이 이미 인증됨
            account_type="social",  # 항상 social로 설정
            phone_number=user_data.phone_number  # 전화번호 저장
        )
        
        try:
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            logger.info(f"새로운 소셜 사용자 생성 완료 - ID: {new_user.id}, 이메일: {new_user.email}")
            
            # 약관 동의 정보 저장
            terms_agreements = [
                UserTermsAgreement(
                    user_id=new_user.id,
                    agreed_at=datetime.utcnow(),
                    terms_type="service",  # 서비스 이용약관
                    is_required=True,
                    device_id=user_data.device_id
                ),
                UserTermsAgreement(
                    user_id=new_user.id,
                    agreed_at=datetime.utcnow(),
                    terms_type="privacy",  # 개인정보 처리방침
                    is_required=True,
                    device_id=user_data.device_id
                ),
                UserTermsAgreement(
                    user_id=new_user.id,
                    agreed_at=datetime.utcnow(),
                    terms_type="media_learning",  # 미디어 컨텐츠 학습 동의
                    is_required=False,  # 선택적 동의
                    device_id=user_data.device_id
                )
            ]
            db.add_all(terms_agreements)
            db.commit()
            
            # 액세스 토큰 생성
            access_token = create_access_token({"sub": new_user.email})
            logger.info(f"액세스 토큰 생성 완료 - 이메일: {new_user.email}")
            
            return {
                "success": True,
                "access_token": access_token,
                "token_type": "bearer",
                "user": {
                    "id": new_user.id,
                    "email": new_user.email,
                    "username": new_user.username,
                    "social_type": new_user.social_type,
                    "account_type": new_user.account_type,
                    "phone_number": new_user.phone_number
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
        logger.error(f"소셜 로그인 HTTP 예외 발생: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"소셜 로그인 오류 발생: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="SERVER_ERROR"
        )

@router.get("/google/login")
async def google_login(client_type: str = "web"):
    if client_type == "android":
        client_id = settings.GOOGLE_ANDROID_CLIENT_ID
    else:
        client_id = settings.GOOGLE_WEB_CLIENT_ID
        
    return RedirectResponse(
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={client_id}&"
        f"redirect_uri={settings.GOOGLE_REDIRECT_URI}&"
        f"response_type=code&"
        f"scope=email profile"
    )

@router.get("/google/callback")
async def google_callback(code: str = None, id_token: str = None, client_type: str = "web", db: Session = Depends(get_db)):
    try:
        if client_type == "android" and id_token:
            # 안드로이드 앱의 경우 id_token을 직접 검증
            async with httpx.AsyncClient() as client:
                # id_token 검증
                user_response = await client.get(
                    "https://oauth2.googleapis.com/tokeninfo",
                    params={"id_token": id_token}
                )
                
                if user_response.status_code != 200:
                    logger.error(f"Google token verification failed: {user_response.text}")
                    raise HTTPException(
                        status_code=400,
                        detail="Google token verification failed"
                    )
                    
                user_data = user_response.json()
                
                # 소셜 로그인 처리
                social_data = SocialSignUpRequest(
                    email=user_data["email"],
                    username=user_data.get("name", user_data["email"].split("@")[0]),
                    social_type="google",
                    social_id=user_data["sub"]
                )
                
                return await social_signup(social_data, db)
                
        else:
            # 웹 클라이언트의 경우 기존 코드 사용
            async with httpx.AsyncClient() as client:
                token_response = await client.post(
                    "https://oauth2.googleapis.com/token",
                    data={
                        "client_id": settings.GOOGLE_WEB_CLIENT_ID,
                        "client_secret": settings.GOOGLE_WEB_CLIENT_SECRET,
                        "code": code,
                        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
                        "grant_type": "authorization_code",
                    },
                )
                
                if token_response.status_code != 200:
                    logger.error(f"Google token request failed: {token_response.text}")
                    raise HTTPException(
                        status_code=400,
                        detail="Google token request failed"
                    )
                    
                token_data = token_response.json()
                
                # 사용자 정보 요청
                user_response = await client.get(
                    "https://www.googleapis.com/oauth2/v2/userinfo",
                    headers={"Authorization": f"Bearer {token_data['access_token']}"},
                )
                
                if user_response.status_code != 200:
                    logger.error(f"Google user info request failed: {user_response.text}")
                    raise HTTPException(
                        status_code=400,
                        detail="Google user info request failed"
                    )
                    
                user_data = user_response.json()
                
                # 소셜 로그인 처리
                social_data = SocialSignUpRequest(
                    email=user_data["email"],
                    username=user_data.get("name", user_data["email"].split("@")[0]),
                    social_type="google",
                    social_id=user_data["id"]
                )
                
                return await social_signup(social_data, db)
                
    except Exception as e:
        logger.error(f"Google login error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Google login error: {str(e)}"
        )

@router.get("/naver/login")
async def naver_login():
    return RedirectResponse(
        f"https://nid.naver.com/oauth2.0/authorize?"
        f"client_id={settings.NAVER_CLIENT_ID}&"
        f"redirect_uri={settings.NAVER_REDIRECT_URI}&"
        f"response_type=code&"
        f"state=random_state"
    )

@router.get("/naver/callback")
async def naver_callback(code: str, state: str, db: Session = Depends(get_db)):
    async with httpx.AsyncClient() as client:
        # 토큰 요청
        token_response = await client.post(
            "https://nid.naver.com/oauth2.0/token",
            data={
                "client_id": settings.NAVER_CLIENT_ID,
                "client_secret": settings.NAVER_CLIENT_SECRET,
                "code": code,
                "state": state,
                "grant_type": "authorization_code",
            },
        )
        token_data = token_response.json()
        
        # 사용자 정보 요청
        user_response = await client.get(
            "https://openapi.naver.com/v1/nid/me",
            headers={"Authorization": f"Bearer {token_data['access_token']}"},
        )
        user_data = user_response.json()
        
        # 소셜 로그인 처리
        social_data = SocialSignUpRequest(
            email=user_data["response"]["email"],
            username=user_data["response"].get("name", user_data["response"]["email"].split("@")[0]),
            social_type="naver",  
            social_id=user_data["response"]["id"]
        )
        
        return await social_signup(social_data, db)

@router.get("/kakao/login")
async def kakao_login():
    return RedirectResponse(
        f"https://kauth.kakao.com/oauth/authorize?"
        f"client_id={settings.KAKAO_CLIENT_ID}&"
        f"redirect_uri={settings.KAKAO_REDIRECT_URI}&"
        f"response_type=code"
    )

@router.get("/kakao/callback")
async def kakao_callback(code: str, db: Session = Depends(get_db)):
    async with httpx.AsyncClient() as client:
        # 토큰 요청
        token_response = await client.post(
            "https://kauth.kakao.com/oauth/token",
            data={
                "client_id": settings.KAKAO_CLIENT_ID,
                "client_secret": settings.KAKAO_CLIENT_SECRET,
                "code": code,
                "redirect_uri": settings.KAKAO_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )
        token_data = token_response.json()
        
        # 사용자 정보 요청
        user_response = await client.get(
            "https://kapi.kakao.com/v2/user/me",
            headers={"Authorization": f"Bearer {token_data['access_token']}"},
        )
        user_data = user_response.json()
        
        # 소셜 로그인 처리
        social_data = SocialSignUpRequest(
            email=user_data["kakao_account"]["email"],
            username=user_data["properties"].get("nickname", user_data["kakao_account"]["email"].split("@")[0]),
            social_type="kakao", 
            social_id=str(user_data["id"])
        )
        
        return await social_signup(social_data, db)

def generate_phone_verification_code() -> str:
    """6자리 숫자 인증코드 생성"""
    return ''.join(random.choices('0123456789', k=6))

async def send_sms(phone_number: str, verification_code: str) -> bool:
    """SMS 발송"""
    try:
        logger.info(f"===== SMS 발송 시작 =====")
        logger.info(f"전화번호: {phone_number}")
        logger.info(f"인증코드: {verification_code}")
        
        result = await security_send_sms(phone_number, verification_code)
        
        if not result:
            logger.error("SMS 발송 실패 - security.send_sms가 False를 반환")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"SMS 발송 중 예외 발생: {str(e)}")
        logger.exception("상세 에러:")  # 스택 트레이스 출력
        return False

@router.post("/send-phone-verification", response_model=SendPhoneVerificationResponse)
async def send_phone_verification(
    request: SendPhoneVerificationRequest,
    user_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"===== 전화번호 인증 시작 =====")
        logger.info(f"전화번호: {request.phone_number}")
        
        # 전화번호 형식 검증
        clean_phone = request.phone_number.replace("-", "")
        if not re.match(r'^\d{10,11}$', clean_phone):
            logger.error(f"잘못된 전화번호 형식: {request.phone_number}")
            raise HTTPException(
                status_code=400,
                detail="올바른 전화번호 형식이 아닙니다."
            )
            
        logger.info(f"정제된 전화번호: {clean_phone}")
        
        # 이전 인증 정보 만료 처리
        existing_verifications = db.query(PhoneVerification).filter(
            PhoneVerification.phone_number == clean_phone,
            PhoneVerification.is_verified == False,
            PhoneVerification.is_expired == False
        ).all()
        
        for verification in existing_verifications:
            verification.is_expired = True
            verification.expired_at = datetime.utcnow()
            logger.info(f"이전 인증 만료 처리: {verification.verification_code}")
        
        # 새 인증 코드 생성
        verification_code = generate_phone_verification_code()
        created_at = datetime.utcnow()
        expires_at = created_at + timedelta(minutes=5)
        
        logger.info(f"새 인증 코드 생성: {verification_code}")
        
        # 새 인증 정보 저장
        verification = PhoneVerification(
            phone_number=clean_phone,
            verification_code=verification_code,
            created_at=created_at,
            expires_at=expires_at,
            is_verified=False,
            is_expired=False,
            user_id=user_id
        )
        db.add(verification)
        
        # SMS 발송
        logger.info("SMS 발송 시도...")
        success = await send_sms(clean_phone, verification_code)
        
        if not success:
            logger.error(f"SMS 발송 실패 - 전화번호: {clean_phone}")
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail="SMS 발송에 실패했습니다."
            )
        
        logger.info(f"SMS 발송 성공 - 전화번호: {clean_phone}, 인증코드: {verification_code}")
        db.commit()
        
        return SendPhoneVerificationResponse(
            message="인증번호가 발송되었습니다."
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"전화번호 인증 코드 발송 중 예외 발생: {str(e)}")
        logger.exception("상세 에러:")  # 스택 트레이스 출력
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="서버 오류가 발생했습니다."
        )

@router.post("/verify-phone", response_model=VerifyPhoneResponse)
async def verify_phone(
    request: VerifyPhoneRequest,
    db: Session = Depends(get_db)
):
    try:
        # 전화번호로 인증 정보 조회
        verification = db.query(PhoneVerification).filter(
            PhoneVerification.phone_number == request.phone_number,
            PhoneVerification.is_verified == False,
            PhoneVerification.is_expired == False,
            PhoneVerification.expires_at > datetime.utcnow()
        ).order_by(PhoneVerification.created_at.desc()).first()
        
        if not verification:
            raise HTTPException(
                status_code=404,
                detail="유효한 인증번호가 없습니다."
            )
        
        # 인증 코드 확인
        if verification.verification_code != request.verification_code:
            raise HTTPException(
                status_code=400,
                detail="잘못된 인증번호입니다."
            )
        
        # 인증 완료 처리
        verification.is_verified = True
        verification.verified_at = datetime.utcnow()
        db.commit()
        
        return VerifyPhoneResponse(
            message="전화번호 인증이 완료되었습니다.",
            verified=True
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"전화번호 인증 확인 오류: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="서버 오류가 발생했습니다."
        )

@router.post("/check-email")
async def check_email(data: EmailCheck, db: Session = Depends(get_db)):
    try:
        # email 필드가 없거나 비어있는 경우 처리
        if not data.email:
            return {
                "available": False,
                "message": "이메일을 입력해주세요."
            }
            
        # 이메일 형식 검사
        if not re.match(r"[^@]+@[^@]+\.[^@]+", data.email):
            return {
                "available": False,
                "message": "올바른 이메일 형식이 아닙니다."
            }
            
        # account_type에 따라 다른 조건으로 조회
        if data.account_type == "normal":
            # 일반 계정에서만 중복 체크 (social_type이 NULL인 경우)
            existing_user = db.query(User).filter(
                User.email == data.email,
                User.social_type.is_(None)
            ).first()
        else:  # social
            # 소셜 계정에서만 중복 체크 (social_type이 NULL이 아닌 경우)
            existing_user = db.query(User).filter(
                User.email == data.email,
                User.social_type.isnot(None)
            ).first()
        
        # 사용자가 있고 이미 인증된 상태(is_verified=True)인 경우만 사용 불가로 처리
        if existing_user and existing_user.is_verified:
            return {
                "available": False,
                "message": "이미 사용 중인 이메일입니다."
            }
        
        # 사용자가 없거나, 있어도 미인증 상태라면 사용 가능으로 처리
        return {
            "available": True,
            "message": "사용 가능한 이메일입니다."
        }
        
    except Exception as e:
        logger.error(f"Email check error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="이메일 중복 확인 중 오류가 발생했습니다."
        ) 

@router.post("/social-login", response_model=Token)
async def social_login(
    login_data: SocialLoginRequest,
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"소셜 로그인 시도 - 이메일: {login_data.email}, 소셜 타입: {login_data.social_type}")
        
        # 소셜 타입 검증
        social_type = login_data.social_type.lower()
        if social_type not in VALID_SOCIAL_TYPES:
            raise HTTPException(
                status_code=400,
                detail="유효하지 않은 소셜 타입입니다."
            )
        
        # 사용자 조회 쿼리 로깅
        query = db.query(User).filter(
            User.email == login_data.email,
            User.social_type == social_type
        )
        logger.info(f"소셜 로그인 쿼리: {str(query)}")
        
        # 사용자 조회
        user = query.first()
        
        if user:
            logger.info(f"소셜 계정 찾음 - ID: {user.id}, 이메일: {user.email}, 소셜 타입: {user.social_type}")
        else:
            logger.warning(f"소셜 계정 없음 - 이메일: {login_data.email}, 소셜 타입: {social_type}")
            # 해당 이메일의 모든 계정 조회 (디버깅용)
            all_accounts = db.query(User).filter(User.email == login_data.email).all()
            if all_accounts:
                logger.info("발견된 계정들:")
                for account in all_accounts:
                    logger.info(f"- ID: {account.id}, 이메일: {account.email}, 소셜 타입: {account.social_type}, 계정 타입: {account.account_type}")
            else:
                logger.info("해당 이메일로 가입된 계정이 없습니다.")
            
            raise HTTPException(
                status_code=404,
                detail="해당하는 소셜 계정을 찾을 수 없습니다."
            )
            
        # device_id 업데이트
        user.device_id = login_data.device_id
        db.commit()
        logger.info(f"device_id 업데이트 완료: {login_data.device_id}")
        
        # 액세스 토큰 생성
        access_token = create_access_token(
            data={"sub": user.email}
        )
        logger.info(f"액세스 토큰 생성 완료 - 이메일: {user.email}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user.id,
                "email": user.email,
                "username": user.username,
                "social_type": user.social_type,
                "device_id": user.device_id,
                "is_verified": user.is_verified,
                "account_type": user.account_type,
                "phone_number": user.phone_number
            }
        }
        
    except HTTPException as he:
        logger.error(f"소셜 로그인 HTTP 예외: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"소셜 로그인 오류: {str(e)}")
        logger.exception("상세 에러:")  # 스택 트레이스 출력
        raise HTTPException(
            status_code=500,
            detail="소셜 로그인 처리 중 오류가 발생했습니다."
        )

# 회원탈퇴
@router.delete("/withdraw")
async def withdraw_user(
    user_email: str,
    social_type: str,  
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"사용자 탈퇴 시작 - 이메일: {user_email}, 소셜 타입: {social_type}")
        
        # 1. 요청된 소셜 타입과 정확히 일치하는 사용자 계정 조회
        user = db.query(User).filter(
            User.email == user_email,
            User.social_type == social_type  # 정확히 일치하는 소셜 타입만
        ).first()
        
        if not user:
            logger.warning(f"탈퇴하려는 사용자를 찾을 수 없음 - 이메일: {user_email}, 소셜 타입: {social_type}")
            raise HTTPException(
                status_code=404,
                detail="사용자를 찾을 수 없습니다."
            )
            
        logger.info(f"탈퇴할 사용자 정보 - ID: {user.id}, 이메일: {user.email}, 소셜 타입: {user.social_type}, 계정 타입: {user.account_type}")
        
        # 삭제 전 연관 데이터 확인 및 로깅
        phone_verifications_count = db.query(PhoneVerification).filter(
            PhoneVerification.user_id == user.id
        ).count()
        
        analyses_count = db.query(Analysis).filter(
            Analysis.user_email == user.email,
            Analysis.social_type == user.social_type
        ).count()
        
        terms_agreements_count = db.query(UserTermsAgreement).filter(
            UserTermsAgreement.user_id == user.id
        ).count()
        
        logger.info(f"삭제할 연관 데이터 - 전화번호 인증: {phone_verifications_count}, 분석 기록: {analyses_count}, 약관 동의: {terms_agreements_count}")
            
        try:
            # 2. 연관된 데이터 삭제 (외래 키 제약조건을 고려한 순서)
            logger.info("연관 데이터 삭제 시작")
            
            # 2-1. 먼저 phone_verifications 테이블의 데이터 삭제
            deleted_phone_verifications = db.query(PhoneVerification).filter(
                PhoneVerification.user_id == user.id
            ).delete()
            db.commit()  # 각 단계별로 커밋
            logger.info(f"전화번호 인증 정보 삭제 완료 - 삭제된 레코드: {deleted_phone_verifications}")
            
            # 2-2. 그 다음 detection_details 테이블의 데이터 삭제
            analyses = db.query(Analysis).filter(
                Analysis.user_email == user.email,
                Analysis.social_type == user.social_type  # 해당 소셜 타입의 분석 기록만
            ).all()
            
            total_detection_details = 0
            for analysis in analyses:
                deleted_details = db.query(DetectionDetail).filter(
                    DetectionDetail.analysis_id == analysis.id
                ).delete()
                total_detection_details += deleted_details
            db.commit()  # 각 단계별로 커밋
            logger.info(f"탐지 상세 정보 삭제 완료 - 삭제된 레코드: {total_detection_details}")
            
            # 2-3. 그 다음 analyses 테이블의 데이터 삭제
            deleted_analyses = db.query(Analysis).filter(
                Analysis.user_email == user.email,
                Analysis.social_type == user.social_type  # 해당 소셜 타입의 분석 기록만
            ).delete()
            db.commit()  # 각 단계별로 커밋
            logger.info(f"분석 기록 삭제 완료 - 삭제된 레코드: {deleted_analyses}")
            
            # 2-4. 약관 동의 정보 삭제
            deleted_terms = db.query(UserTermsAgreement).filter(
                UserTermsAgreement.user_id == user.id
            ).delete()
            db.commit()  # 각 단계별로 커밋
            logger.info(f"약관 동의 정보 삭제 완료 - 삭제된 레코드: {deleted_terms}")
            
            # 3. 사용자 계정 삭제
            logger.info(f"사용자 계정 삭제 시도 - ID: {user.id}")
            db.delete(user)
            db.commit()
            
            logger.info(f"사용자 탈퇴 완료 - 이메일: {user_email}, 소셜 타입: {social_type}")
            return {
                "success": True,
                "message": "회원 탈퇴가 완료되었습니다."
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"사용자 삭제 중 오류 발생: {str(e)}")
            logger.exception("상세 에러 스택 트레이스:")
            
            # 오류 발생 시 남은 연관 데이터 확인
            remaining_phone_verifications = db.query(PhoneVerification).filter(
                PhoneVerification.user_id == user.id
            ).count()
            
            remaining_analyses = db.query(Analysis).filter(
                Analysis.user_email == user.email,
                Analysis.social_type == user.social_type
            ).count()
            
            remaining_terms = db.query(UserTermsAgreement).filter(
                UserTermsAgreement.user_id == user.id
            ).count()
            
            logger.error(f"삭제 실패 후 남은 데이터 - 전화번호 인증: {remaining_phone_verifications}, 분석 기록: {remaining_analyses}, 약관 동의: {remaining_terms}")
            
            raise HTTPException(
                status_code=500,
                detail=f"회원 탈퇴 처리 중 오류가 발생했습니다: {str(e)}"
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"회원 탈퇴 처리 중 예상치 못한 오류 발생: {str(e)}")
        logger.exception("상세 에러 스택 트레이스:")
        raise HTTPException(
            status_code=500,
            detail="회원 탈퇴 처리 중 오류가 발생했습니다."
        )

@router.post("/update-username", response_model=UpdateUsernameResponse)
async def update_username(
    request: UpdateUsernameRequest,
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"사용자 이름 변경 시도 - 이메일: {request.email}, 소셜 타입: {request.social_type}")
        
        # 1. 사용자 존재 여부 확인
        user = db.query(User).filter(
            User.email == request.email,
            User.social_type == request.social_type
        ).first()
        
        if not user:
            logger.warning(f"사용자를 찾을 수 없음 - 이메일: {request.email}, 소셜 타입: {request.social_type}")
            raise HTTPException(
                status_code=404,
                detail="사용자를 찾을 수 없습니다."
            )
            
        # 2. 새 사용자 이름 유효성 검사
        if not request.new_username or len(request.new_username.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="사용자 이름은 비어있을 수 없습니다."
            )
            
        if len(request.new_username) > 50:  # User 모델의 username 필드 길이 제한
            raise HTTPException(
                status_code=400,
                detail="사용자 이름은 50자를 초과할 수 없습니다."
            )
            
        # 3. 현재 이름과 동일한지 확인
        if user.username == request.new_username:
            return UpdateUsernameResponse(
                success=True,
                message="현재 사용 중인 이름과 동일합니다.",
                user={
                    "id": user.id,
                    "email": user.email,
                    "username": user.username,
                    "social_type": user.social_type,
                    "device_id": user.device_id,
                    "is_verified": user.is_verified,
                    "account_type": user.account_type,
                    "phone_number": user.phone_number
                }
            )
            
        try:
            # 4. 사용자 이름 업데이트
            user.username = request.new_username
            db.commit()
            db.refresh(user)
            
            logger.info(f"사용자 이름 변경 완료 - 이메일: {user.email}, 새 이름: {user.username}")
            
            return UpdateUsernameResponse(
                success=True,
                message="사용자 이름이 변경되었습니다.",
                user={
                    "id": user.id,
                    "email": user.email,
                    "username": user.username,
                    "social_type": user.social_type,
                    "device_id": user.device_id,
                    "is_verified": user.is_verified,
                    "account_type": user.account_type,
                    "phone_number": user.phone_number
                }
            )
            
        except Exception as e:
            db.rollback()
            logger.error(f"사용자 이름 변경 중 오류 발생: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="사용자 이름 변경 중 오류가 발생했습니다."
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"사용자 이름 변경 처리 중 예상치 못한 오류 발생: {str(e)}")
        logger.exception("상세 에러 스택 트레이스:")
        raise HTTPException(
            status_code=500,
            detail="사용자 이름 변경 처리 중 오류가 발생했습니다."
        )

@router.post("/update-phone", response_model=UpdatePhoneResponse)
async def update_phone(
    request: UpdatePhoneRequest,
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"전화번호 변경 시도 - 이메일: {request.email}, 소셜 타입: {request.social_type}")
        
        # 1. 사용자 존재 여부 확인
        user = db.query(User).filter(
            User.email == request.email,
            User.social_type == request.social_type
        ).first()
        
        if not user:
            logger.warning(f"사용자를 찾을 수 없음 - 이메일: {request.email}, 소셜 타입: {request.social_type}")
            raise HTTPException(
                status_code=404,
                detail="사용자를 찾을 수 없습니다."
            )
            
        # 2. 전화번호 형식 검증
        clean_phone = request.phone_number.replace("-", "")
        if not re.match(r'^\d{10,11}$', clean_phone):
            logger.error(f"잘못된 전화번호 형식: {request.phone_number}")
            raise HTTPException(
                status_code=400,
                detail="올바른 전화번호 형식이 아닙니다."
            )
            
        # 3. 현재 전화번호와 동일한지 확인
        if user.phone_number == clean_phone:
            return UpdatePhoneResponse(
                success=True,
                message="현재 사용 중인 전화번호와 동일합니다.",
                user={
                    "id": user.id,
                    "email": user.email,
                    "username": user.username,
                    "social_type": user.social_type,
                    "device_id": user.device_id,
                    "is_verified": user.is_verified,
                    "account_type": user.account_type,
                    "phone_number": user.phone_number
                }
            )
            
        # 4. 전화번호 인증 여부 확인
        verification = db.query(PhoneVerification).filter(
            PhoneVerification.phone_number == clean_phone,
            PhoneVerification.is_verified == True,
            PhoneVerification.is_expired == False,
            PhoneVerification.expires_at > datetime.utcnow()
        ).order_by(PhoneVerification.created_at.desc()).first()
        
        if not verification:
            raise HTTPException(
                status_code=400,
                detail="전화번호 인증이 필요합니다. 인증번호를 발송해주세요."
            )
            
        try:
            # 5. 전화번호 업데이트
            user.phone_number = clean_phone
            db.commit()
            db.refresh(user)
            
            logger.info(f"전화번호 변경 완료 - 이메일: {user.email}, 새 전화번호: {user.phone_number}")
            
            # 6. 인증 정보 만료 처리
            verification.is_expired = True
            verification.expired_at = datetime.utcnow()
            db.commit()
            
            return UpdatePhoneResponse(
                success=True,
                message="전화번호가 변경되었습니다.",
                user={
                    "id": user.id,
                    "email": user.email,
                    "username": user.username,
                    "social_type": user.social_type,
                    "device_id": user.device_id,
                    "is_verified": user.is_verified,
                    "account_type": user.account_type,
                    "phone_number": user.phone_number
                }
            )
            
        except Exception as e:
            db.rollback()
            logger.error(f"전화번호 변경 중 오류 발생: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="전화번호 변경 중 오류가 발생했습니다."
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"전화번호 변경 처리 중 예상치 못한 오류 발생: {str(e)}")
        logger.exception("상세 에러 스택 트레이스:")
        raise HTTPException(
            status_code=500,
            detail="전화번호 변경 처리 중 오류가 발생했습니다."
        )

@router.post("/update-password", response_model=UpdatePasswordResponse)
async def update_password(
    request: UpdatePasswordRequest,
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"비밀번호 변경 시도 - 이메일: {request.email}, 소셜 타입: {request.social_type}")
        
        # 1. 사용자 존재 여부 확인
        user = db.query(User).filter(
            User.email == request.email,
            User.social_type == request.social_type
        ).first()
        
        if not user:
            logger.warning(f"사용자를 찾을 수 없음 - 이메일: {request.email}, 소셜 타입: {request.social_type}")
            raise HTTPException(
                status_code=404,
                detail="사용자를 찾을 수 없습니다."
            )
            
        # 2. 소셜 계정인 경우 비밀번호 변경 불가
        if user.account_type == "social":
            raise HTTPException(
                status_code=400,
                detail="소셜 계정은 비밀번호를 변경할 수 없습니다."
            )
            
        # 3. 현재 비밀번호 확인
        if not verify_password(request.current_password, user.hashed_password):
            raise HTTPException(
                status_code=400,
                detail="현재 비밀번호가 일치하지 않습니다."
            )
            
        # 4. 새 비밀번호 유효성 검사
        try:
            validate_password(request.new_password)
        except HTTPException as he:
            raise HTTPException(
                status_code=400,
                detail="새 비밀번호는 8자 이상이며, 숫자와 특수문자를 포함해야 합니다."
            )
            
        try:
            # 5. 새 비밀번호 해시 및 저장
            user.hashed_password = get_password_hash(request.new_password)
            db.commit()
            db.refresh(user)
            
            logger.info(f"비밀번호 변경 완료 - 이메일: {user.email}")
            
            return UpdatePasswordResponse(
                success=True,
                message="비밀번호가 변경되었습니다.",
                user={
                    "id": user.id,
                    "email": user.email,
                    "username": user.username,
                    "social_type": user.social_type,
                    "device_id": user.device_id,
                    "is_verified": user.is_verified,
                    "account_type": user.account_type,
                    "phone_number": user.phone_number
                }
            )
            
        except Exception as e:
            db.rollback()
            logger.error(f"비밀번호 변경 중 오류 발생: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="비밀번호 변경 중 오류가 발생했습니다."
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"비밀번호 변경 처리 중 예상치 못한 오류 발생: {str(e)}")
        logger.exception("상세 에러 스택 트레이스:")
        raise HTTPException(
            status_code=500,
            detail="비밀번호 변경 처리 중 오류가 발생했습니다."
        )