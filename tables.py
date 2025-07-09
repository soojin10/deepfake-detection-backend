from core.database import Base
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON, Text, UniqueConstraint, LargeBinary, ForeignKeyConstraint, Index, Enum
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
import pytz
import enum

def get_korean_time():
    korean_tz = pytz.timezone('Asia/Seoul')
    return datetime.now(korean_tz)

class FileType(enum.Enum):
    video = "video"
    youtube = "youtube"

class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint('email', 'social_type', name='uix_email_social_type'),
    )

    id = Column(Integer, primary_key=True, index=True)  
    email = Column(String(100), index=True)  
    username = Column(String(50))  
    hashed_password = Column(String(255), nullable=True) 
    device_id = Column(String(255), nullable=True)  
    is_active = Column(Boolean, default=True, server_default='1')  # 계정 활성화 상태
    verification_code = Column(String, nullable=True)  
    verification_code_expires_at = Column(DateTime, nullable=True)  # 인증 코드 만료 시간
    is_verified = Column(Boolean, default=False, server_default='0')  # 이메일/전화번호 인증 완료 여부
    created_at = Column(DateTime, default=get_korean_time)  # 계정 생성 시간 (한국 시간)
    social_type = Column(String(20), nullable=True)
    social_id = Column(String(255), nullable=True)   # 소셜 서비스에서 제공하는 고유 ID
    phone_number = Column(String(20), nullable=True)  
    account_type = Column(String(10), nullable=False, server_default='normal')  

    phone_verifications = relationship("PhoneVerification", back_populates="user", cascade="all, delete-orphan")  # 전화번호 인증 기록
    terms_agreements = relationship("UserTermsAgreement", back_populates="user")

class PhoneVerification(Base):
    __tablename__ = "phone_verifications"

    id = Column(Integer, primary_key=True, index=True)  # 전화번호 인증 고유 식별자
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # 관련 사용자 ID
    phone_number = Column(String, nullable=False)  # 인증할 전화번호
    verification_code = Column(String, nullable=False)  # 전송된 인증번호
    created_at = Column(DateTime, default=get_korean_time)  # 인증번호 생성 시간 (한국 시간)
    expires_at = Column(DateTime, nullable=False)  # 인증번호 만료 예정 시간
    expired_at = Column(DateTime, nullable=True)  # 실제 만료된 시간
    verified_at = Column(DateTime, nullable=True)  # 인증 완료 시간
    is_verified = Column(Boolean, default=False)  # 인증 완료 여부
    is_expired = Column(Boolean, default=False)  # 만료 여부

    user = relationship("User", back_populates="phone_verifications")  # 관련 사용자

class UserTermsAgreement(Base):
    __tablename__ = "user_terms_agreements"

    id = Column(Integer, primary_key=True, index=True)  
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # 약관 동의한 사용자의 ID 
    agreed_at = Column(DateTime, default=get_korean_time)  # 약관 동의한 날짜와 시간 (한국 시간)
    terms_type = Column(String(20), nullable=False)  # 약관 종류 (예: "service", "privacy")
    is_required = Column(Boolean, default=True)  # 필수 약관 여부 (True: 필수, False: 선택)
    device_id = Column(String(255))  # 약관 동의 시 사용한 기기의 디바이스 ID

    user = relationship("User", back_populates="terms_agreements")

class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(String(50), primary_key=True, index=True)  # UUID를 저장하기 위한 컬럼 (36자리)
    user_email = Column(String, nullable=False)
    device_id = Column(String, nullable=False)
    social_type = Column(String, nullable=False)
    
    # 파일 관련 정보 (비디오 업로드 시)
    original_file_name = Column(String, nullable=True)
    renamed_file_name = Column(String, nullable=True)
    file_path = Column(String, nullable=False)
    file_url = Column(String, nullable=False)
    file_type = Column(Enum(FileType), nullable=False)  # video or youtube
    file_size = Column(Integer, nullable=True)
    
    # 유튜브 관련 정보
    youtube_url = Column(String, nullable=True)
    youtube_id = Column(String(20), nullable=True)
    video_title = Column(String(255), nullable=True)
    channel_name = Column(String(255), nullable=True)
    
    # 공통 정보
    video_duration = Column(Float, nullable=True)  # 영상 길이(초)
    thumbnail_url = Column(String, nullable=True)
    report_url = Column(String, nullable=True)  # 분석 리포트 URL 추가
    
    # 분석 결과
    confidence = Column(Float, nullable=False, default=0.0)
    analysis_time = Column(Float, nullable=False, default=0.0)
    details = Column(Text)
    status = Column(String(20), nullable=False, default="processing")  # processing, completed, failed, retrying, waiting, post_processing
    progress = Column(Integer, nullable=False, default=0)  # 진행률 (0-100)
    created_at = Column(DateTime, default=get_korean_time)
    updated_at = Column(DateTime, default=get_korean_time, onupdate=get_korean_time)

    detection_details = relationship("DetectionDetail", back_populates="analysis", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_analyses_user', user_email, social_type),
        Index('idx_analyses_youtube_id', youtube_id),
        Index('idx_analyses_created_at', created_at)
    )

class DetectionDetail(Base):
    __tablename__ = "detection_details"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String(36), ForeignKey("analyses.id", ondelete="CASCADE"))
    frame_number = Column(Integer, nullable=True)
    detection_type = Column(String(20), nullable=False)  # upload, youtube
    confidence_score = Column(Float, nullable=False)
    detected_artifacts = Column(Float, nullable=False)
    frame_data = Column(LargeBinary, nullable=True)
    timestamp = Column(DateTime, default=get_korean_time)

    analysis = relationship("Analysis", back_populates="detection_details")

# class AnalysisResult(Base):
#     __tablename__ = "analysis_results"

#     id = Column(Integer, primary_key=True, index=True)
#     analysis_id = Column(String, nullable=False, index=True)  # 분석 세션 ID
#     user_email = Column(String, nullable=False, index=True)  # 사용자 이메일
#     device_id = Column(String, nullable=False)  # 디바이스 ID
#     detection_type = Column(String, nullable=False)  # 탐지 유형 (upload/realtime)
#     file_type = Column(String, nullable=False)  # 파일 유형 (video/image)
#     frame_number = Column(Integer, nullable=False)  # 프레임 번호
#     confidence = Column(Float, nullable=False)  # 신뢰도
#     risk_score = Column(Float, nullable=False)  # 위험도 점수 (0-100)
#     analysis_time = Column(Float, nullable=False)  # 분석 소요 시간
#     details = Column(JSON, nullable=True)  # 상세 정보 (JSON 형식)
#     created_at = Column(DateTime, default=get_korean_time)  # 생성 시간 (한국 시간)

#     __table_args__ = (
#         UniqueConstraint('analysis_id', 'frame_number', name='uix_analysis_frame'),
#     )

#     __mapper_args__ = {
#         'exclude_properties': ['risk_level']  # risk_level 필드 제외
#     }

# 인덱스 생성을 위한 Index 객체들
Index('idx_analyses_user', Analysis.user_email, Analysis.social_type)
Index('idx_analyses_youtube_id', Analysis.youtube_id)
Index('idx_analyses_created_at', Analysis.created_at)