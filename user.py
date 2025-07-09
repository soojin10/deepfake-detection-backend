from pydantic import BaseModel, EmailStr, constr
from typing import Optional, Literal, Dict, Any, Tuple
from datetime import datetime
from pydantic import validator
from enum import Enum

class UserBase(BaseModel):
    email: str
    username: str
    account_type: Literal["normal", "social"]

class UserCreate(UserBase):
    password: str
    confirm_password: str
    device_id: Optional[str] = None
    phone_number: str
    service_terms_agreed: bool  # 서비스 이용약관
    privacy_terms_agreed: bool  # 개인정보 처리방침
    media_learning_terms_agreed: bool  # 미디어 컨텐츠 학습 동의

    @validator('service_terms_agreed', 'privacy_terms_agreed')
    def terms_must_be_agreed(cls, v):
        if not v:
            raise ValueError('필수 약관에 동의해야 합니다.')
        return v

class SocialSignUpRequest(UserBase):
    social_type: str
    social_id: str
    device_id: Optional[str] = None
    phone_number: str
    service_terms_agreed: bool  # 서비스 이용약관
    privacy_terms_agreed: bool  # 개인정보 처리방침
    media_learning_terms_agreed: bool  # 미디어 컨텐츠 학습 동의

    @validator('service_terms_agreed', 'privacy_terms_agreed')
    def terms_must_be_agreed(cls, v):
        if not v:
            raise ValueError('필수 약관에 동의해야 합니다.')
        return v

class LoginRequest(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    device_id: Optional[str] = None
    is_verified: bool
    social_type: str
    phone_number: Optional[str] = None
    account_type: str
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

    class Config:
        from_attributes = True

class TokenData(BaseModel):
    email: Optional[str] = None

class EmailVerification(BaseModel):
    email: str
    verification_code: str

class CheckUserResponse(BaseModel):
    exists: bool
    message: str

# 전화번호 인증번호 전송 요청/응답 스키마
class SendPhoneVerificationRequest(BaseModel):
    phone_number: str

class SendPhoneVerificationResponse(BaseModel):
    message: str

# 전화번호 인증번호 확인 요청/응답 스키마
class VerifyPhoneRequest(BaseModel):
    phone_number: str
    verification_code: str

class VerifyPhoneResponse(BaseModel):
    message: str
    verified: bool

class EmailCheck(BaseModel):
    email: EmailStr
    account_type: Literal["normal", "social"]

class SocialLoginRequest(BaseModel):
    email: str
    social_type: str
    social_id: str
    device_id: str

class AnalysisResponse(BaseModel):
    id: str  # UUID를 저장하기 위한 str 타입
    user_id: Optional[int]
    user_email: str
    device_id: str
    original_file_name: str
    renamed_file_name: str
    file_path: str
    file_url: str
    confidence: float
    confidence_percent: float
    analysis_time: float
    details: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class DetectionType(str, Enum):
    UPLOAD = "upload"
    YOUTUBE = "youtube"
    REALTIME = "realtime"

class FileType(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    RTSP = "rtsp"

class AIAnalysisRequest(BaseModel):
    analysis_id: str
    user_email: str
    device_id: str
    file_url: str
    detection_type: DetectionType
    file_type: FileType = FileType.VIDEO  # 기본값으로 VIDEO 설정

class AnalysisDetails(BaseModel):
    deepfake_score: float
    confidence: float
    rnn_score: float
    gan_score: float
    cnn_score: float
    frequency_score: float
    temporal_consistency: float
    processing_time: float
    frame_count: int
    mode: str
    weights: Dict[str, float]

class AIAnalysisResponse(BaseModel):
    analysis_id: str
    user_email: str
    device_id: str
    confidence: float
    analysis_time: float
    detection_type: DetectionType
    details: AnalysisDetails
    file_url: str
    report_url: str

class RiskLevel(str, Enum):
    VERY_SAFE = "매우안전"
    SAFE = "안전"
    NORMAL = "보통"
    DANGEROUS = "위험"
    VERY_DANGEROUS = "매우위험"

class RiskLevelInfo(BaseModel):
    level: RiskLevel
    score: int  # 실제 점수값

class AnalysisResultResponse(BaseModel):
    id: str  # UUID를 저장하기 위한 str 타입
    user_email: str
    device_id: str
    confidence: float
    confidence_percent: float
    risk_level_info: RiskLevelInfo
    analysis_time: float
    details: Optional[str]
    file_url: str
    report_url: str
    created_at: datetime
    status: str

    class Config:
        from_attributes = True

class DeepfakeResult(BaseModel):
    confidence: float
    details: Optional[Dict[str, Any]] = {}
    risk_level_info: RiskLevelInfo

class UploadResult(BaseModel):
    analysis_id: Optional[str] = None
    status: Optional[str] = None
    progress: Optional[int] = 0  # 진행률 (0-100)
    confidence: Optional[float] = None
    details: Optional[Dict[str, Any]] = {}
    risk_level_info: Optional[RiskLevelInfo] = None

class UploadResponse(BaseModel):
    success: bool
    message: str
    result: Optional[UploadResult] = None

    class Config:
        from_attributes = True

class UploadRequest(BaseModel):
    file_name: str
    file_type: str

class FrameAnalysisResult(BaseModel):
    confidence: float
    risk_level_info: RiskLevelInfo
    timestamp: int

class FrameAnalysisResponse(BaseModel):
    success: bool
    message: str
    result: Optional[FrameAnalysisResult] = None

class AnalysisListResponse(BaseModel):
    id: str  # UUID를 저장하기 위한 str 타입
    created_at: datetime  # 날짜/시간
    original_file_name: Optional[str] = None  # 파일 이름
    file_ext: Optional[str] = None  # 파일 확장자
    file_size: int  # 파일 크기
    video_duration: float  # 영상 길이
    risk_level_info: RiskLevelInfo  # 위험도 정보 (위험도, 위험점수)
    confidence_percent: float  # 신뢰도 (0-100%)
    thumbnail_url: Optional[str] = None  # 썸네일 URL
    youtube_url: Optional[str] = None  # YouTube URL
    video_title: Optional[str] = None  # 영상 제목
    channel_name: Optional[str] = None  # 채널 이름
    status: Optional[str] = None  # 분석 상태
    report_url: Optional[str] = None  # 분석 리포트 URL
    
    class Config:
        from_attributes = True

# YouTube 관련 스키마 추가
class YoutubeAnalysisRequest(BaseModel):
    youtube_url: str
    user_email: str
    device_id: str
    social_type: str

class YoutubeUrlValidationResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

    class Config:
        from_attributes = True

class YoutubeAnalysisResponse(BaseModel):
    id: str
    created_at: datetime
    youtube_url: str
    youtube_id: str
    video_title: str
    channel_name: Optional[str]
    original_duration: float
    file_size: int
    risk_level_info: RiskLevelInfo
    confidence_percent: float
    thumbnail_url: Optional[str] = None
    status: str

    class Config:
        from_attributes = True

class YoutubeAnalysisListResponse(BaseModel):
    id: str
    created_at: datetime
    video_title: str
    channel_name: Optional[str]
    youtube_url: str
    file_size: int
    video_duration: float
    risk_level_info: RiskLevelInfo
    confidence_percent: float
    thumbnail_url: Optional[str] = None
    status: str
    
    class Config:
        from_attributes = True

class ProcessData(BaseModel):
    analysis_id: str
    stage: str  # extract_frames, analyze_frame, post_processing, completed
    progress: str
    message: str
    current: str
    total: str

    class Config:
        from_attributes = True

class UpdateUsernameRequest(BaseModel):
    email: str
    social_type: str
    new_username: str

class UpdateUsernameResponse(BaseModel):
    success: bool
    message: str
    user: Optional[UserResponse] = None

class UpdatePhoneRequest(BaseModel):
    email: str
    social_type: str
    phone_number: str

class UpdatePhoneResponse(BaseModel):
    success: bool
    message: str
    user: Optional[UserResponse] = None

class UpdatePasswordRequest(BaseModel):
    email: str
    social_type: str
    current_password: str
    new_password: str

class UpdatePasswordResponse(BaseModel):
    success: bool
    message: str
    user: Optional[UserResponse] = None
