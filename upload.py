# app/api/endpoints/upload.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import logging
import json
import time
import httpx
import requests
from fastapi import status
import cv2
import tempfile
import uuid
from PIL import Image
import io
import asyncio
from urllib.parse import urlparse, parse_qs
import re
import yt_dlp
from core.database import get_db, SessionLocal

from core.config import settings
from models.tables import Analysis, DetectionDetail, User, get_korean_time, FileType
from schemas.user import (
    UploadResponse,
    DeepfakeResult,
    AnalysisResponse,
    AnalysisResultResponse,
    RiskLevel,
    DetectionType,
    AIAnalysisRequest,
    RiskLevelInfo,
    AnalysisListResponse,
    YoutubeAnalysisRequest
)

# 라우터 생성
router = APIRouter(tags=["upload"])
logger = logging.getLogger(__name__)

def get_korean_time():
    korean_tz = pytz.timezone('Asia/Seoul')
    now = datetime.now(korean_tz)
    return now.replace(microsecond=now.microsecond // 1000 * 1000)  # 밀리초 정밀도로 저장


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


# 썸네일 생성 함수 수정
async def generate_thumbnail(video_path: Path, output_path: Path, timestamp: int = 2) -> bool:
    try:
        # 비디오 열기
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False

        # 2초 지점으로 이동
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return False

        # BGR에서 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # PIL Image로 변환
        img = Image.fromarray(frame_rgb)

        # 원본 비율 유지하면서 크기 조정
        max_size = 320
        ratio = min(max_size / img.width, max_size / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

        # WebP로 저장 (품질 80%)
        try:
            img.save(output_path, 'WEBP', quality=80, method=6)
            logger.info(f"썸네일 저장 성공: {output_path}")

            # 파일 크기 체크 (30KB 제한)
            if output_path.stat().st_size > 30 * 1024:  # 30KB
                # 품질을 낮춰서 다시 저장
                img.save(output_path, 'WEBP', quality=60, method=6)
                logger.info(f"썸네일 품질 조정 후 저장: {output_path}")

            return True

        except Exception as e:
            logger.error(f"썸네일 저장 실패: {str(e)}")
            return False

    except Exception as e:
        logger.error(f"썸네일 생성 중 오류: {str(e)}")
        return False


# 영상 파일 검증 함수 수정
async def validate_video_file(file_path: Path) -> Tuple[bool, str, float]:
    """
    영상 파일 기본 검증
    returns: (is_valid, error_message, duration)
    """
    try:
        # 파일 존재 여부 확인
        if not file_path.exists():
            return False, "파일이 존재하지 않습니다.", 0.0

        # 파일 크기 확인
        file_size = file_path.stat().st_size
        if file_size == 0:
            return False, "파일이 비어있습니다.", 0.0

        # 기본적인 영상 정보만 확인
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            return False, "영상 파일을 열 수 없습니다.", 0.0

        # 영상 길이 확인
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        if duration < 5:
            return False, "영상은 최소 5초 이상이어야 합니다.", 0.0

        cap.release()
        return True, "", duration

    except Exception as e:
        logger.error(f"영상 검증 중 오류: {str(e)}")
        return False, f"영상 검증 중 오류 발생: {str(e)}", 0.0


# 영상 파일 저장 함수 수정
async def save_video_file(content: bytes, file_path: Path) -> Tuple[bool, str, float]:
    """
    영상 파일 저장 및 기본 검증
    returns: (success, error_message, duration)
    """
    temp_path = file_path.with_suffix('.temp')
    try:
        # 임시 파일로 저장
        with temp_path.open("wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())  # 디스크에 강제 쓰기

        # 기본 검증
        is_valid, error_msg, duration = await validate_video_file(temp_path)
        if not is_valid:
            return False, error_msg, 0.0

        # 검증 성공 시 최종 파일로 이동
        os.rename(temp_path, file_path)
        return True, "", duration

    except Exception as e:
        logger.error(f"파일 저장 중 오류: {str(e)}")
        if temp_path.exists():
            os.remove(temp_path)
        return False, f"파일 저장 중 오류 발생: {str(e)}", 0.0


def generate_time_based_id() -> str:
    """
    짧은 UUID 기반 ID 생성 함수
    형식: {uuid의 앞 8자리}
    """
    return str(uuid.uuid4())[:8]  # UUID의 앞 8자리만 사용


async def process_detection(analysis_id: str, detection_type: str, social_type: str, db: Session):
    try:
        # 분석 상태 업데이트 (시작)
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
            logger.error(f"분석 ID를 찾을 수 없음: {analysis_id}")
            return
            
        analysis.status = "processing"
        analysis.progress = 0
        db.commit()
        
        # 분석 서버로 요청
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.SERVER_URL}/send-detection",
                params={
                    "analysis_id": analysis_id,
                    "detection_type": detection_type,
                    "social_type": social_type
                }
            )
            
            if response.status_code == 200:
                # 분석 완료 처리
                analysis.status = "completed"
                analysis.progress = 100
                analysis.updated_at = get_korean_time()
                
                # 분석 완료 메시지 로깅
                logger.info(f"분석 완료 - ID: {analysis_id}, 타입: {detection_type}")
                
                # 분석 결과 데이터 가져오기
                result_data = response.json() if response.headers.get("content-type") == "application/json" else {}
                
                # 분석 결과 업데이트
                if result_data:
                    try:
                        # 기존 details 업데이트
                        current_details = json.loads(analysis.details) if analysis.details else {}
                        current_details.update(result_data.get("details", {}))
                        analysis.details = json.dumps(current_details)
                        
                        # confidence 점수 업데이트
                        if "confidence" in result_data:
                            analysis.confidence = float(result_data["confidence"])
                            
                        # 분석 시간 업데이트
                        if "analysis_time" in result_data:
                            analysis.analysis_time = float(result_data["analysis_time"])
                            
                        # report_url은 그대로 저장 (분석서버에서 받은 URL 그대로 사용)
                        analysis.report_url = result_data.get("report_url")
                            
                        db.commit()
                        logger.info(f"분석 결과 업데이트 완료 - ID: {analysis_id}")
                    except Exception as e:
                        logger.error(f"분석 결과 업데이트 중 오류: {str(e)}")
            else:
                analysis.status = "failed"
                analysis.updated_at = get_korean_time()
                db.commit()
                logger.error(f"분석 실패 - ID: {analysis_id}, 상태 코드: {response.status_code}")
                
    except Exception as e:
        logger.error(f"분석 처리 중 오류: {str(e)}")
        if analysis:
            # 일시적인 오류인 경우 retrying으로 설정
            if isinstance(e, (httpx.TimeoutException, httpx.RequestError, ConnectionError)):
                analysis.status = "retrying"
            else:
                analysis.status = "failed"
            analysis.updated_at = get_korean_time()
            db.commit()
            logger.error(f"분석 상태 업데이트 완료 - ID: {analysis_id}, 상태: {analysis.status}")


# 일반 영상 업로드 분석
@router.post("/upload/", response_model=UploadResponse)
async def upload_file(
        video: UploadFile = File(...),
        user_email: str = Form(...),
        device_id: str = Form(...),
        social_type: str = Form(...),
        background_tasks: BackgroundTasks = BackgroundTasks(),
        db: Session = Depends(get_db)
):
    try:
        logger.info(f"파일 업로드 시작 - 사용자: {user_email}, 파일: {video.filename}, 소셜타입: {social_type}")

        # 분석 ID를 가장 먼저 생성
        analysis_id = generate_time_based_id()
        
        # 분석 요청을 비동기로 처리
        background_tasks.add_task(
            process_detection,
            analysis_id=analysis_id,
            detection_type="upload",
            social_type=social_type,
            db=db
        )
        
        # 분석 ID를 즉시 반환하기 위한 응답 객체 생성
        response = UploadResponse(
            success=True,
            message="영상이 업로드되었으며 분석이 시작되었습니다.",
            result={
                "analysis_id": analysis_id,
                "status": "processing",
                "progress": 0
            }
        )

        # 사용자 정보 조회하여 social_type 검증
        user = db.query(User).filter(
            User.email == user_email,
            User.social_type == social_type
        ).first()

        if not user:
            raise HTTPException(
                status_code=404,
                detail="사용자를 찾을 수 없습니다."
            )

        # 파일 타입 검증
        if not video.content_type.startswith('video/'):
            raise HTTPException(
                status_code=400,
                detail="비디오 파일만 업로드 가능합니다."
            )

        # 파일 저장 경로 생성
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # 파일명 생성 (analysis_id + 원본 파일명)
        new_filename = f"{analysis_id}_{video.filename}"
        file_path = upload_dir / new_filename
        
        # 파일 저장 및 검증
        content = await video.read()
        success, error_msg, duration = await save_video_file(content, file_path)

        if not success:
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )

        # 썸네일 생성 및 저장
        thumbnail_dir = Path("thumbnails")
        thumbnail_dir.mkdir(exist_ok=True)

        thumbnail_filename = f"{analysis_id}_thumb.webp"
        thumbnail_path = thumbnail_dir / thumbnail_filename
        thumbnail_success = await generate_thumbnail(file_path, thumbnail_path)

        # 썸네일 URL 설정 (전체 URL로 저장)
        thumbnail_url = f"{settings.SERVER_URL}/thumbnails/{thumbnail_filename}" if thumbnail_success else None

        analysis = None
        try:
            current_time = get_korean_time()

            # Analysis 객체 생성
            analysis = Analysis(
                id=analysis_id,
                user_email=user_email,
                device_id=device_id,
                social_type=social_type,
                original_file_name=video.filename,  
                renamed_file_name=new_filename,
                file_path=str(file_path),
                file_url=f"/uploads/{new_filename}",  
                file_type=FileType.video,
                file_size=len(content),
                video_duration=duration,
                confidence=0.0,
                analysis_time=0.0,
                status="processing",
                progress=0,
                details=json.dumps({
                    "frames": {
                        "0": {
                            "confidence": 0.0,
                            "artifacts": 0.0
                        }
                    }
                }),
                thumbnail_url=thumbnail_url,
                created_at=current_time
            )
            
            db.add(analysis)
            db.flush()

            # DetectionDetail 객체 생성
            initial_detection = DetectionDetail(
                analysis_id=analysis_id,
                frame_number=0,
                detection_type="upload",
                confidence_score=0.0,
                detected_artifacts=0.0,
                timestamp=current_time
            )

            db.add(initial_detection)
            db.commit()
            db.refresh(analysis)

            return response

        except Exception as e:
            logger.error(f"분석 처리 중 오류: {str(e)}")
            if analysis:
                # 일시적인 오류인 경우 retrying으로 설정
                if isinstance(e, (httpx.TimeoutException, httpx.RequestError, ConnectionError)):
                    analysis.status = "retrying"
                else:
                    analysis.status = "failed"
                analysis.updated_at = get_korean_time()
                db.commit()
                logger.error(f"분석 상태 업데이트 완료 - ID: {analysis_id}, 상태: {analysis.status}")

            # 파일 삭제
            if file_path.exists():
                os.remove(file_path)

            raise HTTPException(
                status_code=500,
                detail="분석 처리 중 오류가 발생했습니다."
            )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"파일 업로드 중 오류: {str(e)}")
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail="파일 업로드 중 오류가 발생했습니다."
        )


# 사용자 분석 기록 조회
@router.get("/analyses", response_model=List[AnalysisListResponse])
async def get_user_analyses(
        user_email: str,
        social_type: str,
        db: Session = Depends(get_db)
):
    try:
        # 전체 분석 기록 수 확인
        total_analyses = db.query(Analysis).filter(
            Analysis.user_email == user_email,
            Analysis.social_type == social_type
        ).count()
        
        # 60건을 초과하는 경우 오래된 기록 삭제
        if total_analyses > 60:
            logger.info(f"분석 기록 60건 초과 - 사용자: {user_email}, 현재: {total_analyses}건")
            # 가장 오래된 기록부터 삭제할 개수 계산
            delete_count = total_analyses - 60
            
            # 삭제할 분석 ID 목록 조회
            old_analyses = db.query(Analysis.id).filter(
                Analysis.user_email == user_email,
                Analysis.social_type == social_type
            ).order_by(Analysis.created_at.asc()).limit(delete_count).all()
            
            old_analysis_ids = [analysis.id for analysis in old_analyses]
            
            if old_analysis_ids:
                # 연관된 detection_details 삭제
                db.query(DetectionDetail).filter(
                    DetectionDetail.analysis_id.in_(old_analysis_ids)
                ).delete(synchronize_session=False)
                
                # 분석 기록 삭제
                db.query(Analysis).filter(
                    Analysis.id.in_(old_analysis_ids)
                ).delete(synchronize_session=False)
                
                db.commit()
                logger.info(f"오래된 분석 기록 {delete_count}건 삭제 완료 - 사용자: {user_email}")
        
        # 최신 60건의 분석 기록 조회
        analyses = db.query(Analysis).filter(
            Analysis.user_email == user_email,
            Analysis.social_type == social_type
        ).order_by(Analysis.created_at.desc()).limit(60).all()

        response_list = []
        for analysis in analyses:
            # 위험도 점수 계산
            details = json.loads(analysis.details) if analysis.details else {}
            score = details.get("score", 0.0)
            risk_level = calculate_risk_level(score)
            
            # 파일 확장자 추출
            file_ext = None
            if analysis.original_file_name and "." in analysis.original_file_name:
                file_ext = analysis.original_file_name.split(".")[-1].lower()

            # 썸네일 URL을 전체 URL로 변환
            thumbnail_url = None
            if analysis.thumbnail_url:
                # 이미 전체 URL인 경우 그대로 사용
                if analysis.thumbnail_url.startswith('http'):
                    thumbnail_url = analysis.thumbnail_url
                else:
                    # 상대 경로를 전체 URL로 변환
                    thumbnail_url = f"{settings.SERVER_URL}{analysis.thumbnail_url}"

            # 응답 데이터 구성
            response_data = {
                "id": analysis.id,
                "created_at": analysis.created_at,
                "confidence_percent": round(analysis.confidence * 100, 2),
                "risk_level_info": RiskLevelInfo(
                    level=risk_level,
                    score=int(score)
                ),
                "file_size": analysis.file_size,
                "video_duration": analysis.video_duration,
                "thumbnail_url": thumbnail_url,  # 변환된 전체 URL 사용
                "report_url": analysis.report_url,  # report_url 추가
                "status": analysis.status
            }

            # 파일 타입에 따라 다른 정보 추가
            if analysis.file_type == FileType.video:
                response_data.update({
                    "original_file_name": analysis.original_file_name,
                    "file_ext": file_ext,
                    "youtube_url": None,
                    "video_title": None,
                    "channel_name": None
                })
            else:  # FileType.youtube
                response_data.update({
                    "original_file_name": None,
                    "file_ext": None,
                    "youtube_url": analysis.youtube_url,
                    "video_title": analysis.video_title,
                    "channel_name": analysis.channel_name
                })

            response_list.append(response_data)

        return response_list

    except Exception as e:
        logger.error(f"분석 기록 조회 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="분석 기록 조회 중 오류가 발생했습니다."
        )


# 분석 기록 삭제
@router.delete("/analyses/{analysis_id}")
async def delete_analysis(
        analysis_id: str,
        user_email: str,
        social_type: str,
        db: Session = Depends(get_db)
):
    try:
        logger.info("=== 분석 기록 삭제 요청 시작 ===")
        logger.info(f"요청 파라미터: id={analysis_id}, email={user_email}, social_type={social_type}")

        # 1. 사용자 존재 여부 확인
        user_query = db.query(User).filter(User.email == user_email)

        # social_type이 normal이면 social_type이 'normal'이고 account_type도 'normal'인 경우 검색
        if social_type == "normal":
            user_query = user_query.filter(
                User.social_type == "normal",
                User.account_type == "normal"
            )
        else:
            # 소셜 로그인 계정 (naver, google, kakao)
            user_query = user_query.filter(
                User.social_type == social_type,
                User.account_type == "social"
            )

        logger.info(f"사용자 조회 쿼리: {str(user_query.statement.compile(compile_kwargs={'literal_binds': True}))}")

        user = user_query.first()
        if not user:
            logger.warning(f"사용자를 찾을 수 없음: email={user_email}, social_type={social_type}")
            raise HTTPException(
                status_code=404,
                detail="사용자를 찾을 수 없습니다."
            )
        logger.info(
            f"사용자 확인됨: id={user.id}, email={user.email}, social_type={user.social_type}, account_type={user.account_type}")

        # 2. 분석 기록 조회
        analysis_query = db.query(Analysis).filter(
            Analysis.id == analysis_id,
            Analysis.user_email == user_email,
            Analysis.social_type == social_type  # social_type이 정확히 일치하는 경우만 검색
        )

        logger.info(f"분석 기록 조회 쿼리: {str(analysis_query.statement.compile(compile_kwargs={'literal_binds': True}))}")

        analysis = analysis_query.first()
        if not analysis:
            logger.warning(f"분석 기록을 찾을 수 없음: id={analysis_id}, email={user_email}, social_type={social_type}")

            # 디버깅을 위해 ID로만 조회
            analysis_by_id = db.query(Analysis).filter(Analysis.id == analysis_id).first()
            if analysis_by_id:
                logger.warning(f"ID로만 조회한 결과:")
                logger.warning(f"- user_email: {analysis_by_id.user_email}")
                logger.warning(f"- social_type: {analysis_by_id.social_type}")
            else:
                logger.warning(f"ID {analysis_id}로 조회된 데이터 없음")

            raise HTTPException(
                status_code=404,
                detail="분석 기록을 찾을 수 없습니다."
            )
        logger.info(f"분석 기록 확인됨: id={analysis.id}, file_path={analysis.file_path}")

        # 3. 파일 경로 검증
        file_path = analysis.file_path
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"파일이 존재하지 않음: {file_path}")
            file_path = None
        else:
            logger.info(f"파일 존재 확인: {file_path}")

        try:
            # 4. DB에서 분석 기록 삭제 (CASCADE로 DetectionDetail도 자동 삭제)
            db.delete(analysis)
            logger.info("DB에서 분석 기록 삭제 완료")

            # 5. 파일 시스템에서 파일 삭제
            if file_path:
                try:
                    os.remove(file_path)
                    logger.info(f"파일 삭제 완료: {file_path}")
                except Exception as e:
                    logger.error(f"파일 삭제 실패: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail="파일 삭제 중 오류가 발생했습니다."
                    )

            # 6. 트랜잭션 커밋
            db.commit()
            logger.info("트랜잭션 커밋 완료")

            logger.info(f"분석 기록 삭제 완료 - ID: {analysis_id}, 사용자: {user_email}")
            return {
                "success": True,
                "message": "분석 기록이 삭제되었습니다."
            }

        except HTTPException as he:
            db.rollback()
            logger.error(f"HTTP 예외 발생: {str(he)}")
            raise he
        except Exception as e:
            db.rollback()
            logger.error(f"삭제 중 오류 발생: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="분석 기록 삭제 중 오류가 발생했습니다."
            )

    except HTTPException as he:
        logger.error(f"HTTP 예외 발생: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"분석 기록 삭제 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="분석 기록 삭제 중 오류가 발생했습니다."
        )
    finally:
        logger.info("=== 분석 기록 삭제 요청 종료 ===")

# YouTube URL 검증 함수 수정
def is_valid_youtube_url(url: str) -> Tuple[bool, str, Optional[str]]:
    """
    YouTube URL 검증
    returns: (is_valid, error_message, video_id)
    """
    try:
        # URL 패턴 검사
        youtube_regex = (
            r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
            r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
        )
        
        match = re.match(youtube_regex, url)
        if not match:
            return False, "올바른 YouTube URL이 아닙니다.", None
            
        video_id = match.group(6)
        
        # yt-dlp를 사용하여 영상 정보 확인
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # 영상 길이 확인 (초 단위)
            duration = info.get('duration', 0)
            if duration < 5:
                return False, "영상은 최소 5초 이상이어야 합니다.", None
                
            if duration > 240:  # 4분 제한
                return False, "4분 이하의 영상만 분석 가능합니다.", None
                
            return True, "", video_id
            
    except Exception as e:
        logger.error(f"YouTube URL 검증 중 오류: {str(e)}")
        return False, "YouTube 영상 정보를 가져올 수 없습니다.", None

# YouTube URL 검증 엔드포인트 수정
@router.get("/youtube/validate")
async def validate_youtube_url(url: str):
    try:
        # yt-dlp를 사용하여 영상 정보 확인
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # 영상 길이 확인 (초 단위)
            duration = info.get('duration', 0)
            if duration < 5:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": "영상은 최소 5초 이상이어야 합니다."
                    }
                )
                
            if duration > 240:  # 4분 제한
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": "4분 이하의 영상만 분석 가능합니다."
                    }
                )
                
            # 응답 데이터 구성
            return {
                "success": True,
                "data": {
                    "title": info.get('title', ''),
                    "duration": duration,
                    "thumbnail": info.get('thumbnail', ''),
                    "channel_name": info.get('uploader', '')
                }
            }
            
    except Exception as e:
        logger.error(f"YouTube URL 검증 중 오류: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "유효하지 않은 YouTube URL입니다."
            }
        )

# YouTube 영상 다운로드 함수 수정
async def download_youtube_video(url: str, output_path: Path) -> Tuple[bool, dict]:
    """
    YouTube 영상을 다운로드하고 메타데이터를 반환합니다.
    returns: (success, metadata)
    """
    try:
        ydl_opts = {
            'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best',  # format 설정 변경
            'outtmpl': str(output_path),
            'quiet': True,
            'no_warnings': True,
            'noprogress': True,
            'no_color': True,
            'ignoreerrors': True,
            'no_deprecation_warning': True  # deprecation 경고 무시
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            if info is None:
                logger.error("영상 정보를 가져올 수 없습니다.")
                return False, {}
                
            metadata = {
                'title': info.get('title', ''),
                'duration': info.get('duration', 0),
                'thumbnail': info.get('thumbnail', ''),
                'channel_name': info.get('uploader', ''),
                'file_size': os.path.getsize(output_path) if os.path.exists(output_path) else 0
            }
            
            if metadata['file_size'] == 0:
                logger.error("다운로드된 파일이 비어있습니다.")
                return False, {}
                
            return True, metadata
            
    except Exception as e:
        logger.error(f"YouTube 영상 다운로드 중 오류: {str(e)}")
        return False, {}

# YouTube 영상 분석 
@router.post("/youtube/", response_model=UploadResponse)
async def analyze_youtube_video(
    request: YoutubeAnalysisRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"YouTube 분석 시작 - 사용자: {request.user_email}, URL: {request.youtube_url}")

        # URL 검증
        is_valid, error_msg, video_id = is_valid_youtube_url(request.youtube_url)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )

        # 사용자 검증
        user = db.query(User).filter(
            User.email == request.user_email,
            User.social_type == request.social_type
        ).first()

        if not user:
            raise HTTPException(
                status_code=404,
                detail="사용자를 찾을 수 없습니다."
            )

        # 분석 ID를 가장 먼저 생성
        analysis_id = generate_time_based_id()

        # 분석 요청을 비동기로 처리
        background_tasks.add_task(
            process_detection,
            analysis_id=analysis_id,
            detection_type="youtube",
            social_type=request.social_type,
            db=db
        )

        # 분석 ID를 즉시 반환하기 위한 응답 객체 생성
        response = UploadResponse(
            success=True,
            message="영상이 업로드되었으며 분석이 시작되었습니다.",
            result={
                "analysis_id": analysis_id,
                "status": "processing",
                "progress": 0
            }
        )

        # 파일 저장 경로 생성
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)

        # 원본 파일명 생성 (YouTube ID + .mp4)
        original_filename = f"{video_id}.mp4"
        
        # 리네임된 파일명 생성 (analysis_id + YouTube ID.mp4)
        new_filename = f"{analysis_id}_{original_filename}"
        file_path = upload_dir / new_filename

        # YouTube 영상 다운로드
        success, metadata = await download_youtube_video(request.youtube_url, file_path)
        if not success:
            raise HTTPException(
                status_code=500,
                detail="YouTube 영상 다운로드에 실패했습니다."
            )

        # 상대 URL 생성 (renamed_file_name 사용)
        relative_url = f"/uploads/{new_filename}"
        
        # 썸네일 URL을 전체 URL로 저장
        thumbnail_url = metadata['thumbnail']  # YouTube 썸네일은 이미 전체 URL

        analysis = None
        try:
            current_time = get_korean_time()

            # Analysis 객체 생성
            analysis = Analysis(
                id=analysis_id,
                user_email=request.user_email,
                device_id=request.device_id,
                social_type=request.social_type,
                original_file_name=original_filename,  # YouTube ID를 원본 파일명으로
                renamed_file_name=new_filename,        # 타임스탬프가 포함된 새 파일명
                file_path=str(file_path),
                file_url=relative_url,
                file_type=FileType.youtube,
                file_size=metadata['file_size'],
                video_duration=metadata['duration'],
                youtube_url=request.youtube_url,
                youtube_id=video_id,
                video_title=metadata['title'],
                channel_name=metadata['channel_name'],
                thumbnail_url=thumbnail_url,
                confidence=0.0,
                analysis_time=0.0,
                status="processing",
                progress=0,
                details=json.dumps({
                    "frames": {
                        "0": {
                            "confidence": 0.0,
                            "artifacts": 0.0
                        }
                    }
                }),
                created_at=current_time
            )

            db.add(analysis)
            db.flush()

            # DetectionDetail 객체 생성
            initial_detection = DetectionDetail(
                analysis_id=analysis_id,
                frame_number=0,
                detection_type="youtube",
                confidence_score=0.0,
                detected_artifacts=0.0,
                timestamp=current_time
            )

            db.add(initial_detection)
            db.commit()
            db.refresh(analysis)

            return response

        except Exception as e:
            logger.error(f"분석 처리 중 오류: {str(e)}")
            if analysis:
                # 일시적인 오류인 경우 retrying으로 설정
                if isinstance(e, (httpx.TimeoutException, httpx.RequestError, ConnectionError)):
                    analysis.status = "retrying"
                else:
                    analysis.status = "failed"
                analysis.updated_at = get_korean_time()
                db.commit()
                logger.error(f"분석 상태 업데이트 완료 - ID: {analysis_id}, 상태: {analysis.status}")

            # 파일 삭제
            if file_path.exists():
                os.remove(file_path)

            raise HTTPException(
                status_code=500,
                detail="분석 처리 중 오류가 발생했습니다."
            )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"YouTube 분석 중 오류: {str(e)}")
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail="YouTube 영상 분석 중 오류가 발생했습니다."
        )

# 분석 결과 조회 엔드포인트 추가
@router.get("/analysis/{analysis_id}", response_model=AnalysisResultResponse)
async def get_analysis_result(
    analysis_id: str,
    user_email: str,
    social_type: str,
    db: Session = Depends(get_db)
):
    try:
        logger.info("===== 분석 결과 조회 시작 =====")
        logger.info("요청 파라미터:")
        logger.info(f"- analysis_id: {analysis_id}")
        logger.info(f"- user_email: {user_email}")
        logger.info(f"- social_type: {social_type}")

        logger.info("트랜잭션 시작")
        # 분석 기록 조회
        logger.info(f"분석 기록 조회 시도 - ID: {analysis_id}")
        analysis = db.query(Analysis).filter(
            Analysis.id == analysis_id,
            Analysis.user_email == user_email,
            Analysis.social_type == social_type
        ).first()

        if not analysis:
            logger.warning(f"분석 기록을 찾을 수 없음 - ID: {analysis_id}")
            raise HTTPException(
                status_code=404,
                detail="분석 기록을 찾을 수 없습니다."
            )

        logger.info("분석 기록 조회 성공:")
        logger.info(f"- ID: {analysis.id}")
        logger.info(f"- user_email: {analysis.user_email}")
        logger.info(f"- social_type: {analysis.social_type}")
        logger.info(f"- status: {analysis.status}")
        logger.info(f"- created_at: {analysis.created_at}")
        logger.info(f"- updated_at: {analysis.updated_at}")

        # 분석이 아직 진행 중인 경우
        if analysis.status == "processing":
            logger.info("분석 상태: processing")
            return {
                "id": analysis.id,
                "user_email": analysis.user_email,
                "device_id": analysis.device_id,
                "confidence": analysis.confidence,
                "confidence_percent": round(analysis.confidence * 100, 2),
                "risk_level_info": RiskLevelInfo(
                    level=RiskLevel.VERY_SAFE,
                    score=0
                ),
                "analysis_time": analysis.analysis_time,
                "details": analysis.details,
                "file_url": analysis.file_url,
                "report_url": analysis.report_url,
                "created_at": analysis.created_at,
                "status": "processing"
            }

        # 분석이 실패한 경우
        if analysis.status == "failed":
            logger.warning("분석 상태: failed")
            raise HTTPException(
                status_code=500,
                detail="분석 처리에 실패했습니다."
            )

        # 분석이 완료된 경우
        logger.info("분석 상태: completed")
        logger.info("details 필드 파싱 시도")
        details = json.loads(analysis.details) if analysis.details else {}
        score = details.get("score", 0.0)
        risk_level = calculate_risk_level(score)
        logger.info("details 파싱 성공")

        logger.info("응답 데이터 생성")
        response_data = {
            "id": analysis.id,
            "user_email": analysis.user_email,
            "device_id": analysis.device_id,
            "confidence": analysis.confidence,
            "confidence_percent": round(analysis.confidence * 100, 2),
            "risk_level_info": RiskLevelInfo(
                level=risk_level,
                score=int(score)
            ),
            "analysis_time": analysis.analysis_time,
            "details": analysis.details,
            "file_url": analysis.file_url,
            "report_url": analysis.report_url,
            "created_at": analysis.created_at,
            "status": analysis.status
        }

        logger.info("트랜잭션 커밋")
        logger.info("===== 분석 결과 조회 종료 =====")
        logger.info("최종 응답:")
        logger.info(f"- analysis_id: {response_data['id']}")
        logger.info(f"- status: {response_data['status']}")
        logger.info(f"- created_at: {response_data['created_at']}")
        logger.info(f"- updated_at: {analysis.updated_at}")
        logger.info("===== 분석 결과 조회 종료 =====")

        return response_data

    except HTTPException as he:
        logger.error(f"HTTP 예외 발생: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"분석 결과 조회 중 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="분석 결과 조회 중 오류가 발생했습니다."
        )

