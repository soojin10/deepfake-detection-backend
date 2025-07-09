from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from core.database import get_db
from models.tables import Analysis
from schemas.user import ProcessData
import logging

# 라우터 생성
router = APIRouter()

# 로깅 설정
logger = logging.getLogger(__name__)

# 진행률 정보 저장을 위한 딕셔너리
# {analysis_id: ProcessData}
process_status = {}

@router.post("/process")
async def update_process_status(request: Request, db: Session = Depends(get_db)):
    """
    분석 서버로부터 진행률 정보를 받아 저장하는 엔드포인트
    """
    try:
        # 원본 요청 데이터 로깅
        raw_body = await request.body()
        raw_str = raw_body.decode('utf-8')
        
        # Python 딕셔너리 형식을 JSON 형식으로 변환
        if raw_str.startswith('{') and "'" in raw_str:
            import ast
            try:
                python_dict = ast.literal_eval(raw_str)
                import json
                raw_str = json.dumps(python_dict)
                logger.info(f"분석 서버 진행률: {python_dict['analysis_id']} - {python_dict['stage']} - {python_dict['current']}/{python_dict['total']} - {python_dict.get('message', '')} - 진행률: {python_dict.get('progress', '')}%")
            except Exception as e:
                logger.error(f"딕셔너리 변환 실패: {str(e)}")
                raise HTTPException(
                    status_code=422,
                    detail="데이터 형식 변환에 실패했습니다."
                )
        
        # JSON 파싱 및 데이터 검증
        import json
        raw_data = json.loads(raw_str)
        data = ProcessData(**raw_data)
        
        # 분석 ID 존재 여부 확인
        analysis = db.query(Analysis).filter(Analysis.id == data.analysis_id).first()
        if not analysis:
            logger.warning(f"존재하지 않는 분석 ID: {data.analysis_id}")
            raise HTTPException(
                status_code=404,
                detail="분석을 찾을 수 없습니다."
            )

        # 진행률 계산 (0-100 사이의 값)
        try:
            current = int(data.current)
            total = int(data.total)
            progress = int((current / total) * 100) if total > 0 else 0
            progress = min(max(progress, 0), 100)  # 0-100 사이로 제한
        except (ValueError, ZeroDivisionError):
            progress = 0

        # 단계별 상태 및 진행률 처리
        if data.stage == "extract_frames":
            analysis.status = "waiting"
            analysis.progress = 0
        elif data.stage == "analyze_frame":
            analysis.status = "processing"  # analyze_frame 단계는 processing 상태로 설정
            analysis.progress = progress    # 실제 진행률 업데이트
        elif data.stage == "post_processing":
            analysis.status = "post_processing"  # 후처리 단계 추가
            analysis.progress = 100  # 후처리 중에는 100% 유지
        elif data.stage == "error":
            # 오류 메시지에 따라 상태 구분
            error_message = data.message.lower()
            if any(keyword in error_message for keyword in ["파일 손상", "지원하지 않는 형식", "분석 불가능"]):
                analysis.status = "failed"
            else:
                analysis.status = "retrying"
            analysis.progress = progress

        db.commit()

        # 메모리에 진행률 정보 저장
        process_status[data.analysis_id] = data

        # 로그 메시지 수정
        if data.stage == "extract_frames":
            logger.info(f"프레임 추출 중 - 분석ID: {data.analysis_id}, 단계: {data.stage}, 상태: waiting")
        elif data.stage == "post_processing":
            logger.info(f"후처리 진행 중 - 분석ID: {data.analysis_id}, 단계: {data.stage}, 상태: post_processing")
        else:
            logger.info(f"분석 진행 중 - 분석ID: {data.analysis_id}, 단계: {data.stage}, 진행률: {progress}% ({current}/{total}), 메시지: {data.message}, 서버 진행률: {data.progress}%")

        return {"message": "진행률 업데이트 완료"}

    except UnicodeDecodeError as e:
        logger.error(f"문자열 디코딩 실패: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail="요청 데이터를 UTF-8로 디코딩할 수 없습니다."
        )
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 실패: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail=f"잘못된 JSON 형식입니다: {str(e)}"
        )
    except Exception as e:
        logger.error(f"진행률 업데이트 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="진행률 업데이트 중 오류가 발생했습니다."
        )

@router.get("/process/{analysis_id}")
async def get_process_status(analysis_id: str, db: Session = Depends(get_db)):
    """
    특정 분석의 진행 상태와 진행률을 조회하는 엔드포인트
    """
    try:
        # Analysis 테이블에서 상태 확인
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
            raise HTTPException(
                status_code=404,
                detail="분석을 찾을 수 없습니다."
            )

        # 상태별 메시지 정의
        status_messages = {
            "processing": "분석 처리중",
            "completed": "분석 완료",
            "failed": "분석 불가능 (파일 손상 또는 지원하지 않는 형식)",
            "retrying": "일시적 오류로 재시도 중",
            "waiting": "프레임 추출 중",
            "post_processing": "후처리 진행중"  # 후처리 상태 메시지 추가
        }

        # 기본 응답 데이터 구성
        response_data = {
            "analysis_id": analysis_id,
            "status": analysis.status,
            "message": status_messages.get(analysis.status, "처리중"),
            "updated_at": analysis.updated_at
        }

        # 분석이 완료된 경우 진행률 조회하지 않고 바로 반환
        if analysis.status == "completed":
            return {
                **response_data,
                "message": "분석이 완료되었습니다.",
                "progress": 100,
                "stage": "completed",
                "current": "100",
                "total": "100"
            }
        elif analysis.status == "post_processing":  # 후처리 상태 처리 추가
            return {
                **response_data,
                "message": "후처리 진행중입니다.",
                "progress": 100,
                "stage": "post_processing",
                "current": "100",
                "total": "100"
            }
        elif analysis.status == "failed":
            return {
                **response_data,
                "message": "분석 처리에 실패했습니다.",
                "progress": analysis.progress or 0,
                "stage": "failed",
                "current": "0",
                "total": "0",
                "error_type": "permanent"  # 영구적인 오류임을 나타냄
            }
        elif analysis.status == "retrying":
            return {
                **response_data,
                "message": "일시적인 오류가 발생하여 재시도 중입니다.",
                "progress": analysis.progress or 0,
                "stage": "retrying",
                "current": "0",
                "total": "0",
                "error_type": "temporary"  # 일시적인 오류임을 나타냄
            }
        elif analysis.status == "waiting":
            return {
                **response_data,
                "message": "프레임 추출 중입니다.",
                "progress": 0,
                "stage": "extract_frames",
                "current": "0",
                "total": "0"
            }
        elif analysis.status != "processing":
            return {
                **response_data,
                "message": "잘못된 분석 상태입니다.",
                "progress": analysis.progress or 0,
                "stage": "unknown",
                "current": "0",
                "total": "0"
            }

        # 처리중인 경우에만 진행률 정보 조회
        progress_data = process_status.get(analysis_id)
        if progress_data:
            if progress_data.stage == "analyze_frame":
                # analyze_frame 단계일 때는 processing 상태로 진행률 표시
                response_data.update({
                    "status": "processing",  # 명시적으로 processing 상태 설정
                    "stage": progress_data.stage,
                    "message": progress_data.message,
                    "current": progress_data.current,
                    "total": progress_data.total,
                    "progress": progress_data.progress
                })
            else:
                # extract_frames 단계는 waiting 상태로 처리
                response_data.update({
                    "status": "waiting",
                    "message": "프레임 추출 중입니다.",
                    "progress": 0,
                    "stage": "extract_frames",
                    "current": "0",
                    "total": "0"
                })
        else:
            # 진행률 정보가 없는 경우 기본값 설정
            response_data.update({
                "status": "waiting",
                "stage": "waiting",
                "message": "분석 준비중",
                "current": "0",
                "total": "0",
                "progress": "0"
            })

        return response_data

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"진행 상태 조회 중 오류 발생: {str(e)}")
        logger.exception("상세 에러:")  # 스택 트레이스 추가
        raise HTTPException(
            status_code=500,
            detail="진행 상태 조회 중 오류가 발생했습니다."
        )

@router.delete("/process/{analysis_id}")
async def clear_process_status(analysis_id: str, db: Session = Depends(get_db)):
    """
    분석이 완료되거나 실패했을 때 진행률 정보를 삭제하는 엔드포인트
    """
    try:
        # 분석 ID 존재 여부 확인
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
            logger.warning(f"존재하지 않는 분석 ID: {analysis_id}")
            raise HTTPException(
                status_code=404,
                detail="분석을 찾을 수 없습니다."
            )

        # 진행률 정보 삭제
        if analysis_id in process_status:
            del process_status[analysis_id]
            logger.info(f"진행률 정보 삭제 완료 - 분석ID: {analysis_id}")

        return {"message": "진행률 정보 삭제 완료"}

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"진행률 정보 삭제 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="진행률 정보 삭제 중 오류가 발생했습니다."
        )