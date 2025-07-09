from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from sqlalchemy.orm import Session
import logging
from datetime import datetime
import time
import json
import traceback
import pytz
import asyncio
import websockets
from jose import jwt, JWTError
from ..websocket.manager import manager
from core.database import get_db, SessionLocal
from core.config import settings
from models.tables import Analysis, DetectionDetail, User
from schemas.user import (
    FrameAnalysisResult,
    RiskLevelInfo,
    RiskLevel,
    DetectionType
)
from starlette.websockets import WebSocketState
from contextlib import asynccontextmanager
import random
import numpy as np
import cv2

router = APIRouter(tags=["realtime"])
logger = logging.getLogger(__name__)

# AI 서버 WebSocket URL 설정
AI_WS_URL = "ws://211.254.212.141:8004"
AI_WS_TIMEOUT = 300  # 5분
AI_WS_RETRY_COUNT = 3  # 재시도 횟수
AI_WS_RETRY_DELAY = 5  # 재시도 대기 시간 (초)
AI_WS_CONNECT_TIMEOUT = 30  # 연결 시도 타임아웃 (초)

def get_korean_time():
    korean_tz = pytz.timezone('Asia/Seoul')
    return datetime.now(korean_tz)

def calculate_risk_level(artifacts_score: float) -> RiskLevel:
    """
    위험도 점수를 기반으로 위험 레벨을 계산합니다.
    
    Args:
        artifacts_score (float): 0~100 사이의 위험도 점수
        
    Returns:
        RiskLevel: 계산된 위험 레벨
    """
    if not 0 <= artifacts_score <= 100:
        logger.error(f"잘못된 위험도 점수: {artifacts_score} (0~100 사이여야 함)")
        return RiskLevel.NORMAL  # 기본값 반환
        
    if artifacts_score <= 25:
        return RiskLevel.VERY_SAFE
    elif artifacts_score <= 50:
        return RiskLevel.SAFE
    elif artifacts_score <= 70:
        return RiskLevel.NORMAL
    elif artifacts_score <= 80:
        return RiskLevel.DANGEROUS
    else:  # 80 < artifacts_score <= 100
        return RiskLevel.VERY_DANGEROUS

def validate_ai_response(result: dict) -> bool:
    """
    AI 서버로부터 받은 분석 결과의 형식을 검증합니다.
    
    Args:
        result (dict): AI 서버로부터 받은 분석 결과
        
    Returns:
        bool: 검증 결과 (True: 유효한 결과, False: 잘못된 결과)
    """
    # 필수 필드 확인
    required_fields = ["confidence", "risk_score"]
    
    # 모든 필수 필드가 있는지 확인
    if not all(field in result for field in required_fields):
        logger.error(f"필수 필드 누락: {required_fields}")
        return False
        
    # confidence 값 검증 (0.0 ~ 1.0)
    confidence = result.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        logger.error(f"잘못된 confidence 값: {confidence}")
        return False
        
    # risk_score 값 검증 (0.0 ~ 1.0)
    risk_score = result.get("risk_score")
    if not isinstance(risk_score, (int, float)) or not 0 <= risk_score <= 1:
        logger.error(f"잘못된 risk_score 값: {risk_score}")
        return False
        
    return True

@asynccontextmanager
async def websocket_connection(websocket: WebSocket):
    """WebSocket 연결을 관리하는 컨텍스트 매니저"""
    try:
        if websocket.client_state == WebSocketState.CONNECTING:
            await websocket.accept()
            yield websocket
        else:
            raise RuntimeError(f"잘못된 WebSocket 상태: {websocket.client_state}")
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()

async def connect_to_ai_server(token: str, metadata: dict = None, retry_count: int = 0) -> websockets.WebSocketClientProtocol:
    """AI 서버에 WebSocket 연결을 설정합니다."""
    try:
        # 기본 URI에 토큰 추가
        base_uri = f"{AI_WS_URL}/ws/analyze?token={token}"
        
        # 메타데이터가 있는 경우 쿼리 파라미터로 추가
        if metadata:
            import urllib.parse
            metadata_params = urllib.parse.urlencode(metadata)
            uri = f"{base_uri}&{metadata_params}"
        else:
            uri = base_uri
            
        logger.info(f"AI 서버 WebSocket 연결 시도: {uri} (시도 {retry_count + 1}/{AI_WS_RETRY_COUNT})")
        
        websocket = await asyncio.wait_for(
            websockets.connect(
                uri,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10,
                subprotocols=['analyze']
            ),
            timeout=AI_WS_CONNECT_TIMEOUT
        )
        logger.info("AI 서버 WebSocket 연결 성공")
        return websocket
    except asyncio.TimeoutError:
        if retry_count < AI_WS_RETRY_COUNT - 1:
            logger.warning(f"AI 서버 연결 타임아웃 (시도 {retry_count + 1}/{AI_WS_RETRY_COUNT})")
            logger.info(f"{AI_WS_RETRY_DELAY}초 후 재시도...")
            await asyncio.sleep(AI_WS_RETRY_DELAY)
            return await connect_to_ai_server(token, metadata, retry_count + 1)
        else:
            logger.error("AI 서버 연결 최종 타임아웃")
            raise ConnectionError("AI 서버 연결 타임아웃")
    except Exception as e:
        if retry_count < AI_WS_RETRY_COUNT - 1:
            logger.warning(f"AI 서버 연결 실패 (시도 {retry_count + 1}/{AI_WS_RETRY_COUNT}): {str(e)}")
            logger.info(f"{AI_WS_RETRY_DELAY}초 후 재시도...")
            await asyncio.sleep(AI_WS_RETRY_DELAY)
            return await connect_to_ai_server(token, metadata, retry_count + 1)
        else:
            logger.error(f"AI 서버 연결 최종 실패: {str(e)}")
            raise

async def ensure_ai_connection(ai_websocket: websockets.WebSocketClientProtocol, token: str) -> websockets.WebSocketClientProtocol:
    """AI 서버 연결이 유효한지 확인하고, 필요시 재연결합니다."""
    if not ai_websocket or ai_websocket.closed:
        logger.warning("AI 서버 연결이 끊어짐. 재연결 시도...")
        try:
            return await connect_to_ai_server(token)
        except Exception as e:
            logger.error(f"AI 서버 재연결 실패: {str(e)}")
            raise ConnectionError("AI 서버 재연결 실패")
    return ai_websocket

async def send_frame_to_ai(ai_websocket: websockets.WebSocketClientProtocol, metadata: dict, frame_data: bytes) -> dict:
    """AI 서버에 프레임 데이터를 전송하고 결과를 받아옵니다."""
    if not ai_websocket or ai_websocket.closed:
        raise ConnectionError("AI 서버 연결이 닫혔습니다")
        
    try:
        # 이미지 데이터 전처리
        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("이미지 디코딩 실패")
            
        # 메모리 정렬 보장
        frame = np.ascontiguousarray(frame)
        
        # JPEG로 인코딩 (고품질)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, frame_encoded = cv2.imencode('.jpg', frame, encode_param)
        frame_data = frame_encoded.tobytes()
        
        # 메타데이터와 프레임 데이터를 동시에 전송하기 위한 코루틴 생성
        async def send_data():
            # 메타데이터와 프레임 데이터를 연속적으로 전송
            await ai_websocket.send(json.dumps(metadata))   
            await ai_websocket.send(frame_data)
            
        # 데이터 전송과 결과 수신을 병렬로 처리
        send_task = asyncio.create_task(send_data())
        receive_task = asyncio.create_task(
            asyncio.wait_for(ai_websocket.recv(), timeout=AI_WS_TIMEOUT)
        )
        
        # 두 작업이 모두 완료될 때까지 대기
        await send_task
        result = await receive_task
            
        # 연결 종료 메시지 확인
        if isinstance(result, dict) and result.get("type") == "websocket.close":
            raise ConnectionError("AI 서버가 연결을 종료했습니다")
            
        analysis_result = json.loads(result)
        if not validate_ai_response(analysis_result):
            raise ValueError("AI 서버로부터 받은 결과가 유효하지 않습니다")
            
        return analysis_result
            
    except asyncio.TimeoutError:
        raise TimeoutError("AI 서버 응답 시간 초과")
    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"AI 서버 연결 종료: {str(e)}")
        raise ConnectionError(f"AI 서버 연결이 종료됨: {str(e)}")
    except Exception as e:
        logger.error(f"AI 서버 통신 중 오류: {str(e)}")
        raise

async def is_websocket_connected(websocket: WebSocket) -> bool:
    """WebSocket 연결 상태를 확인합니다."""
    return websocket.client_state == WebSocketState.CONNECTED

async def safe_send_json(websocket: WebSocket, data: dict):
    """WebSocket을 통해 안전하게 JSON 데이터를 전송합니다."""
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            logger.info(f"앱으로 메시지 전송 시작: {json.dumps(data)[:200]}...")  # 처음 200자만 로깅
            await websocket.send_json(data)
            logger.info("앱으로 메시지 전송 완료")
        else:
            logger.warning("WebSocket이 연결되지 않은 상태에서 메시지 전송 시도")
    except Exception as e:
        logger.error(f"메시지 전송 중 오류 발생: {str(e)}")

@router.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket, token: str):
    """실시간 분석을 위한 WebSocket 엔드포인트"""
    client_id = None
    db = None
    ai_websocket = None
    metadata = None
    headers = dict(websocket.headers)  # headers 변수 정의
    
    try:
        logger.info("WebSocket 연결 시도")
        
        # 1. 토큰 검증
        try:
            # 토큰 디코딩 시 만료 검증 비활성화
            payload = jwt.decode(
                token, 
                settings.SECRET_KEY, 
                algorithms=["HS256"], 
                options={
                    "verify_aud": False,
                    "verify_exp": False,
                    "verify_iat": False,
                    "verify_nbf": False
                }
            )
            
            if not payload or "sub" not in payload:
                logger.error("토큰 페이로드가 유효하지 않음")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token payload")
                return
                
            token_email = payload["sub"]
            logger.info(f"사용자 정보 조회 성공: {token_email}")
            
            # 필수 헤더 검증
            required_headers = {
                'x-client-id': 'X-Client-ID',
                'x-user-email': 'X-User-Email',
                'x-device-id': 'X-Device-ID'
            }
            
            missing_headers = []
            for header_key, header_name in required_headers.items():
                if header_key not in headers:
                    missing_headers.append(header_name)
            
            if missing_headers:
                logger.error(f"필수 헤더 누락: {missing_headers}")
                await websocket.close(
                    code=status.WS_1008_POLICY_VIOLATION,
                    reason=f"Missing required headers: {', '.join(missing_headers)}"
                )
                return
            
            # 헤더 값 검증
            client_id = headers['x-client-id']
            user_email = headers['x-user-email']
            device_id = headers['x-device-id']
            
            # 이메일 검증
            if user_email != token_email:
                logger.error(f"이메일 불일치: 헤더({user_email}) != 토큰({token_email})")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Email mismatch")
                return
                
            logger.info(f"클라이언트 정보 검증 성공 - 클라이언트 ID: {client_id}, 디바이스 ID: {device_id}")
            
        except jwt.JWTError as e:
            logger.error(f"토큰 검증 실패: {str(e)}")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
            return
            
        # 2. WebSocket 연결 수락
        if websocket.client_state == WebSocketState.CONNECTING:
            await websocket.accept()
            logger.info(f"WebSocket 연결 수락: {user_email}")
        else:
            logger.error(f"잘못된 WebSocket 상태: {websocket.client_state}")
            return
            
        # 3. 연결 등록
        if not await manager.register_connection(websocket, client_id):
            logger.error("연결 등록 실패")
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Connection registration failed")
            return

        # 4. DB 세션 생성
        db = SessionLocal()
        
        # 5. AI 서버 연결
        try:
            # 메타데이터가 있을 때까지 대기
            while metadata is None:
                try:
                    message = await websocket.receive()
                    if "text" in message:
                        metadata_str = message["text"]
                        try:
                            metadata = json.loads(metadata_str)
                            # 메타데이터 검증
                            required_fields = ["analysis_id", "user_email", "device_id", "detection_type", "file_type", "frame_number"]
                            missing_fields = [field for field in required_fields if field not in metadata]
                            if missing_fields:
                                logger.error(f"필수 필드 누락: {missing_fields}")
                                await safe_send_json(websocket, {
                                    "type": "error",
                                    "message": f"Missing required fields: {', '.join(missing_fields)}",
                                    "code": "MISSING_FIELDS",
                                    "missing_fields": missing_fields
                                })
                                metadata = None
                                continue
                            break
                        except json.JSONDecodeError:
                            logger.error("메타데이터 JSON 파싱 실패")
                            continue
                except Exception as e:
                    logger.error(f"메타데이터 수신 중 오류: {str(e)}")
                    break
            
            # 메타데이터를 포함하여 AI 서버 연결
            ai_websocket = await connect_to_ai_server(token, metadata)
            
        except Exception as e:
            logger.error(f"AI 서버 연결 실패: {str(e)}")
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="AI server connection failed")
            return

        # 프레임 처리 큐 생성
        frame_queue = asyncio.Queue(maxsize=30)  # 최대 30개 프레임 버퍼
        ai_message_queue = asyncio.Queue(maxsize=100)  # AI 서버 메시지 큐
        
        # 프레임 처리 워커
        async def process_frames(ai_ws: websockets.WebSocketClientProtocol):
            """프레임을 AI 서버로 전송하는 워커"""
            while True:
                try:
                    if websocket.client_state == WebSocketState.DISCONNECTED:
                        break
                        
                    # AI 서버 연결 상태 확인
                    try:
                        ai_ws = await ensure_ai_connection(ai_ws, token)
                    except ConnectionError as e:
                        logger.error(f"AI 서버 연결 유지 실패: {str(e)}")
                        break
                        
                    # 큐에서 프레임 데이터 가져오기
                    metadata, frame_data = await frame_queue.get()
                    
                    try:
                        # 프레임 데이터 전송
                        await ai_ws.send(json.dumps(metadata))
                        await ai_ws.send(frame_data)
                        
                    except Exception as e:
                        logger.error(f"프레임 처리 중 오류: {str(e)}")
                        if await is_websocket_connected(websocket):
                            await safe_send_json(websocket, {
                                "type": "error",
                                "message": f"Frame processing error: {str(e)}",
                                "code": "PROCESSING_ERROR",
                                "frame_number": metadata["frame_number"]
                            })
                    finally:
                        frame_queue.task_done()
                        
                except Exception as e:
                    logger.error(f"프레임 처리 워커 오류: {str(e)}")
                    break

        # AI 서버 메시지 수신 워커
        async def receive_ai_messages(ws: websockets.WebSocketClientProtocol):
            """AI 서버로부터 메시지를 수신하여 큐에 넣는 워커"""
            while True:
                try:
                    if websocket.client_state == WebSocketState.DISCONNECTED:
                        logger.info("앱 연결이 종료되어 AI 서버 메시지 수신 중단")
                        break
                        
                    # AI 서버 연결 상태 확인
                    try:
                        ws = await ensure_ai_connection(ws, token)
                    except ConnectionError as e:
                        logger.error(f"AI 서버 연결 유지 실패: {str(e)}")
                        break
                        
                    # AI 서버로부터 메시지 수신
                    try:
                        message = await ws.recv()
                        if isinstance(message, str):
                            try:
                                data = json.loads(message)
                                # 메시지를 큐에 넣기
                                await ai_message_queue.put(data)
                            except json.JSONDecodeError:
                                logger.error("AI 서버로부터 받은 메시지가 유효한 JSON이 아님")
                                continue
                    except websockets.exceptions.ConnectionClosed as e:
                        logger.error(f"AI 서버 연결 종료: {str(e)}")
                        break
                    except Exception as e:
                        logger.error(f"AI 서버 메시지 수신 중 오류: {str(e)}")
                        break
                        
                except Exception as e:
                    logger.error(f"AI 서버 메시지 수신 워커 오류: {str(e)}")
                    break

        # AI 서버 메시지 처리 워커
        async def process_ai_messages():
            """AI 서버 메시지 큐에서 메시지를 가져와 처리하는 워커"""
            while True:
                try:
                    if websocket.client_state == WebSocketState.DISCONNECTED:
                        break
                        
                    # 큐에서 메시지 가져오기
                    data = await ai_message_queue.get()
                    
                    try:
                        # 분석 서버 데이터 형식 검증
                        required_fields = {
                            "analysis_id": str,
                            "user_email": str,
                            "device_id": str,
                            "detection_type": str,
                            "file_type": str,
                            "frame_number": (str, int),  # 문자열 또는 정수 허용
                            "confidence": (int, float),
                            "analysis_time": (int, float),
                            "risk_score": (int, float),
                            "details": dict
                        }
                        
                        # 필드 존재 여부와 타입 검증
                        missing_fields = []
                        type_mismatch_fields = []
                        
                        for field, expected_type in required_fields.items():
                            if field not in data:
                                missing_fields.append(field)
                            elif not isinstance(data[field], expected_type):
                                type_mismatch_fields.append(f"{field}({type(data[field]).__name__})")
                        
                        if missing_fields or type_mismatch_fields:
                            if missing_fields:
                                logger.error(f"필수 필드 누락: {missing_fields}")
                            if type_mismatch_fields:
                                logger.error(f"타입 불일치: {type_mismatch_fields}")
                            continue
                        
                        # frame_number를 정수로 변환 (문자열이면 변환 시도)
                        try:
                            if isinstance(data["frame_number"], str):
                                data["frame_number"] = int(data["frame_number"])
                        except (ValueError, TypeError) as e:
                            logger.error(f"frame_number를 정수로 변환 실패: {data['frame_number']}, 오류: {str(e)}")
                            continue
                        
                        # 위험도 점수 검증 및 변환 (0~1 -> 0~100)
                        try:
                            risk_score = float(data["risk_score"])
                            if not 0 <= risk_score <= 1:
                                logger.error(f"잘못된 위험도 점수 범위: {risk_score} (0~1 사이여야 함)")
                                continue
                            risk_score = risk_score * 100  # 0~1 -> 0~100으로 변환
                        except (ValueError, TypeError) as e:
                            logger.error(f"위험도 점수 변환 실패: {str(e)}")
                            continue
                        
                        # 위험도 단계 계산
                        risk_level = calculate_risk_level(risk_score)
                        risk_level_info = RiskLevelInfo(
                            level=risk_level,
                            score=int(risk_score)  # 정수로 변환
                        )
                        
                        # 분석 서버 데이터에 risk_level_info 추가
                        result_data = data.copy()
                        result_data["risk_level_info"] = risk_level_info.dict()
                        result_data["timestamp"] = get_korean_time().isoformat()
                        
                        if await is_websocket_connected(websocket):
                            await safe_send_json(websocket, result_data)
                            logger.info(f"분석 결과 전송 - 프레임: {result_data['frame_number']}, 위험도: {risk_score:.1f}, 위험레벨: {risk_level}")
                        else:
                            logger.warning("앱 연결이 끊어져 분석 결과를 전송할 수 없음")
                            
                    except Exception as e:
                        logger.error(f"메시지 처리 중 오류: {str(e)}")
                    finally:
                        ai_message_queue.task_done()
                        
                except Exception as e:
                    logger.error(f"AI 서버 메시지 처리 워커 오류: {str(e)}")
                    break

        # 워커 태스크 시작
        frame_worker = asyncio.create_task(process_frames(ai_websocket))
        ai_receive_worker = asyncio.create_task(receive_ai_messages(ai_websocket))
        ai_process_worker = asyncio.create_task(process_ai_messages())
        
        # 메인 메시지 처리 루프
        while True:
            try:
                if websocket.client_state == WebSocketState.DISCONNECTED:
                    break
                
                # 메시지 수신
                message = await websocket.receive()
                # === stop/종료 메시지 처리, 반드시 여기! ===
                if "type" in message and message["type"] == "stop":
                    logger.info("클라이언트에서 세션 종료 요청")
                    break
                if set(message.keys()) == {"type", "code", "reason"}:
                    logger.info("WebSocket 종료 메시지 수신")
                    break
                # === 이후에만 메타데이터 파싱 ===
                if "text" in message:
                    logger.info(f"텍스트 메시지 길이: {len(message['text'])}")
                    logger.info(f"텍스트 메시지 내용: {message['text'][:200]}...")  # 처음 200자만 로깅
                elif "bytes" in message:
                    logger.info(f"바이너리 메시지 크기: {len(message['bytes'])} bytes")
                    logger.info(f"바이너리 메시지 헥사덤프 (처음 32바이트): {message['bytes'][:32].hex()}")
                    with open(f"debug_recv_{metadata['frame_number']}.jpg", "wb") as f:
                        f.write(message['bytes'])
                logger.info("="*50)
                
                # 연결 종료 메시지 처리
                if "type" in message and message["type"] == "websocket.close":
                    logger.info(f"클라이언트 연결 종료 요청: {message.get('reason', 'No reason provided')}")
                    break
                    
                # 메시지 타입에 따른 처리
                if "text" in message:
                    # 텍스트 메시지 처리 (메타데이터)
                    metadata_str = message["text"]
                    logger.info(f"메타데이터 문자열: {metadata_str}")
                    try:
                        metadata = json.loads(metadata_str)
                        logger.info(f"파싱된 메타데이터: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
                    except json.JSONDecodeError as e:
                        logger.error(f"메타데이터 JSON 파싱 실패: {str(e)}")
                        logger.error(f"파싱 실패한 원본 데이터: {metadata_str}")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid metadata format. Expected valid JSON.",
                            "code": "INVALID_JSON"
                        })
                        continue
                        
                    # 메타데이터 검증
                    required_fields = ["analysis_id", "user_email", "device_id", "detection_type", "file_type", "frame_number"]
                    missing_fields = [field for field in required_fields if field not in metadata]
                    if missing_fields:
                        logger.error(f"필수 필드 누락: {missing_fields}")
                        await safe_send_json(websocket, {
                            "type": "error",
                            "message": f"Missing required fields: {', '.join(missing_fields)}",
                            "code": "MISSING_FIELDS",
                            "missing_fields": missing_fields
                        })
                        continue
                        
                    # 메타데이터 유효성 검사
                    if metadata["detection_type"] != "realtime":
                        logger.error(f"잘못된 detection_type: {metadata['detection_type']}")
                        await safe_send_json(websocket, {
                            "type": "error",
                            "message": "Invalid detection_type. Must be 'realtime'",
                            "code": "INVALID_DETECTION_TYPE"
                        })
                        continue
                        
                    if metadata["file_type"] not in ["image", "video"]:
                        logger.error(f"잘못된 file_type: {metadata['file_type']}")
                        await safe_send_json(websocket, {
                            "type": "error",
                            "message": "Invalid file_type. Must be 'image' or 'video'",
                            "code": "INVALID_FILE_TYPE"
                        })
                        continue
                        
                    if metadata["user_email"] != user_email or metadata["device_id"] != device_id:
                        logger.error("메타데이터와 헤더 정보가 일치하지 않음")
                        await safe_send_json(websocket, {
                            "type": "error",
                            "message": "Metadata does not match header information",
                            "code": "METADATA_MISMATCH"
                        })
                        continue
                        
                    # 분석 세션 시작 알림
                    await websocket.send_json({
                        "type": "notification",
                        "message": "Analysis session started",
                        "analysis_id": metadata["analysis_id"]
                    })
                            
                elif "bytes" in message:
                    # 바이너리 메시지 처리 (프레임 데이터)
                    frame_data = message["bytes"]
                    
                    # 메타데이터가 없는 경우 오류 처리
                    if not metadata:
                        logger.error("프레임 데이터는 메타데이터 이후에 전송되어야 합니다")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Frame data must be sent after metadata",
                            "code": "INVALID_MESSAGE_ORDER"
                        })
                        continue
                        
                    # 프레임 큐에 데이터 추가
                    try:
                        await frame_queue.put((metadata.copy(), frame_data))
                    except asyncio.QueueFull:
                        logger.warning("프레임 큐가 가득 찼습니다. 이전 프레임을 건너뜁니다.")
                        continue
                            
                else:
                    logger.error(f"지원하지 않는 메시지 형식: {message.keys()}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Unsupported message format",
                        "code": "UNSUPPORTED_FORMAT"
                    })
                    continue
                    
            except WebSocketDisconnect:
                logger.info(f"클라이언트 연결 종료: {client_id}")
                break
            except Exception as e:
                logger.error(f"메시지 처리 중 오류: {str(e)}")
                break
                
        # 워커 태스크 종료 대기
        frame_worker.cancel()
        ai_receive_worker.cancel()
        ai_process_worker.cancel()
        try:
            await frame_worker
            await ai_receive_worker
            await ai_process_worker
        except asyncio.CancelledError:
            pass
                
    except Exception as e:
        logger.error(f"WebSocket 처리 중 오류: {str(e)}")
        logger.error(f"상세 오류: {traceback.format_exc()}")
    finally:
        if client_id:
            await manager.remove_connection(client_id)
            logger.info(f"연결 종료 처리 완료: {client_id}")
        if db:
            db.close()
            logger.info("DB 세션 정리 완료")
        if ai_websocket and not ai_websocket.closed:
            await ai_websocket.close()
            logger.info("AI 서버 연결 종료")