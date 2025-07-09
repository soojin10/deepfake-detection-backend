import asyncio
import websockets
import json
import base64
from PIL import Image
import io
import requests
import logging
import os
from datetime import datetime
import pytz
from dotenv import load_dotenv
import socket
import traceback
import time
from websockets.exceptions import (
    ConnectionClosed,
    WebSocketException
)

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG 레벨로 변경하여 더 자세한 로그 확인
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('websocket_test.log'),  # 파일로도 로그 저장
        logging.StreamHandler()  # 콘솔에도 출력
    ]
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# 서버 URL 설정
BASE_URL = os.getenv("MAIN_SERVER_URL", "http://211.254.212.141:8000")
AI_SERVER_URL = os.getenv("AI_SERVER_URL", "http://211.254.212.141:8004")
BACKEND_WS_URL = os.getenv("BACKEND_WS_URL", "ws://211.254.212.141:8003")  # 백엔드 서버 WebSocket URL을 8003으로 변경

# WebSocket 설정
WS_TIMEOUT = 300  # 5분
WS_CONNECT_TIMEOUT = 30  # 연결 시도 타임아웃 (초)
MAX_RETRIES = 3
RETRY_DELAY = 5  # 초

def get_korean_time():
    korean_tz = pytz.timezone('Asia/Seoul')
    return datetime.now(korean_tz)

async def login(email: str, password: str) -> str:
    """로그인하여 토큰을 받아옵니다."""
    try:
        login_data = {
            "email": email,
            "password": password
        }
        
        # 로그인 엔드포인트 시도
        endpoints = [
            "/auth/login",
            "/api/auth/login",
            "/login",
            "/api/login"
        ]
        
        last_error = None
        for endpoint in endpoints:
            try:
                response = requests.post(f"{BASE_URL}{endpoint}", json=login_data, timeout=30)
                response.raise_for_status()
                token = response.json()["access_token"]
                logger.info(f"로그인 성공 (엔드포인트: {endpoint})")
                return token
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.debug(f"엔드포인트 {endpoint} 시도 실패: {str(e)}")
                continue
        
        if last_error:
            logger.error(f"모든 로그인 엔드포인트 시도 실패. 마지막 에러: {str(last_error)}")
            raise last_error
            
    except Exception as e:
        logger.error(f"로그인 실패: {str(e)}")
        raise

async def create_test_image(width: int = 640, height: int = 480, pattern: str = "gradient") -> bytes:
    """테스트용 이미지를 생성합니다.
    
    Args:
        width (int): 이미지 너비 (기본값: 640)
        height (int): 이미지 높이 (기본값: 480)
        pattern (str): 이미지 패턴 ("gradient", "checker", "solid", "random" 중 하나)
    
    Returns:
        bytes: JPEG 형식의 이미지 데이터
    """
    try:
        from PIL import Image, ImageDraw
        import random
        import numpy as np
        
        # 이미지 생성
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)
        
        if pattern == "gradient":
            # 그라데이션 패턴 생성
            for y in range(height):
                r = int(255 * (1 - y/height))
                g = int(255 * (y/height))
                b = int(128 + 127 * np.sin(y/30))
                for x in range(width):
                    draw.point((x, y), fill=(r, g, b))
                    
        elif pattern == "checker":
            # 체크무늬 패턴 생성
            square_size = 40
            for y in range(0, height, square_size):
                for x in range(0, width, square_size):
                    color = (255, 255, 255) if (x//square_size + y//square_size) % 2 == 0 else (0, 0, 0)
                    draw.rectangle([x, y, x+square_size, y+square_size], fill=color)
                    
        elif pattern == "solid":
            # 단색 이미지 생성 (랜덤 색상)
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            draw.rectangle([0, 0, width, height], fill=color)
            
        elif pattern == "random":
            # 랜덤 노이즈 패턴 생성
            for y in range(height):
                for x in range(width):
                    color = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )
                    draw.point((x, y), fill=color)
        
        # JPEG로 저장
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=95)
        return img_byte_arr.getvalue()
        
    except Exception as e:
        logger.error(f"이미지 생성 실패: {str(e)}")
        raise

async def handle_websocket_message(websocket, message):
    """WebSocket 메시지를 처리합니다."""
    try:
        result = json.loads(message)
        logger.info("="*50)
        logger.info("분석 결과 수신")
        logger.info(f"수신 시간: {get_korean_time().isoformat()}")
        
        # 분석 결과 상세 로깅
        if "confidence" in result:
            logger.info(f"신뢰도: {result['confidence']*100:.2f}%")
        if "risk_score" in result:
            logger.info(f"위험도 점수: {result['risk_score']*100:.2f}%")
        if "risk_level" in result:
            logger.info(f"위험도 레벨: {result['risk_level']}")
        if "analysis_time" in result:
            logger.info(f"분석 소요시간: {result['analysis_time']:.2f}초")
            
        # 상세 정보가 있는 경우
        if "details" in result:
            logger.info("상세 정보:")
            for key, value in result["details"].items():
                logger.info(f"  - {key}: {value}")
                
        logger.info("="*50)
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 실패: {str(e)}")
        logger.error(f"수신된 원본 메시지: {message[:200]}...")  # 처음 200자만 로깅
    except Exception as e:
        logger.error(f"메시지 처리 중 오류: {str(e)}")
        logger.error(f"상세 오류: {traceback.format_exc()}")

async def test_websocket(email: str, password: str, num_frames: int = 5):
    """백엔드 서버 WebSocket 연결을 테스트합니다."""
    retry_count = 0
    analysis_id = int(time.time())
    device_id = "f831fe5b633b0b8f"
    connection_success = False
    
    patterns = ["gradient", "checker", "solid", "random"]
    
    while retry_count < MAX_RETRIES and not connection_success:
        try:
            # 1. 로그인
            logger.debug("로그인 시도 중...")
            token = await login(email, password)
            logger.debug(f"발급받은 토큰: {token[:20]}...")
            
            # 2. 백엔드 서버 WebSocket 연결
            uri = f"{BACKEND_WS_URL}/ws/analyze?token={token}"
            logger.info(f"백엔드 서버 WebSocket 연결 시도: {uri}")
            
            # 서버 연결 가능 여부 확인
            try:
                host = uri.split("://")[1].split(":")[0]
                port = int(uri.split(":")[2].split("/")[0])
                logger.debug(f"백엔드 서버 연결 테스트: {host}:{port}")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                if result == 0:
                    logger.info("백엔드 서버 포트 연결 가능")
                else:
                    logger.error(f"백엔드 서버 포트 연결 불가 (에러 코드: {result})")
                    logger.error("백엔드 서버가 실행 중인지, 방화벽 설정을 확인해주세요.")
                    return
                sock.close()
            except Exception as e:
                logger.error(f"백엔드 서버 연결 테스트 중 오류: {str(e)}")
                return
            
            logger.debug("백엔드 서버 WebSocket 연결 시도 전 상태 확인 완료")
            
            # WebSocket 연결 설정
            headers = {
                'User-Agent': 'TestClient/1.0',
                'Authorization': f'Bearer {token}',
                'X-Client-ID': f"{email}_{analysis_id}",
                'X-User-Email': email,
                'X-Device-ID': device_id
            }
            
            try:
                websocket = await asyncio.wait_for(
                    websockets.connect(
                uri,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10,
                subprotocols=['analyze'],
                additional_headers=headers
                    ),
                    timeout=WS_CONNECT_TIMEOUT
                )
                
                async with websocket:
                    logger.info("백엔드 서버 WebSocket 연결 성공")
                    logger.info(f"분석 세션 ID: {analysis_id}")
                connection_success = True
                
                # 여러 프레임 전송 (1부터 시작)
                for frame_num in range(1, num_frames + 1):
                    try:
                        # 테스트 이미지 생성 (패턴 순환)
                        pattern = patterns[(frame_num - 1) % len(patterns)]
                        logger.debug(f"테스트 이미지 {frame_num}/{num_frames} 생성 중... (패턴: {pattern})")
                        img_data = await create_test_image(pattern=pattern)
                        logger.debug(f"이미지 크기: {len(img_data)} bytes")
                        
                        # 메타데이터 전송
                        metadata = {
                            "analysis_id": analysis_id,
                            "user_email": email,
                            "device_id": device_id,
                            "detection_type": "realtime",
                            "file_type": "image",
                            "frame_number": frame_num,
                            "phone_number": "01036065406",
                            "token": token  # 분석 서버에 토큰 전달
                        }
                        logger.info(f"메타데이터 {frame_num}/{num_frames} 전송 중...")
                        logger.debug(f"전송할 메타데이터: {json.dumps(metadata, ensure_ascii=False)}")
                        await websocket.send(json.dumps(metadata))
                        logger.debug(f"메타데이터 전송 완료")
                        
                        # 이미지 데이터 전송
                        logger.info(f"이미지 데이터 {frame_num}/{num_frames} 전송 중...")
                        await websocket.send(img_data)
                        logger.debug(f"이미지 데이터 {frame_num}/{num_frames} 전송 완료")
                        
                        # 분석 결과 수신 대기
                        logger.info(f"프레임 {frame_num}/{num_frames} 분석 결과 대기 중...")
                        result = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=WS_TIMEOUT
                        )
                        # stop 또는 종료 메시지 처리
                        try:
                            msg_obj = json.loads(result)
                            if isinstance(msg_obj, dict):
                                if msg_obj.get("type") == "stop":
                                    logger.info("서버에서 stop 메시지 수신, 세션 종료")
                                    break
                                if set(msg_obj.keys()) == {"type", "code", "reason"}:
                                    logger.info("서버에서 WebSocket 종료 메시지 수신, 세션 종료")
                                    break
                        except Exception:
                            pass
                        await handle_websocket_message(websocket, result)
                        
                    except ConnectionClosed as e:
                        logger.info(f"연결이 종료되었습니다. 코드: {e.code}, 이유: {e.reason}")
                        break
                    except WebSocketException as e:
                        logger.error(f"WebSocket 에러 발생: {str(e)}")
                        break
                    except Exception as e:
                        logger.error(f"프레임 {frame_num} 처리 중 오류: {str(e)}")
                        logger.error(f"상세 오류: {traceback.format_exc()}")
                        break
                
                logger.info(f"총 {num_frames}개 프레임 전송 완료")
                return  # 성공적으로 완료되면 함수 종료
                    
            except asyncio.TimeoutError:
                logger.error("백엔드 서버 연결 타임아웃")
                if retry_count < MAX_RETRIES - 1:
                    logger.info(f"{RETRY_DELAY}초 후 재시도... (시도 {retry_count + 1}/{MAX_RETRIES})")
                    await asyncio.sleep(RETRY_DELAY)
                    retry_count += 1
                    continue
                else:
                    logger.error("최대 재시도 횟수 초과")
                    return
                    
        except WebSocketException as e:
            if hasattr(e, 'status_code'):
                logger.error(f"WebSocket 연결 실패 (상태 코드: {e.status_code})")
                if e.status_code == 401:
                    logger.error("인증 실패 - 토큰이 유효하지 않습니다.")
                    break
            else:
                logger.error(f"WebSocket 에러 발생: {str(e)}")
        except ConnectionClosed as e:
            logger.error(f"연결 종료: 코드={e.code}, 이유={e.reason}")
        except Exception as e:
            logger.error(f"에러 발생: {str(e)}")
            logger.error(f"상세 오류: {traceback.format_exc()}")
            
        retry_count += 1
        if retry_count < MAX_RETRIES and not connection_success:
            logger.info(f"{RETRY_DELAY}초 후 재시도... (시도 {retry_count}/{MAX_RETRIES})")
            await asyncio.sleep(RETRY_DELAY)
        else:
            if not connection_success:
                logger.error("최대 재시도 횟수 초과")
            else:
                logger.info("테스트 완료")

if __name__ == "__main__":
    # 실제 DB의 사용자 정보 사용
    email = "gooreami0710@naver.com"
    password = "zz153133!"  # 테스트용 비밀번호
    device_id = "f831fe5b633b0b8f"  # 실제 device_id
    
    # 환경 변수가 설정되어 있으면 환경 변수 사용
    if os.getenv("TEST_EMAIL") and os.getenv("TEST_PASSWORD"):
        email = os.getenv("TEST_EMAIL")
        password = os.getenv("TEST_PASSWORD")
        device_id = os.getenv("TEST_DEVICE_ID", device_id)  # device_id도 환경 변수에서 가져올 수 있도록
        logger.info("환경 변수에서 계정 정보를 사용합니다.")
    
    # 전송할 프레임 수 설정 (기본값: 5)
    num_frames = int(os.getenv("TEST_FRAMES", "5"))
    logger.info(f"테스트 프레임 수: {num_frames}")
    logger.info(f"사용자 정보 - 이메일: {email}, 디바이스: {device_id}")
        
    asyncio.run(test_websocket(email, password, num_frames)) 