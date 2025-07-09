import asyncio
import websockets
import json
import requests
import logging
import os
from datetime import datetime
import pytz
from dotenv import load_dotenv
import socket
import traceback
import time

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_websocket_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# 서버 URL 설정
BASE_URL = os.getenv("MAIN_SERVER_URL", "http://211.254.212.141:8000")
APP_WS_URL = os.getenv("MAIN_WS_URL", "ws://211.254.212.141:8004") + "/ws/analyze"

# WebSocket 설정
WS_TIMEOUT = 300  # 5분
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
        
        response = requests.post(f"{BASE_URL}/login", json=login_data, timeout=30)
        response.raise_for_status()
        
        token = response.json()["access_token"]
        logger.info("로그인 성공")
        return token
        
    except requests.exceptions.RequestException as e:
        logger.error(f"로그인 실패: {str(e)}")
        raise

async def handle_app_message(websocket, message):
    """앱 WebSocket 메시지를 처리합니다."""
    try:
        result = json.loads(message)
        logger.info("="*50)
        logger.info("서버로부터 메시지 수신")
        logger.info(f"수신 시간: {get_korean_time().isoformat()}")
        logger.info(f"메시지 내용: {json.dumps(result, indent=2, ensure_ascii=False)}")
        logger.info("="*50)
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 실패: {str(e)}")
        logger.error(f"수신된 원본 메시지: {message[:200]}...")
    except Exception as e:
        logger.error(f"메시지 처리 중 오류: {str(e)}")
        logger.error(f"상세 오류: {traceback.format_exc()}")

async def test_app_websocket(email: str, password: str, duration: int = 10):
    """
    앱 WebSocket 연결을 테스트합니다. (수신만 테스트)
    
    Args:
        email: 사용자 이메일
        password: 사용자 비밀번호
        duration: 테스트 지속 시간(초)
    """
    retry_count = 0
    connection_success = False
    
    while retry_count < MAX_RETRIES and not connection_success:
        try:
            # 1. 로그인
            token = await login(email, password)
            
            # 2. WebSocket 연결
            uri = f"{APP_WS_URL}?token={token}"
            logger.info(f"앱 WebSocket 연결 시도: {uri}")
            
            # WebSocket 연결 설정
            headers = {
                'User-Agent': 'TestAppClient/1.0',
                'Authorization': f'Bearer {token}'
            }
            
            async with websockets.connect(
                uri,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10,
                subprotocols=['analyze'],
                additional_headers=headers,
                max_size=1024*1024
            ) as websocket:
                logger.info("앱 WebSocket 연결 성공")
                connection_success = True
                
                # 연결 유지 시간
                start_time = time.time()
                end_time = start_time + duration
                
                # 메시지 수신 대기
                while time.time() < end_time:
                    try:
                        message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=1.0  # 1초 타임아웃
                        )
                        await handle_app_message(websocket, message)
                    except asyncio.TimeoutError:
                        continue  # 타임아웃은 무시하고 계속 수신 대기
                    except websockets.exceptions.ConnectionClosed:
                        logger.info("서버가 연결을 종료했습니다.")
                        break
                
                logger.info("="*50)
                logger.info("테스트 완료")
                logger.info(f"총 실행 시간: {time.time() - start_time:.2f}초")
                logger.info("="*50)
                
                return
                    
        except websockets.exceptions.InvalidStatusCode as e:
            logger.error(f"WebSocket 연결 실패 (상태 코드: {e.status_code})")
            if e.status_code == 401:
                logger.error("인증 실패 - 토큰이 유효하지 않습니다.")
                break
        except websockets.exceptions.ConnectionClosed as e:
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
    password = "zz153133!"
    
    # 환경 변수가 설정되어 있으면 환경 변수 사용
    if os.getenv("TEST_EMAIL") and os.getenv("TEST_PASSWORD"):
        email = os.getenv("TEST_EMAIL")
        password = os.getenv("TEST_PASSWORD")
        logger.info("환경 변수에서 계정 정보를 사용합니다.")
    
    # 테스트 지속 시간 설정 (초)
    duration = int(os.getenv("TEST_DURATION", "10"))
    
    logger.info(f"테스트 모드: 웹소켓 연결 테스트 ({duration}초)")
    logger.info(f"사용자 정보 - 이메일: {email}")
        
    asyncio.run(test_app_websocket(email, password, duration)) 