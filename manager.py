from fastapi import WebSocket
from typing import Dict, Set, Optional
import logging
import json
from datetime import datetime, timedelta
import time
import pytz
import asyncio

logger = logging.getLogger(__name__)

def get_korean_time():
    korean_tz = pytz.timezone('Asia/Seoul')
    return datetime.now(korean_tz)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._connection_times: Dict[str, datetime] = {}
        self.connection_stats: Dict[str, Dict] = {}
        self.max_connections = 100
        self.cleanup_interval = 300  # 5분
        self.last_cleanup = time.time()
        
    def _get_connection_lock(self, client_id: str) -> asyncio.Lock:
        """클라이언트 ID에 대한 락을 가져옵니다."""
        if client_id not in self._locks:
            self._locks[client_id] = asyncio.Lock()
        return self._locks[client_id]
        
    async def register_connection(self, websocket: WebSocket, client_id: str) -> bool:
        """새로운 WebSocket 연결을 등록합니다."""
        try:
            lock = self._get_connection_lock(client_id)
            async with lock:
                if client_id in self.active_connections:
                    logger.warning(f"이미 존재하는 연결: {client_id}")
                    return False
                    
                self.active_connections[client_id] = websocket
                self._connection_times[client_id] = datetime.now(pytz.UTC)
                self.connection_stats[client_id] = {
                    "connected_at": get_korean_time().isoformat(),
                    "total_frames": 0,
                    "error_count": 0,
                    "reconnect_attempts": 0
                }
                logger.info(f"새로운 연결 등록: {client_id}")
                return True
                
        except Exception as e:
            logger.error(f"연결 등록 중 오류 발생: {str(e)}")
            return False
            
    async def remove_connection(self, client_id: str):
        """WebSocket 연결을 제거합니다."""
        try:
            lock = self._get_connection_lock(client_id)
            async with lock:
                if client_id in self.active_connections:
                    del self.active_connections[client_id]
                if client_id in self._connection_times:
                    del self._connection_times[client_id]
                if client_id in self._locks:
                    del self._locks[client_id]
                logger.info(f"연결 제거 완료: {client_id}")
        except Exception as e:
            logger.error(f"연결 제거 중 오류 발생: {str(e)}")
            
    async def get_connection(self, client_id: str) -> Optional[WebSocket]:
        """클라이언트 ID에 해당하는 WebSocket 연결을 가져옵니다."""
        return self.active_connections.get(client_id)
        
    async def broadcast(self, message: str, exclude_client_id: Optional[str] = None):
        """모든 연결된 클라이언트에게 메시지를 브로드캐스트합니다."""
        disconnected_clients = []
        
        for client_id, connection in self.active_connections.items():
            if client_id == exclude_client_id:
                continue
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"메시지 전송 실패 ({client_id}): {str(e)}")
                disconnected_clients.append(client_id)
                
        # 연결이 끊긴 클라이언트 정리
        for client_id in disconnected_clients:
            await self.remove_connection(client_id)
            
    def get_connection_time(self, client_id: str) -> Optional[datetime]:
        """클라이언트의 연결 시간을 가져옵니다."""
        return self._connection_times.get(client_id)
        
    def get_active_connections_count(self) -> int:
        """현재 활성화된 연결 수를 반환합니다."""
        return len(self.active_connections)
        
    def get_active_client_ids(self) -> list:
        """현재 활성화된 모든 클라이언트 ID를 반환합니다."""
        return list(self.active_connections.keys())
        
    async def cleanup_connection(self, client_id: str):
        """연결 정리를 수행합니다."""
        if client_id in self.active_connections:
            try:
                ws = self.active_connections[client_id]
                await ws.close(code=1000, reason="정상 종료")
            except Exception as e:
                logger.debug(f"연결 종료 중 오류: {str(e)}")
            finally:
                del self.active_connections[client_id]
                
    async def disconnect(self, client_id: str):
        """연결을 안전하게 해제합니다."""
        if client_id in self.active_connections:
            # 연결 통계 업데이트
            if client_id in self.connection_stats:
                self.connection_stats[client_id].update({
                    "disconnected_at": get_korean_time().isoformat(),
                    "session_status": "disconnected",
                    "last_activity": get_korean_time().isoformat()
                })
            logger.info(f"WebSocket 연결 해제: {client_id}")
            await self.cleanup_connection(client_id)
        
    async def cleanup_old_stats(self):
        """오래된 통계 데이터 정리"""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            # 24시간 이상 된 통계 데이터 삭제
            cutoff_time = (get_korean_time() - timedelta(days=1)).isoformat()
            self.connection_stats = {
                client_id: stats
                for client_id, stats in self.connection_stats.items()
                if stats.get("connected_at", "") > cutoff_time
            }
            self.last_cleanup = current_time
            logger.info(f"오래된 통계 데이터 정리 완료. 남은 통계: {len(self.connection_stats)}개")
            
    async def send_result(self, client_id: str, result: dict):
        """분석 결과를 클라이언트에 전송합니다."""
        if client_id in self.active_connections:
            try:
                # 결과에 타임스탬프 추가
                result_with_timestamp = {
                    **result,
                    "timestamp": get_korean_time().isoformat()
                }
                await self.active_connections[client_id].send_json(result_with_timestamp)
                
                # 성공적인 프레임 처리 통계 업데이트
                if client_id in self.connection_stats:
                    self.connection_stats[client_id].update({
                        "total_frames": self.connection_stats[client_id].get("total_frames", 0) + 1,
                        "last_activity": get_korean_time().isoformat()
                    })
                    
            except Exception as e:
                logger.error(f"결과 전송 실패: {str(e)}")
                # 에러 통계 업데이트
                if client_id in self.connection_stats:
                    self.connection_stats[client_id].update({
                        "error_count": self.connection_stats[client_id].get("error_count", 0) + 1,
                        "last_error": str(e),
                        "last_activity": get_korean_time().isoformat()
                    })
                await self.disconnect(client_id)
                
    async def update_stats(self, client_id: str, stats: dict):
        """연결 통계를 업데이트합니다."""
        if client_id in self.connection_stats:
            self.connection_stats[client_id].update({
                **stats,
                "last_activity": get_korean_time().isoformat()
            })
            logger.info(f"연결 통계 업데이트 - {client_id}: {stats}")
            
    async def get_stats(self, client_id: str) -> Optional[dict]:
        """특정 클라이언트의 연결 통계를 반환합니다."""
        return self.connection_stats.get(client_id)
        
    async def get_all_stats(self) -> Dict[str, dict]:
        """모든 활성 연결의 통계를 반환합니다."""
        return {
            client_id: stats 
            for client_id, stats in self.connection_stats.items()
            if client_id in self.active_connections
        }

manager = ConnectionManager()