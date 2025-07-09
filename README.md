# Deepfake Detection Assistant Backend

## 프로젝트 개요
딥페이크 탐지 보조 애플리케이션의 백엔드 서버입니다.  
AI 서버와 연동하여 영상 및 프레임 단위 분석 결과를 영상업로드 및 실시간으로 처리하고 앱에 알림 기능을 제공합니다.

## 사용 기술
- Python 
- FastAPI
- WebSocket (실시간 통신)
- MariaDB
- Ubuntu 서버 (원격 관리)
- TablePlus (DB 관리)
- Git 

## 주요 기능
- 회원 가입 및 로그인, 인증 처리 (이메일, sms )
- AI 서버 및 앱과 REST API 및 WebSocket 연동
- 영상업로드 및 실시간 프레임 단위 딥페이크 분석 결과 수신 및 전송
- 앱 클라이언트 연동
- 서버 및 DB 관리

## 실행 방법
```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn main:app --reload
