from core.database import engine

def test_database_connection():
    try:
        # 데이터베이스 연결 테스트
        with engine.connect() as connection:
            print("✅ 데이터베이스 연결 성공!")
            # 연결 정보 출력
            print(f"Connected to: {connection.engine.url}")
            
    except Exception as e:
        print("❌ 데이터베이스 연결 실패!")
        print(f"에러 메시지: {str(e)}")

if __name__ == "__main__":
    test_database_connection()