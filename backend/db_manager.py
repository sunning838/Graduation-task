import sqlite3
import os

# 현재 파일(db_manager.py)의 절대 좌표를 기반으로 DB 파일 위치(텐서 저장소) 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(CURRENT_DIR, "storage", "user_data.db")

def init_db():
    """데이터베이스 텐서 공간을 초기화하고 기록 테이블을 생성하는 연산"""
    # storage 폴더가 존재하지 않으면 생성
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 퀴즈 로그를 기록할 테이블 텐서 생성 (id, 주제, 정답여부, 시간)
    c.execute('''
        CREATE TABLE IF NOT EXISTS quiz_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            is_correct INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def log_quiz_result(topic: str, is_correct: bool):
    """학생이 문제를 풀었을 때 그 결과를 영구 텐서로 기록"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # boolean(True/False) 값을 정수형 텐서(1/0)로 변환하여 DB에 꽂아 넣음
    c.execute("INSERT INTO quiz_logs (topic, is_correct) VALUES (?, ?)", (topic, int(is_correct)))
    
    conn.commit()
    conn.close()

def get_weakest_topic() -> str:
    """오답(is_correct=0) 텐서가 가장 많이 누적된 취약 개념을 연산하여 반환"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # SQL 쿼리 텐서를 통해 오답 개수를 그룹화하고 가장 많이 틀린 주제 1개를 추출
    c.execute('''
        SELECT topic, COUNT(*) as wrong_count 
        FROM quiz_logs 
        WHERE is_correct = 0 
        GROUP BY topic 
        ORDER BY wrong_count DESC 
        LIMIT 1
    ''')
    
    result = c.fetchone()
    conn.close()
    
    # 틀린 기록이 추출되었다면 해당 주제 반환, 기록이 전혀 없다면 기본 텐서 반환
    if result:
        return result[0]
    else:
        # DB가 비어있거나 모든 문제를 맞혔을 때의 기본 타겟팅 챕터
        return "요구사항 확인"

def reset_quiz_logs():
    """지금까지 누적된 모든 퀴즈 풀이 및 오답 기록 텐서를 영구적으로 삭제"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # quiz_logs 테이블 내부의 모든 데이터를 지우는 파괴 연산
    c.execute("DELETE FROM quiz_logs")
    
    conn.commit()
    conn.close()