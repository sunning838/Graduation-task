import sqlite3
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(CURRENT_DIR, "storage", "user_data.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS quiz_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cert TEXT,
            topic TEXT,
            is_correct INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# 기록할 때 자격증 정보(도 같이 저장
def log_quiz_result(cert, topic, is_correct):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO quiz_logs (cert, topic, is_correct) VALUES (?, ?, ?)", (cert, topic, 1 if is_correct else 0))
    conn.commit()
    conn.close()

# 취약점을 찾을 때 현재 자격증 안에서만 검색
def get_weakest_topic(cert):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        SELECT topic, 
               COUNT(*) as total_attempts, 
               SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) as wrong_count
        FROM quiz_logs
        WHERE cert = ?
        GROUP BY topic
        ORDER BY wrong_count DESC, total_attempts ASC
        LIMIT 1
    ''', (cert,))
    
    result = c.fetchone()
    conn.close()
    
    if result and result[2] > 0:
        return result[0]
    else:
        return "데이터 부족 (랜덤 출제)"

def reset_quiz_logs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM quiz_logs")
    conn.commit()
    conn.close()

def get_subject_stats(cert):
    """자격증에 대한 과목별 정답률 통계를 반환합니다."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 각 과목(topic)별로 푼 문제 수와 정답 수를 집계하는 SQL 쿼리
    c.execute('''
        SELECT topic, 
               COUNT(*) as total_attempts, 
               SUM(is_correct) as correct_count
        FROM quiz_logs
        WHERE cert = ?
        GROUP BY topic
    ''', (cert,))
    
    rows = c.fetchall()
    conn.close()
    
    stats = {}
    for row in rows:
        topic, total, correct = row
        # 정답률(%) 계산
        accuracy = (correct / total) * 100 if total > 0 else 0
        stats[topic] = round(accuracy, 1)
        
    return stats

def get_top_weakest_topics(cert, n=3):
    """특정 자격증에서 가장 많이 틀린 Top N개의 단원 목록을 반환합니다."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        SELECT topic, 
               SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) as wrong_count
        FROM quiz_logs
        WHERE cert = ?
        GROUP BY topic
        HAVING wrong_count > 0
        ORDER BY wrong_count DESC
        LIMIT ?
    ''', (cert, n))
    
    results = c.fetchall()
    conn.close()
    
    # 예: ['software_design', 'database'] 형태로 반환
    return [row[0] for row in results]