import os
import requests
import xml.etree.ElementTree as ET
import time
import re
import shutil
from dotenv import load_dotenv

#  .env에서 API 키 로드
load_dotenv()
API_KEY = os.getenv("LAW_API_KEY")

if not API_KEY:
    print("[오류] .env 파일에 LAW_API_KEY가 설정되지 않았습니다!")
    exit()

#  1차 시험에 필요한 6대 법령과 영문 파일명 매핑 딕셔너리
TARGET_LAWS = {
    "민법": "civil_law",
    "주택임대차보호법": "housing_lease",
    "상가건물 임대차보호법": "commercial_lease",
    "집합건물의 소유 및 관리에 관한 법률": "aggregate_building",
    "가등기담보 등에 관한 법률": "provisional_registration",
    "부동산 실권리자명의 등기에 관한 법률": "real_name_registration"
}

SAVE_DIR = os.path.join("backend", "storage", "data", "LREA_1", "civil_law")

if os.path.exists(SAVE_DIR): # 기존 데이터 삭제
    shutil.rmtree(SAVE_DIR)
    print("기존 데이터 삭제 완료")


os.makedirs(SAVE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def get_base_article_num(num_str):
    """ '777의2' 같은 조문 번호에서 숫자 '777'만 안전하게 추출하는 헬퍼 함수 """
    match = re.match(r'^(\d+)', num_str)
    return int(match.group(1)) if match else 0

def crawl_full_lrea_law_data():
    print(f"🚀 [시스템] 공인중개사 1차 DB 완벽 구축 파이프라인 가동 (총 {len(TARGET_LAWS)}개 법령)")
    
    # 6개의 법령을 순회하며 크롤링 시작
    for law_name, file_prefix in TARGET_LAWS.items():
        print(f"\n==================================================")
        print(f"📚 [{law_name}] 수집을 시작합니다...")
        
        search_url = "https://www.law.go.kr/DRF/lawSearch.do"
        search_params = {"OC": API_KEY, "target": "law", "type": "XML", "query": law_name}
        
        try:
            # 1단계: MST 고유 번호 검색
            search_response = requests.get(search_url, params=search_params, headers=HEADERS)
            search_response.raise_for_status()
            search_root = ET.fromstring(search_response.content)
            
            mst = None
            for law in search_root.findall('.//law'):
                if law.findtext('법령명한글') == law_name:
                    mst = law.findtext('법령일련번호')
                    break
                    
            if not mst:
                print(f"⚠️ '{law_name}' 검색 실패. 건너뜁니다.")
                continue
                
            print(f"✅ MST 획득: [{mst}] -> 본문 다운로드 중...")
            time.sleep(1) # 서버 보호를 위한 1초 대기

            # 2단계: 본문 데이터 다운로드
            service_url = "https://www.law.go.kr/DRF/lawService.do"
            service_params = {"OC": API_KEY, "target": "law", "type": "XML", "MST": mst}
            
            service_response = requests.get(service_url, params=service_params, headers=HEADERS)
            service_response.raise_for_status()
            service_root = ET.fromstring(service_response.content)
            
            articles = service_root.findall('.//조문단위')
            
            success_count = 0
            deleted_count = 0
            filtered_count = 0
            
            for article in articles: 
                article_num = article.findtext('조문번호')
                if not article_num:
                    continue
                
                # 🚀 [업그레이드 2] 민법의 경우 제777조(가족법) 이상은 가차 없이 버림!
                if law_name == "민법" and get_base_article_num(article_num) >= 777:
                    filtered_count += 1
                    continue

                title = article.findtext('조문제목', '')
                
                # 본문, 항, 호 데이터 병합
                content_lines = []
                jo_content = article.findtext('조문내용')
                if jo_content: content_lines.append(jo_content.strip())
                
                for hang in article.findall('.//항내용'):
                    if hang.text: content_lines.append(hang.text.strip())
                    
                for ho in article.findall('.//호내용'):
                    if ho.text: content_lines.append(ho.text.strip())
                    
                full_content = "\n".join(content_lines)

                # 삭제된 조문 필터링
                if "삭제" in full_content and len(full_content) < 20: 
                    deleted_count += 1
                    continue

                if full_content:
                    #충돌 방지를 위해 접두사를 사용한 고유 파일명 생성
                    filename = f"{file_prefix}_{article_num.replace('의', '_')}.md"
                    filepath = os.path.join(SAVE_DIR, filename)
                    
                    title_text = title if title else f"제{article_num}조"
                    md_content = f"## {law_name} {title_text}\n\n**내용:**\n{full_content}\n"
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(md_content)
                        
                    success_count += 1
                    
            print(f"🎉 [{law_name}] 완료! 저장됨: {success_count}개 (삭제제외: {deleted_count}개, 범위초과차단: {filtered_count}개)")

        except Exception as e:
            print(f"❌ '{law_name}' 크롤링 중 오류 발생: {e}")

if __name__ == "__main__":
    crawl_full_lrea_law_data()