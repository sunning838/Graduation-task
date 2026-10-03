import os
import sys
from functools import lru_cache

# python -m backend.test.tensor_search_smoke

# 윈도우 터미널 출력 깨짐 방지

# 1. 텐서 DB 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(os.path.dirname(CURRENT_DIR), "chroma_db")

# 2. 질의(Query)를 텐서로 변환할 임베딩 모델 로드 
# (주의: 텐서를 적재할 때 썼던 모델 및 파라미터와 완벽히 동일해야 합니다)
@lru_cache
def get_vector_db():
    # Importing this manual check must not load models or access the real DB.
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    if not os.path.isdir(persist_directory):
        raise FileNotFoundError(f"Chroma DB 폴더가 없습니다: {persist_directory}")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("[System] 기존 텐서 저장소를 불러옵니다...")
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# 4. 텐서 검색 테스트 함수
def search_tensor_db(query, k=3):
    print(f"\n🔎 [검색 질의 텐서화 및 매칭]: '{query}'")
    
    # similarity_search_with_score: 텐서 간의 거리(유사도 점수)를 함께 반환합니다.
    # 거리(점수)가 낮을수록 의미가 완벽히 일치하는 텐서입니다.
    docs_and_scores = get_vector_db().similarity_search_with_score(query, k=k)
    
    if not docs_and_scores:
        print("매칭되는 텐서를 찾을 수 없습니다.")
        return

    for i, (doc, score) in enumerate(docs_and_scores):
        print(f"\n--- [매칭 텐서 {i+1} (거리 점수: {score:.4f})] ---")
        print(f" 출처 맵핑 좌표: {doc.metadata.get('source', '알 수 없음')}")
        # 텐서 내용이 너무 길 수 있으므로 200자만 출력
        print(f" 텐서 내용: {doc.page_content[:200]}...") 

# 5. 테스트 실행부
if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    # 여러 가지 질문을 텐서 공간에서 테스트해 봅니다.
    search_tensor_db("폭포수 모형이란?", k=2)
    search_tensor_db("XP의 핵심 가치는?", k=1)
