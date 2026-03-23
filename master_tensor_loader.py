import os
from dotenv import load_dotenv
import sys
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import time

# 윈도우 터미널 출력 텐서 깨짐 방지
sys.stdout.reconfigure(encoding='utf-8')

print("--- 🚀 통합 텐서 추출 파이프라인 가동 ---")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 텐서 분할기(Splitter) 세팅
# MD용: 의미 공간(헤더) 기반 텐서 분할
headers_to_split_on = [("#", "대분류"), ("##", "중분류"), ("###", "소분류")]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# TXT용: 시퀀스 길이 기반 텐서 분할 (1000자 단위, 100자 오버랩으로 문맥 소실 방지)
txt_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# 전체 텐서 블록을 담을 거대한 마스터 배열
master_tensor_chunks = []

# 2. 데이터가 있는 최상위 폴더 경로 설정 
base_dir = os.path.join(CURRENT_DIR, "storage")

# 타겟 폴더 필터링 (기출문제 등 불필요한 텐서 유입 차단)
target_folders = ["data_1", "data_2", "data_3"]

print("[System] 원시 데이터 스캔 및 텐서 변환 시작...\n")

# 3. 폴더 순회 및 라우팅 알고리즘
for root, dirs, files in os.walk(base_dir):
    folder_name = os.path.basename(root)
    
    # 타겟 폴더가 아니면 텐서화 생략
    if folder_name not in target_folders:
        continue

    for file in files:
        file_path = os.path.join(root, file)
        
        # [라우터 A] 마크다운(.md) 파일 -> 헤더 기준 의미 텐서화
        if file.endswith(".md"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    md_text = f.read()
                    chunks = md_splitter.split_text(md_text)
                    
                    # 출처 메타데이터 추가 (어느 파일에서 온 텐서 좌표인지 추적하기 위함)
                    for chunk in chunks:
                        chunk.metadata['source'] = file
                        
                    master_tensor_chunks.extend(chunks)
                    print(f"📖 [MD 텐서화 완료] {file} -> {len(chunks)}개 블록")
            except Exception as e:
                print(f"❌ [에러] {file} 텐서 변환 실패: {e}")

        # [라우터 B] 일반 텍스트(.txt) 파일 -> 시퀀스 길이 기준 키워드 텐서화
        elif file.endswith(".txt"):
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                chunks = txt_splitter.split_documents(docs)
                
                for chunk in chunks:
                    chunk.metadata['source'] = file
                    
                master_tensor_chunks.extend(chunks)
                print(f"📝 [TXT 텐서화 완료] {file} -> {len(chunks)}개 블록")
            except Exception as e:
                print(f"❌ [에러] {file} 텐서 변환 실패: {e}")

# 4. 최종 텐서 변환 결과 집계
print(f"\n🎉 [성공] 지정된 폴더의 모든 원시 데이터 파싱 완료!")
print(f"🧠 총 {len(master_tensor_chunks)}개의 통합 텐서 블록(Chunk)이 생성되어 임베딩 대기 중입니다.")

# 첫 번째 텐서 샘플 맵핑 구조 확인
if master_tensor_chunks:
    print("\n--- [첫 번째 텐서 블록 좌표 및 시퀀스 샘플] ---")
    print(f"📌 메타데이터: {master_tensor_chunks[0].metadata}")
    print(f"📝 입력 내용: {master_tensor_chunks[0].page_content[:100]}...")

print("--- 💎 Gemini 텐서 적재 파이프라인 가동 ---")

# 1. API 키 설정
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# 2. 임베딩 모델 설정
# 이 모델이 텍스트 조각을 768차원의 숫자 벡터(텐서)로 변환합니다.
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 3. 텐서 저장소(ChromaDB) 경로 설정
# D 드라이브의 해당 경로에 'chroma_db'라는 저장 창고가 생성됩니다.
persist_directory = os.path.join(CURRENT_DIR, "chroma_db")

# 4. 텐서 적재 실행 (스로틀링 배치 아키텍처 적용)
print(f"[System] 텐서 변환 및 DB 적재 시작 (데이터 양: {len(master_tensor_chunks)}개 블록)...")

# 먼저 빈 텐서 저장소를 셋업하네
vector_db = Chroma(
    embedding_function=embeddings,
    persist_directory=persist_directory
)

# 무료 API 한도 방어를 위해 90개씩 묶어서(Batch) 텐서를 전송
batch_size = 90

for i in range(0, len(master_tensor_chunks), batch_size):
    batch_chunks = master_tensor_chunks[i : i + batch_size]
    current_end = min(i + batch_size, len(master_tensor_chunks))
    
    print(f"🔄 [배치 전송 중] {i+1} ~ {current_end} 번째 텐서 블록 적재 중...")
    
    # 해당 배치만큼만 DB에 밀어넣기
    vector_db.add_documents(documents=batch_chunks)
    
    # 아직 보낼 텐서가 남았다면, 1분당 100회 제한을 피하기 위해 60초 대기
    if i + batch_size < len(master_tensor_chunks):
        print("⏳ API 트래픽 한도 보호를 위해 60초간 파이프라인을 일시 정지합니다...")
        time.sleep(60)

# 5. 저장 완료 확인
print(f"\n✅ [성공] 텐서 저장소 구축 완료! 위치: {persist_directory}")

# 6. 간단한 검색 테스트
query = "폭포수 모형의 특징이 뭐야?"
docs = vector_db.similarity_search(query, k=1)

print(f"\n🔎 [DB 검색 테스트 결과]")
if docs:
    print(f"가장 유사한 텍스트 텐서: {docs[0].page_content[:150]}...")
    print(f"출처 맵핑 좌표: {docs[0].metadata.get('source', '알 수 없음')}")