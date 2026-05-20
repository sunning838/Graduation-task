import os
import shutil
import re
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(CURRENT_DIR, "chroma_db")

BASE_DATA_DIR = os.path.join(CURRENT_DIR, "storage", "data")
BASE_QUIZ_DIR = os.path.join(CURRENT_DIR, "storage", "quiz")

SUBJECT_KOR_TO_ENG = {
    "소프트웨어 설계": "software_design",
    "소프트웨어 개발": "software_development",
    "데이터베이스 구축": "database",
    "프로그래밍 언어 활용": "programming_language",
    "정보시스템 구축 관리": "info_system"
}

def create_tensor_db():
    print("1. 기존 텐서 공간(DB) 초기화 중...")
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    all_chunks = []

    # ==========================================
    # [A 파트] 개념 데이터
    # ==========================================
    print("\n2. [개념 데이터] 폴더 경로 기반 메타데이터 부여 중...")
    if os.path.exists(BASE_DATA_DIR):
        loader = DirectoryLoader(BASE_DATA_DIR, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        concept_documents = loader.load()
        
        if concept_documents:
            concept_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=700, chunk_overlap=50)
            concept_chunks = concept_splitter.split_documents(concept_documents)
            
            for chunk in concept_chunks:
                file_path = chunk.metadata.get("source", "")
                rel_path = os.path.relpath(file_path, BASE_DATA_DIR)
                path_parts = rel_path.split(os.sep)
                
                cert = path_parts[0] if len(path_parts) > 0 else "UNKNOWN"
                subject = path_parts[1] if len(path_parts) > 2 else cert
                
                chunk.metadata["doc_type"] = "concept"
                chunk.metadata["cert"] = cert
                chunk.metadata["subject"] = subject
                
            all_chunks.extend(concept_chunks)
            print(f" - 개념 텐서 조각: {len(concept_chunks)}개 생성 완료")

    # ==========================================
    # [B 파트] 퀴즈 데이터
    # ==========================================
    print("\n3. [퀴즈 데이터] 본문 정규식 파싱 및 메타데이터 부여 중...")
    if os.path.exists(BASE_QUIZ_DIR):
        quiz_loader = DirectoryLoader(BASE_QUIZ_DIR, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        quiz_documents = quiz_loader.load()
        
        if quiz_documents:
            #'---' 구분자 지정
            quiz_splitter = CharacterTextSplitter(separator="\n---\n", chunk_size=1500, chunk_overlap=0)
            quiz_chunks = quiz_splitter.split_documents(quiz_documents)
            
            for chunk in quiz_chunks:
                file_path = chunk.metadata.get("source", "")
                rel_path = os.path.relpath(file_path, BASE_QUIZ_DIR)
                path_parts = rel_path.split(os.sep)
                
                # 자격증 이름(EIP)은 폴더 트리에서 가져옴 (quiz/EIP/...)
                cert = path_parts[0] if len(path_parts) > 0 else "UNKNOWN"
                
                #  과목명 파싱
                content = chunk.page_content
                match = re.search(r'#\s*과목\s*:\s*([^#\n]+)', content)
                
                if match:
                    kor_subject = match.group(1).strip()
                    # 한글 과목명을 매핑 딕셔너리를 이용해 영문으로 변환 (없으면 unknown)
                    eng_subject = SUBJECT_KOR_TO_ENG.get(kor_subject, "unknown_subject")
                else:
                    # 과목명 태그가 없는 문제의 경우의 안전망
                    eng_subject = "unknown_subject"
                
                chunk.metadata["doc_type"] = "quiz"
                chunk.metadata["cert"] = cert
                chunk.metadata["subject"] = eng_subject
                
            all_chunks.extend(quiz_chunks)
            print(f" - 퀴즈 텐서 조각: {len(quiz_chunks)}개 생성 완료")

    # ==========================================
    # [C 파트] 글로벌 병합 적재
    # ==========================================
    if not all_chunks:
        print("\n오류: 로드할 텐서 조각이 없습니다.")
        return

    print(f"\n4. 총 {len(all_chunks)}개의 텐서 조각을 DB에 적재 중...")
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask", encode_kwargs={'normalize_embeddings': True})
    
    Chroma.from_documents(documents=all_chunks, embedding=embeddings, persist_directory=DB_DIR, collection_metadata={"hnsw:space": "cosine"})
    print(f"\n5. 완료! DB가 '{DB_DIR}' 폴더에 재구축되었습니다.")

if __name__ == "__main__":
    create_tensor_db()