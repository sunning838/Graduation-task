import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil

#python .\backend\local_tensor_loder.py 로 실행

# 현재 파일의 디렉토리 절대 경로를 구합니다.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(CURRENT_DIR, "chroma_db")

# 1. 일반 개념 마크다운 파일들이 있는 폴더
DATA_DIRS = [
    os.path.join(CURRENT_DIR, "storage/data_t")
]

# 2. 기출문제 마크다운 파일들이 있는 전용 폴더
QUIZ_DATA_DIR = os.path.join(CURRENT_DIR, "storage/data_quiz")

def create_tensor_db():
    print("1. 기존 텐서 공간(DB) 초기화 중...")
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    all_chunks = []

    # ==========================================
    # [A 파트] 일반 개념 텐서 처리 로직
    # ==========================================
    print("\n2. [일반 개념] 지식 데이터 로딩 및 분할 중...")
    concept_documents = []
    for data_dir in DATA_DIRS:
        if not os.path.exists(data_dir):
            print(f"경고: {data_dir} 폴더를 찾을 수 없습니다. 건너뜁니다.")
            continue
            
        loader = DirectoryLoader(data_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        concept_documents.extend(loader.load())

    if concept_documents:
        # 일반 개념은 500자 단위로 쪼개고 앞뒤 문맥을 50자씩 겹치게(Overlap) 함
        concept_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=500,
            chunk_overlap=50
        )
        concept_chunks = concept_splitter.split_documents(concept_documents)
        
        # 일반 개념에는 'concept' 꼬리표 부착
        for chunk in concept_chunks:
            chunk.metadata["doc_type"] = "concept"
            
        all_chunks.extend(concept_chunks)
        print(f" - 일반 개념 텐서 조각: {len(concept_chunks)}개 생성 완료")

    # ==========================================
    # [B 파트] 기출문제 텐서 처리 로직
    # ==========================================
    print("\n3. [기출문제] 지식 데이터 로딩 및 분할 중...")
    if os.path.exists(QUIZ_DATA_DIR):
        quiz_loader = DirectoryLoader(QUIZ_DATA_DIR, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        quiz_documents = quiz_loader.load()
        
        if quiz_documents:
            #  기출문제는 문제 하나가 통째로 들어가도록 설정 (Overlap 없음, 구분선 사용)
            quiz_splitter = CharacterTextSplitter(
                separator="\n---\n", 
                chunk_size=1000,
                chunk_overlap=0
            )
            quiz_chunks = quiz_splitter.split_documents(quiz_documents)
            
            # 기출문제에는 'quiz' 꼬리표 부착
            for chunk in quiz_chunks:
                chunk.metadata["doc_type"] = "quiz"
                
            all_chunks.extend(quiz_chunks)
            print(f" - 기출문제 텐서 조각: {len(quiz_chunks)}개 생성 완료")
    else:
        print(f"알림: {QUIZ_DATA_DIR} 폴더가 아직 없어 기출문제 처리를 건너뜁니다.")

    # ==========================================
    # [C 파트] 크로마 DB 최종 병합 적재
    # ==========================================
    if not all_chunks:
        print("\n오류: 로드할 텐서 조각이 전혀 없습니다. 폴더와 파일을 확인해주세요.")
        return

    print(f"\n4. 총 {len(all_chunks)}개의 텐서 조각을 DB에 적재 중... (시간이 조금 걸릴 수 있습니다)")
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask",encode_kwargs={'normalize_embeddings': True})
    
    # 꼬리표가 달린 모든 텐서를 한 번에 섞어서 DB에 적재
    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"\n5. 완료! 하이브리드 텐서 DB가 '{DB_DIR}' 폴더에 성공적으로 구축되었습니다.")

if __name__ == "__main__":
    create_tensor_db()