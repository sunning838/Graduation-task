import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import json
import random

# 환경 변수 로드
load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(CURRENT_DIR, "chroma_db")

class AITutorEngine:
    def __init__(self):
        print("[시스템] 메모리 텐서가 탑재된 파이프라인 초기화 중...")
        self.embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        self.vector_db = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
        
        # 최신 텐서 모델 유지
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        print("[시스템] 튜터 엔진 가동 준비 완료!\n")

    def get_relevant_tensor(self, query: str, k: int = 3) -> str:
        """지식 공간에서 텐서를 인출하는 함수"""
        docs = self.vector_db.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])

    def generate_response(self, query: str, chat_history: list) -> str:
        """대화 기록(Memory Tensor)을 포함하여 최종 답변을 생성하는 메인 로직"""
        
        context = self.get_relevant_tensor(query)
        
        # 프롬프트 텐서에 'MessagesPlaceholder'를 추가하여 기억력을 이식!
        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 정보처리기사 자격증 합격을 돕는 친절하고 똑똑한 AI 튜터다. 
주어진 [참고 지식]과 이전 대화 맥락을 바탕으로 학생의 질문에 답변해라. 
설명은 이해하기 쉽게 예시를 들어주고, 지식에 없는 내용이라면 모른다고 대답해라.

[참고 지식]
{context}"""),
            # 이곳에 이전까지의 대화 텐서들이 통째로 삽입됨
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        # 연산 실행 시 chat_history 변수도 함께 넘겨줌
        return chain.invoke({
            "context": context, 
            "chat_history": chat_history, 
            "question": query
        })
    
    def generate_quiz(self) -> dict:
        """지식 공간의 데이터를 기반으로 실제 객관식 문제 텐서를 생성하는 함수"""
        
        # 1. 벡터 데이터베이스에서 임의의 문서 텐서들을 인출 (다양한 문제 출제를 위해 무작위 검색어 사용)
        random_keywords = ["데이터", "정규화", "조인", "트랜잭션", "인덱스", "설계", "구조"]
        search_query = random.choice(random_keywords)
        
        # 유사도 검색을 통해 문제 출제용 원천 지식 텐서 확보
        docs = self.vector_db.similarity_search(search_query, k=2)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 2. 문제 출제를 위한 명령 텐서(프롬프트) 설계
        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 정보처리기사 자격증 시험을 출제하는 교수다. 
주어진 [참고 지식]을 바탕으로 학생이 풀 수 있는 객관식 문제 1개를 출제해라.
반드시 아래의 출력 형식을 엄격하게 지켜야 하며, 파이썬의 json.loads()로 즉시 파싱 가능한 순수한 JSON 형태만 출력해라. 코드 블록(```json) 같은 감싸기 표식을 쓰지 말고 중괄호로만 시작하고 끝내라.

[출력 형식]
{{
    "question": "여기에 문제 질문을 작성",
    "choices": ["1) 보기1", "2) 보기2", "3) 보기3", "4) 보기4"],
    "answer": 3,
    "explanation": "여기에 정답에 대한 상세한 해설을 작성 (예: 3번이 정답인 이유 등)"
}}

[참고 지식]
{context}"""),
            ("human", "위 지식을 바탕으로 자격증 시험에 나올법한 객관식 문제를 하나 출제해줘.")
        ])

        chain = prompt | self.llm | StrOutputParser()
        raw_output = chain.invoke({"context": context})

        # 3. 생성된 문자열 텐서를 파이썬 딕셔너리 구조체로 변환
        try:
            # 혹시 모델이 코드 블록을 붙여 출력할 경우를 대비한 방어 텐서 연산
            clean_output = raw_output.replace("```json", "").replace("```", "").strip()
            quiz_data = json.loads(clean_output)
            return quiz_data
        except Exception as e:
            # 파싱 실패 시 예외 처리용 기본 텐서 반환
            return {
                "question": "정규화의 목적으로 가장 적절하지 않은 것은?",
                "choices": ["1) 중복 제거", "2) 이상 현상 방지", "3) 무결성 유지", "4) 저장 공간의 낭비 증가"],
                "answer": 4,
                "explanation": "정규화는 데이터 중복을 제거하여 저장 공간을 효율적으로 사용하기 위함입니다."
            }

# --- 실행 테스트 블록 ---
if __name__ == "__main__":
    tutor = AITutorEngine()
    
    # 세션 동안의 대화 텐서를 저장할 빈 리스트 생성 (메모리 모듈)
    session_chat_history = []
    
    print("="*50)
    print(" [기억력 탑재 완료] 정보처리기사 AI 튜터가 작동을 시작했습니다.")
    print("질문을 입력해주세요. (종료하려면 'q' 입력)")
    print("="*50)
    
    while True:
        user_input = input("\n학생 : ")
        
        if user_input.lower() == 'q':
            print("튜터 : 학습을 종료합니다. 수고하셨습니다!")
            break
            
        print("튜터 : (텐서 맥락 분석 및 답변 생성 중...)")
        
        try:
            # 질문과 함께 누적된 대화 기록을 튜터 엔진에 전달
            answer = tutor.generate_response(user_input, session_chat_history)
            print(f"\n튜터 : {answer}")
            
            # 답변이 무사히 생성되면, 현재 턴의 문답을 기억 텐서에 업데이트
            session_chat_history.append(HumanMessage(content=user_input))
            session_chat_history.append(AIMessage(content=answer))
            
        except Exception as e:
            print(f"\n[오류] 텐서 연산 중 문제가 발생했습니다: {e}")

