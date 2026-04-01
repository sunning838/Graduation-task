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


    def generate_response(self, query: str, chat_history: list, student_status: str = "분석된 상태 없음") -> str:
        """대화 기록(Memory Tensor)과 학생의 상태 텐서를 포함하여 최종 답변을 생성하는 메인 로직"""
        
        context = self.get_relevant_tensor(query)
        
        # 시스템 프롬프트 텐서에 [학생의 현재 상태] 영역 추가
        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 정보처리기사 자격증 합격을 돕는 친절하고 똑똑한 AI 튜터다. 
아래의 [학생의 현재 상태]를 반드시 인지하고, 학생이 자신의 취약점이나 성적에 대해 물어보면 이 데이터를 바탕으로 정확하게 대답해라. 
설명은 이해하기 쉽게 예시를 들어주고, 지식에 없는 내용이라면 모른다고 대답해라.

[학생의 현재 상태]
{student_status}

[참고 지식]
{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        # 연산 실행 시 student_status 변수도 함께 주입
        return chain.invoke({
            "context": context, 
            "chat_history": chat_history, 
            "student_status": student_status,
            "question": query
        })
    
    def generate_quiz(self, target_topic: str = None) -> dict:
        """지식 공간의 데이터를 기반으로 실제 객관식 문제 텐서를 생성하는 함수 (취약점 타겟팅 지원)"""
        
        topics = [
            "요구사항 확인", "화면 설계", "데이터 입출력 구현", 
            "통합 구현", "인터페이스 구현", "소프트웨어 개발 보안 구축",
            "응용 SW 기초 기술 활용"
        ]
        
        selected_topic = target_topic if target_topic else random.choice(topics)
        docs = self.vector_db.similarity_search(selected_topic, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 국가공인 '정보처리기사' 자격증 시험을 출제하는 전담 교수다. 
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
            ("human", f"위 지식을 바탕으로 '{selected_topic}' 파트에서 자격증 시험에 나올법한 객관식 문제를 하나 출제해줘.")
        ])

        chain = prompt | self.llm | StrOutputParser()
        raw_output = chain.invoke({"context": context})

        try:
            clean_output = raw_output.replace("```json", "").replace("```", "").strip()
            quiz_data = json.loads(clean_output)
            quiz_data["topic"] = selected_topic 
            return quiz_data
        except Exception as e:
            return {
                "question": "데이터베이스 설계 순서로 올바른 것은? (파싱 오류 임시 문제)",
                "choices": ["1) 개념-논리-물리", "2) 물리-논리-개념", "3) 논리-개념-물리", "4) 개념-물리-논리"],
                "answer": 1,
                "explanation": "요구사항 분석 후 개념적, 논리적, 물리적 설계 순으로 진행됩니다.",
                "topic": selected_topic
            }

# --- 실행 테스트 블록 ---
if __name__ == "__main__":
    tutor = AITutorEngine()
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
            # 터미널 로컬 테스트 환경에서는 상태 텐서가 없으므로 기본값으로 작동함
            answer = tutor.generate_response(user_input, session_chat_history)
            print(f"\n튜터 : {answer}")
            
            session_chat_history.append(HumanMessage(content=user_input))
            session_chat_history.append(AIMessage(content=answer))
            
        except Exception as e:
            print(f"\n[오류] 텐서 연산 중 문제가 발생했습니다: {e}")