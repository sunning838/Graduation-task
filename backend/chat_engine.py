import os
import json
import random
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# 환경 변수 로드
load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(CURRENT_DIR, "chroma_db")

# Pydantic 출력 스키마 정의 (엔진이 뱉어낼 JSON 구조 강제)
class QuizResponse(BaseModel):
    question: str = Field(description="객관식 문제의 질문 내용")
    table_data: Optional[str] = Field(default=None, description="문제에 표 데이터가 있는 경우 반드시 마크다운 표 문법(|---|)으로 여기에 작성 (없으면 null)")
    code_block: Optional[str] = Field(default=None, description="문제에 포함될 **소스 코드**나 **SQL 쿼리문**을 여기에 작성 (없으면 null)")
    options: List[str] = Field(description="4개의 보기 리스트 (예: ['1) 보기1', '2) 보기2', ...])")
    answer: int = Field(description="정답 번호 (1, 2, 3, 4 중 하나 정수형)")
    explanation: str = Field(description="정답 및 오답에 대한 상세한 해설")

class QuizVerification(BaseModel):
    is_valid: bool = Field(description="문제에 오류가 없고 출처에 기반한 완벽한 문제인지 여부 (True/False)")
    feedback: str = Field(description="불합격(False)인 경우 그 이유와 수정 방향, 합격(True)이면 '완벽함'이라고 작성")

class AITutorEngine:
    def __init__(self):
        print("[시스템] 메모리 텐서가 탑재된 파이프라인 초기화 중...")
        self.embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask", encode_kwargs={'normalize_embeddings': True})
        self.vector_db = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings, collection_metadata={"hnsw:space": "cosine"})
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        
        # JsonOutputParser 인스턴스화
        self.quiz_parser = JsonOutputParser(pydantic_object=QuizResponse)
        self.verify_parser = JsonOutputParser(pydantic_object=QuizVerification)
        print("[시스템] 튜터 엔진 가동 준비 완료!\n")

    def get_relevant_tensor(self, query: str, k: int = 3) -> str:
        docs = self.vector_db.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])

    def generate_response(self, query: str, chat_history: list, student_status: str = "분석된 상태 없음") -> str:
        # 기존 로직과 동일 (생략)
        context = self.get_relevant_tensor(query)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 정보처리기사 AI 튜터다...\n\n[학생상태]\n{student_status}\n\n[참고지식]\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "chat_history": chat_history, "student_status": student_status, "question": query})
    
    def generate_quiz(self, target_topic: str = None) -> dict:
        topics = ["요구사항 확인", "화면 설계", "데이터 입출력 구현", "통합 구현", "인터페이스 구현", "소프트웨어 개발 보안 구축", "응용 SW 기초 기술 활용"]
        selected_topic = target_topic if target_topic else random.choice(topics)
        docs = self.vector_db.similarity_search(selected_topic, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 국가공인 '정보처리기사' 자격증 시험을 출제하는 전담 교수다. 
주어진 [참고 지식]을 바탕으로 학생이 풀 수 있는 객관식 문제 1개를 출제해라.
{format_instructions}

[참고 지식]
{context}"""),
            ("human", f"위 지식을 바탕으로 '{selected_topic}' 파트에서 자격증 시험에 나올법한 객관식 문제를 하나 출제해줘.")
        ])

        # 체인에 JsonOutputParser 적용
        chain = prompt | self.llm | self.quiz_parser
        
        try:
            quiz_data = chain.invoke({
                "context": context,
                "format_instructions": self.quiz_parser.get_format_instructions()
            })
            quiz_data["topic"] = selected_topic 
            return quiz_data
        except Exception as e:
            print(f"[오류] 파싱 실패: {e}")
            return {
                "question": "데이터베이스 설계 순서로 올바른 것은? (파싱 오류 임시 문제)",
                "code_block": None,
                "options": ["1) 개념-논리-물리", "2) 물리-논리-개념", "3) 논리-개념-물리", "4) 개념-물리-논리"],
                "answer": 1,
                "explanation": "요구사항 분석 후 개념적, 논리적, 물리적 설계 순으로 진행됩니다.",
                "topic": selected_topic
            }
        
    # 검수 에이전트 함수
    def verify_quiz(self, quiz_data: dict, context: str) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 깐깐한 정보처리기사 문제 검수위원이다. 
            아래 [참고 지식]과 [출제된 문제]를 꼼꼼히 비교하여 다음 3가지를 검증해라:
            1. 정답이 확실히 맞으며, 해설이 논리적인가?
            2. 4개의 보기 중에 중복된 내용이 없는가?
            3. 문제가 [참고 지식]에 기반하고 있으며, 없는 내용을 지어내지(환각) 않았는가?

{format_instructions}"""),
            ("human", "[참고 지식]\n{context}\n\n[출제된 문제]\n{quiz}")
        ])
        
        chain = prompt | self.llm | self.verify_parser
        return chain.invoke({
            "context": context,
            "quiz": json.dumps(quiz_data, ensure_ascii=False),
            "format_instructions": self.verify_parser.get_format_instructions()
        })

    # 🚀 [수정] 피드백 루프가 적용된 출제 로직
    def generate_advanced_quiz(self, target_topic: str = None) -> dict:
        topics = ["요구사항 확인", "화면 설계", "데이터 입출력 구현", "통합 구현", "인터페이스 구현", "소프트웨어 개발 보안 구축", "응용 SW 기초 기술 활용"]
        selected_topic = target_topic if target_topic else random.choice(topics)
        
        quiz_docs = self.vector_db.similarity_search(query=selected_topic, k=2, filter={"doc_type": "quiz"})
        if not quiz_docs:
            return self.generate_quiz(selected_topic)
            
        context = "\n\n".join([doc.page_content for doc in quiz_docs])

        # 출제 위원 프롬프트
        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 정보처리기사 전문 출제위원이다. 
아래의 [실제 기출 데이터]를 분석하여 신규 변형 객관식 문제를 1개 출제해라.
[변형 규칙]
1. 핵심 개념 유지하되, 정답 보기나 상황을 새롭게 만들어라.
2. 매력적인 오답 보기를 포함해라.
3. 해설에는 "왜 정답이고, 왜 오답인지" 상세히 적어라.
4. 문제에 표(Table)나 릴레이션 데이터가 포함되어 있다면, 절대 누락하지 말고 반드시 JSON의 "table_data" 필드에 마크다운 표 형식으로 작성해라. (HTML 금지)
5. **문제에 소스 코드(C, Java, Python 등)나 SQL 쿼리문이 포함된다면, 절대 누락하지 말고 반드시 JSON의 "code_block" 필드에 작성해라.**

{format_instructions}

[실제 기출 데이터]
{context}

[이전 검수위원의 반려 피드백 (있을 경우 반영할 것)]
{feedback_history}"""),
            ("human", f"위 기출 데이터를 바탕으로 '{selected_topic}' 단원의 실전 변형 문제를 만들어줘.")
        ])

        chain = prompt | self.llm | self.quiz_parser
        
        # 🔄 에이전트 피드백 루프 (최대 3번 시도)
        max_retries = 3
        feedback_history = "피드백 없음 (최초 출제)"
        
        for attempt in range(max_retries):
            print(f"\n[에이전트] 출제 시도 {attempt + 1}/{max_retries}...")
            try:
                # 1. 출제 에이전트 가동
                quiz_data = chain.invoke({
                    "context": context,
                    "format_instructions": self.quiz_parser.get_format_instructions(),
                    "feedback_history": feedback_history
                })
                quiz_data["topic"] = selected_topic 
                
                # 2. 검수 에이전트 가동
                print("[에이전트] 검수위원이 문제를 검토 중입니다...")
                verification = self.verify_quiz(quiz_data, context)
                
                if verification["is_valid"]:
                    print("✅ [에이전트] 검수 통과! 완벽한 문제입니다.")
                    return quiz_data # 통과 시 즉시 반환
                else:
                    print(f"❌ [에이전트] 검수 반려! 사유: {verification['feedback']}")
                    feedback_history = verification['feedback'] # 반려 사유를 다음 출제에 반영
                    
            except Exception as e:
                print(f"[오류] 출제/검수 중 파싱 실패: {e}")
                # 오류 시 재시도
        
        # 3번 다 실패하거나 오류가 나면 마지막으로 만든 데이터를 일단 반환 (Fallback)
        print("⚠️ [에이전트] 최대 재시도 횟수 초과. 강제 반환합니다.")
        return quiz_data