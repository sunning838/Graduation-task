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

load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(CURRENT_DIR, "chroma_db")

# 자격증별 하위 도메인(폴더명) 매핑 딕셔너리
CERT_TOPICS = {
    "EIP": ["software_design", "software_development", "database", "programming_language", "info_system"],
    "LREA_1": ["civil_law", "housing_lease", "commercial_lease", "aggregate_building", "provisional_registration", "real_name_registration"]
}

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
        self.quiz_parser = JsonOutputParser(pydantic_object=QuizResponse)
        self.verify_parser = JsonOutputParser(pydantic_object=QuizVerification)
        print("[시스템] 튜터 엔진 가동 준비 완료!\n")

    
    def get_relevant_tensor(self, query: str, cert: str, k: int = 3) -> str:
        docs = self.vector_db.similarity_search(query, k=k, filter={"cert": cert})
        return "\n\n".join([doc.page_content for doc in docs])

    
    def generate_response(self, query: str, chat_history: list, student_status: str = "분석된 상태 없음", cert: str = "EIP") -> str:
        context = self.get_relevant_tensor(query, cert)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 자격증 전담 AI 튜터다...\n\n[학생상태]\n{student_status}\n\n[참고지식]\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "chat_history": chat_history, "student_status": student_status, "question": query})
    
    
    def generate_quiz(self, target_topic: str = None, cert: str = "EIP") -> dict:
        topics = CERT_TOPICS.get(cert, ["일반 개념"])
        selected_topic = target_topic if target_topic else random.choice(topics)
        
        # 특정 자격증(cert)이면서, 개념 데이터(doc_type=concept)인 것만 추출
        search_filter = {
            "$and": [
                {"doc_type": "concept"},
                {"cert": cert}
            ]
        }
        docs = self.vector_db.similarity_search(selected_topic, k=3, filter=search_filter)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 국가공인 자격증 시험을 출제하는 전담 교수다. 
주어진 [참고 지식]을 바탕으로 학생이 풀 수 있는 객관식 문제 1개를 출제해라.
{format_instructions}

[참고 지식]
{context}"""),
            ("human", f"위 지식을 바탕으로 '{selected_topic}' 파트에서 자격증 시험에 나올법한 객관식 문제를 하나 출제해줘.")
        ])

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
                "question": f"({selected_topic}) 데이터베이스 설계 순서로 올바른 것은? (파싱 오류 임시 문제)",
                "code_block": None,
                "options": ["1) 개념-논리-물리", "2) 물리-논리-개념", "3) 논리-개념-물리", "4) 개념-물리-논리"],
                "answer": 1,
                "explanation": "요구사항 분석 후 개념적, 논리적, 물리적 설계 순으로 진행됩니다.",
                "topic": selected_topic
            }
        
    def verify_quiz(self, quiz_data: dict, context: str) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 깐깐한 문제 검수위원이다. 
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
    
    def revise_quiz(self, original_quiz: dict, feedback: str, context: str) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 전문 출제위원이다.
네가 출제한 문제에 대해 검수위원이 오류를 발견하고 피드백을 주었다.
아래의 [출제된 문제]와 [검수위원 피드백], 그리고 [참고 지식]을 바탕으로 문제를 **수정(보완)**해라.
문제를 아예 새로 내는 것이 아니라, 피드백을 반영하여 기존 문제의 오류만 정확하게 바로잡아야 한다.

{format_instructions}"""),
            ("human", """[참고 지식]
{context}

[출제된 문제 (수정 대상)]
{original_quiz}

[검수위원 피드백]
{feedback}

이 피드백을 반영하여 문제를 완벽하게 수정해줘.""")
        ])
        
        chain = prompt | self.llm | self.quiz_parser
        return chain.invoke({
            "context": context,
            "original_quiz": json.dumps(original_quiz, ensure_ascii=False),
            "feedback": feedback,
            "format_instructions": self.quiz_parser.get_format_instructions()
        })
    
    

    
    def generate_advanced_quiz(self, target_topic: str = None, cert: str = "EIP") -> dict:
        topics = CERT_TOPICS.get(cert, ["일반 개념"])
        selected_topic = target_topic if target_topic else random.choice(topics)
        
        search_filter = {
            "$and": [
                {"doc_type": "quiz"},
                {"cert": cert}
            ]
        }
        
        quiz_docs = self.vector_db.similarity_search(query=selected_topic, k=5, filter=search_filter)
        if not quiz_docs: 
            return self.generate_quiz(selected_topic, cert)
            
        context = "\n\n".join([doc.page_content for doc in quiz_docs])

        # 최초 출제용 프롬프트 (feedback_history 제거)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 전문 출제위원이다. 
아래의 [실제 기출 데이터]를 분석하여 신규 변형 객관식 문제를 1개 출제해라.
[변형 규칙]
1. 핵심 개념 유지하되, 정답 보기나 상황을 새롭게 만들어라.
2. 매력적인 오답 보기를 포함해라.
3. 해설에는 "왜 정답이고, 왜 오답인지" 상세히 적어라.
4. 문제에 표(Table)나 릴레이션 데이터가 포함되어 있다면, 절대 누락하지 말고 반드시 JSON의 "table_data" 필드에 마크다운 표 형식으로 작성해라. (HTML 금지)
5. **문제에 소스 코드(C, Java, Python 등)나 SQL 쿼리문이 포함된다면, 절대 누락하지 말고 반드시 JSON의 "code_block" 필드에 작성해라.**

{format_instructions}"""),
            ("human", "[실제 기출 데이터]\n{context}\n\n위 기출 데이터를 바탕으로 '{selected_topic}' 단원의 실전 변형 문제를 만들어줘.")
        ])

        chain = prompt | self.llm | self.quiz_parser
        
        max_retries = 3
        feedback_history = ""
        quiz_data = None 
        
        for attempt in range(max_retries):
            print(f"\n[에이전트] {cert} 출제 시도 {attempt + 1}/{max_retries}...")
            try:
                if attempt == 0:
                    quiz_data = chain.invoke({
                        "context": context,
                        "selected_topic": selected_topic,
                        "format_instructions": self.quiz_parser.get_format_instructions()
                    })
                else:
                    # 검수위원의 피드백을 듣고 기존 문제를 수정함
                    print("[에이전트] 피드백을 반영하여 기존 문제를 수정 중입니다...")
                    quiz_data = self.revise_quiz(quiz_data, feedback_history, context)
                
                quiz_data["topic"] = selected_topic 
                
                print("[에이전트] 문제를 검토 중입니다...")
                verification = self.verify_quiz(quiz_data, context)
                
                if verification["is_valid"]:
                    print("✅ [에이전트] 검수 통과.")
                    return quiz_data 
                else:
                    print(f"❌ [에이전트] 검수 반려! 사유: {verification['feedback']}")
                    feedback_history = verification['feedback'] 
                    
            except Exception as e:
                print(f"🚨 [오류] 출제/수정/검수 중 파싱 실패: {e}")
        
        print("⚠️ [에이전트] 최대 재시도 횟수 초과. 수정된 결과물을 강제 반환합니다.")
        return quiz_data