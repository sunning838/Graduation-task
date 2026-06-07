import os
import re
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
from backend.cert_config import CERT_TOPICS

load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(CURRENT_DIR, "chroma_db")


class QuizResponse(BaseModel):
    question: str = Field(description="객관식 문제의 질문 내용")
    table_data: Optional[str] = Field(default=None, description="문제에 표 데이터가 있는 경우 반드시 마크다운 표 문법(|---|)으로 여기에 작성 (없으면 null)")
    code_block: Optional[str] = Field(default=None, description="문제에 포함될 **소스    코드**나 **SQL 쿼리문**을 여기에 작성 (없으면 null)")
    options: List[str] = Field(description="4개의 보기 리스트 (예: ['1) 보기1', '2) 보기2', ...])")
    answer: int = Field(description="정답 번호 (1, 2, 3, 4 중 하나 정수형)")
    explanation: str = Field(description="정답 및 오답에 대한 상세한 해설")

class QuizVerification(BaseModel):
    is_valid: bool = Field(description="문제에 오류가 없고 출처에 기반한 완벽한 문제인지 여부 (True/False)")
    feedback: str = Field(description="불합격(False)인 경우 그 이유와 수정 방향, 합격(True)이면 '완벽함'이라고 작성")

class EssayGradeResult(BaseModel):
    score: int = Field(description="0~100 종합 점수")
    keyword_score: int = Field(description="키워드 점수 (40점 만점)")
    logic_score: int = Field(description="논리 흐름 점수 (30점 만점)")
    accuracy_score: int = Field(description="오개념 없음 점수 (30점 만점)")
    found_keywords: List[str] = Field(description="학생이 언급한 핵심 키워드 리스트")
    missing_keywords: List[str] = Field(description="누락된 핵심 키워드 리스트")
    misconceptions: List[str] = Field(description="발견된 오개념 (없으면 빈 리스트)")
    feedback: str = Field(description="AI 피드백 2~3문장")
    model_answer_hint: str = Field(description="모범 답안 힌트 (정답 직접 노출 금지)")

class SocraticTurnResult(BaseModel):
    status: str = Field(description="'continue' 또는 'complete'. 학생이 정답/원리를 깨달으면 complete")
    question: str = Field(description="다음 유도 질문 (status가 continue일 때만 작성, 정답 직접 언급 금지)")
    acknowledgment: str = Field(description="학생 답변에서 맞는 부분을 칭찬/인정해주는 멘트")
    hint_level: int = Field(description="현재 힌트 수준 (0~3)")

class UnderstandingMap(BaseModel):
    concept: str = Field(description="최종적으로 이해한 핵심 개념")
    struggle_points: List[str] = Field(description="학생이 처음에 헷갈려했던 부분들")
    breakthrough_moment: str = Field(description="이해의 전환점이 된 질문이나 순간")
    confidence_level: str = Field(description="이해도 (상 / 중 / 하)")
    recommended_next: str = Field(description="다음에 공부하면 좋을 연관 개념")


class AITutorEngine:
    def __init__(self):
        print("[시스템] 메모리 텐서가 탑재된 파이프라인 초기화 중...")
        self.embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask", encode_kwargs={'normalize_embeddings': True})
        self.vector_db = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings, collection_metadata={"hnsw:space": "cosine"})
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        self.quiz_parser = JsonOutputParser(pydantic_object=QuizResponse)
        self.verify_parser = JsonOutputParser(pydantic_object=QuizVerification)
        self.essay_parser = JsonOutputParser(pydantic_object=EssayGradeResult)
        self.socratic_parser = JsonOutputParser(pydantic_object=SocraticTurnResult)
        self.map_parser = JsonOutputParser(pydantic_object=UnderstandingMap)
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
    
    def _build_avoid_prompt(self, generated_history: list = None) -> str:
        if not generated_history:
             return ""

        past_questions = [
            q.get("question", "").strip()
            for q in generated_history
            if isinstance(q, dict) and q.get("question")
        ]

        if not past_questions:
            return ""

        avoid_prompt = """
        [중복 방지 규칙]
        - 아래 문제들과 동일한 문장, 동일한 보기 구성, 동일한 정답 패턴을 반복하지 마라.
        - 같은 단원이라도 새로운 상황, 새로운 보기, 새로운 표현으로 변형하라.
        - 기존 문제와 지나치게 유사한 문제는 출제하지 마라.

        [기존 출제 문제 목록]
        """
        avoid_prompt += "\n".join([f"- {pq}" for pq in past_questions])
        return avoid_prompt


    def _fallback_quiz(self, selected_topic: str, reason: str = "문제 생성 실패") -> dict:
        return {
            "question": f"({selected_topic}) 다음 중 해당 개념에 대한 설명으로 가장 적절한 것은? ({reason})",
            "table_data": None,
            "code_block": None,
            "options": [
                "1) 핵심 개념을 올바르게 설명한 보기이다.",
                "2) 핵심 개념과 관련 없는 설명이다.",
                "3) 일부만 맞고 전체적으로는 틀린 설명이다.",
                "4) 시험에서 일반적으로 옳지 않은 설명이다."
            ],
            "answer": 1,
            "explanation": "문제 생성 과정에서 오류가 발생하여 임시 문항이 반환되었습니다. 실제 시험 대비용으로는 다시 생성하는 것을 권장합니다.",
            "topic": selected_topic
        }
    
    def _normalize_quiz_data(self, quiz_data: dict, selected_topic: str) -> dict:
        if not quiz_data:
            return self._fallback_quiz(selected_topic, "빈 문제 데이터")

        options = quiz_data.get("options", [])
        answer = quiz_data.get("answer", 1)

        # options 형식 검사
        if not isinstance(options, list):
            return self._fallback_quiz(selected_topic, "options 형식 오류")

        cleaned_options = []
        for opt in options:
            if opt is None:
                continue
            text = str(opt).strip()
            if not text:
                continue

            # 앞에 붙은 번호 제거: "1) 보기", "2. 보기", "3 - 보기" 등
            text = re.sub(r'^\s*\d+\s*[).\-:]?\s*', '', text)
            cleaned_options.append(text)

        # 보기 4개 초과 시 앞의 4개만 사용
        cleaned_options = cleaned_options[:4]

        # 보기 수가 4개 아니면 fallback
        if len(cleaned_options) != 4:
            return self._fallback_quiz(selected_topic, "보기 개수 오류")

        # 번호를 강제로 다시 붙임
        normalized_options = [
            f"{i+1}) {text}" for i, text in enumerate(cleaned_options)
        ]

        # 정답 번호 보정
        try:
            answer = int(answer)
        except Exception:
            return self._fallback_quiz(selected_topic, "정답 형식 오류")

        if answer < 1 or answer > 4:
            return self._fallback_quiz(selected_topic, "정답 번호 범위 오류")

        quiz_data["options"] = normalized_options
        quiz_data["answer"] = answer
        quiz_data["topic"] = selected_topic
        return quiz_data

    
    
    def generate_quiz(self, target_topic: str = None, cert: str = "EIP", generated_history: list = None) -> dict:
        topics = CERT_TOPICS.get(cert, ["일반 개념"])
        selected_topic = target_topic if target_topic else random.choice(topics)

        search_filter = {
            "$and": [
                {"doc_type": "concept"},
                {"cert": cert}
            ]
        }

        docs = self.vector_db.similarity_search(selected_topic, k=10, filter=search_filter)
        if not docs:
            print(f"⚠️ [경고] {cert} / {selected_topic} 관련 concept 문서를 찾지 못했습니다.")
            return self._fallback_quiz(selected_topic, "참고 지식 없음")

        sampled_docs = random.sample(docs, min(3, len(docs)))
        context = "\n\n".join([doc.page_content for doc in sampled_docs])

        avoid_prompt = self._build_avoid_prompt(generated_history)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 국가공인 자격증 시험을 출제하는 전담 교수다.
                        주어진 [참고 지식]을 바탕으로 학생이 풀 수 있는 객관식 문제 1개를 출제해라.

                        {avoid_prompt}

                        [출제 규칙]
                        1. 반드시 보기 4개를 작성해라.
                        2. 정답은 1~4 중 하나의 정수로 작성해라.
                        3. 해설은 왜 정답이고 왜 오답인지 분명하게 작성해라.
                        4. 표가 필요하면 table_data에 넣고, 없으면 null로 둬라.
                        5. 코드나 SQL이 필요하면 code_block에 넣고, 없으면 null로 둬라.

{format_instructions}

[참고 지식]
{context}"""),
            ("human", "위 지식을 바탕으로 '{selected_topic}' 파트에서 자격증 시험에 나올 법한 객관식 문제를 하나 출제해줘.")
        ])

        chain = prompt | self.llm | self.quiz_parser

        try:
            quiz_data = chain.invoke({
                "context": context,
                "selected_topic": selected_topic,
                "avoid_prompt": avoid_prompt,
                "format_instructions": self.quiz_parser.get_format_instructions()
            })
            quiz_data = self._normalize_quiz_data(quiz_data, selected_topic)
            return quiz_data

        except Exception as e:
            print(f"[오류] generate_quiz 파싱 실패: {e}")
            return self._fallback_quiz(selected_topic, "파싱 오류")
        
    def verify_quiz(self, quiz_data: dict, context: str) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 깐깐한 문제 검수위원이다. 
            아래 [참고 지식]과 [출제된 문제]를 꼼꼼히 비교하여 다음 3가지를 검증해라:
            1. 정답이 확실히 맞으며, 해설이 논리적인가?
            2. 보기 개수가 정확히 4개인가?
            3. 정답 번호가 1~4 범위 안에 있는가?
            4. 4개의 보기 중에 중복된 내용이 없는가?
            5. 문제가 [참고 지식]에 기반하고 있으며, 없는 내용을 지어내지(환각) 않았는가?

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
        if not original_quiz:
            raise ValueError("revise_quiz에 original_quiz가 없습니다.")
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
        quiz_data = chain.invoke({
            "context": context,
            "original_quiz": json.dumps(original_quiz, ensure_ascii=False),
            "feedback": feedback,
            "format_instructions": self.quiz_parser.get_format_instructions()
        })

        selected_topic = original_quiz.get("topic", "일반 개념")
        quiz_data = self._normalize_quiz_data(quiz_data, selected_topic)
        return quiz_data
    

    
    def generate_advanced_quiz(self, target_topic: str = None, cert: str = "EIP",  generated_history: list = None) -> dict:
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
            return self.generate_quiz(
                target_topic=selected_topic,
                cert=cert,
                generated_history=generated_history
            )
    
            
        context = "\n\n".join([doc.page_content for doc in quiz_docs])

        # 이미 출제한 문제들과 너무 비슷한 문제를 피하도록 지시문 생성
        avoid_prompt = self._build_avoid_prompt(generated_history)

        # 최초 출제용 프롬프트
        prompt = ChatPromptTemplate.from_messages([
        ("system", """너는 전문 출제위원이다. 
아래의 [실제 기출 데이터]를 분석하여 신규 변형 객관식 문제를 1개 출제해라.

{avoid_prompt}

[변형 규칙]
1. 핵심 개념 유지하되, 정답 보기나 상황을 새롭게 만들어라.
2. 매력적인 오답 보기를 포함해라.
3. 해설에는 "왜 정답이고, 왜 오답인지" 상세히 적어라.
4. 문제에 표(Table)나 릴레이션 데이터가 포함되어 있다면, 절대 누락하지 말고 반드시 JSON의 "table_data" 필드에 마크다운 표 형식으로 작성해라. (HTML 금지)
5. 문제에 소스 코드(C, Java, Python 등)나 SQL 쿼리문이 포함된다면, 절대 누락하지 말고 반드시 JSON의 "code_block" 필드에 작성해라.

{format_instructions}"""),
        ("human", "[실제 기출 데이터]\n{context}\n\n위 기출 데이터를 바탕으로 '{selected_topic}' 단원의 실전 변형 문제를 만들어줘.")
    ])

        chain = prompt | self.llm | self.quiz_parser
        
        max_retries = 5
        feedback_history = ""
        quiz_data = None 
        
        for attempt in range(max_retries):
            print(f"\n[에이전트] {cert} 출제 시도 {attempt + 1}/{max_retries}...")
            try:
                if attempt == 0:
                    quiz_data = chain.invoke({
                        "context": context,
                        "selected_topic": selected_topic,
                        "avoid_prompt": avoid_prompt,
                        "format_instructions": self.quiz_parser.get_format_instructions()
                    })
                else:
                    # 검수위원의 피드백을 듣고 기존 문제를 수정함
                    print("[에이전트] 피드백을 반영하여 기존 문제를 수정 중입니다...")
                    quiz_data = self.revise_quiz(quiz_data, feedback_history, context)
                
                quiz_data = self._normalize_quiz_data(quiz_data, selected_topic)
                
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
        
        if quiz_data is None:
            return self._fallback_quiz(selected_topic, "최대 재시도 초과")
        return quiz_data
    
    # 서술형 채점 에이전트
    def grade_essay(self, quiz_question: str, cert: str, student_essay: str) -> dict:
        print(f"\n[에이전트] 서술형 답안 채점 중...")
        

        search_filter = {
            "$and": [
                {"doc_type": "concept"},
                {"cert": cert}
            ]
        }
        docs = self.vector_db.similarity_search(query=quiz_question, k=2, filter=search_filter)
        context = "\n".join([doc.page_content for doc in docs])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 엄격하지만 친절한 채점관이다.
아래 [출제되었던 문제]와 [참고 지식]을 기준으로 학생의 [서술형 답안]을 평가해라.
학생은 [출제되었던 문제]에서 다룬 핵심 개념에 대해 서술한 것이다.
반드시 아래 기준을 지켜라:
- 핵심 키워드 포함 (40점)
- 논리 흐름 (30점)
- 오개념 여부 (30점, 오개념이 없으면 30점 만점)

{format_instructions}"""),
            ("human", "[출제되었던 문제]\n{quiz_question}\n\n[참고 지식]\n{context}\n\n[학생 서술형 답안]\n{student_essay}")
        ])
        
        chain = prompt | self.llm | self.essay_parser
        return chain.invoke({
            "context": context, 
            "quiz_question": quiz_question,  #LLM에게 무슨 문제였는지 알려줌
            "student_essay": student_essay,
            "format_instructions": self.essay_parser.get_format_instructions()
        })
    
    # 🚀 기능 B-1: 소크라테스 꼬리 질문 에이전트
    def socratic_dialogue_turn(self, topic: str, cert: str, quiz_question: str, history: str, student_answer: str, hint_level: int) -> dict:
        context = self.get_relevant_tensor(topic, cert, k=2)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 학생 스스로 정답을 깨닫게 만드는 소크라테스식 1타 강사다.
[규칙]
1. 절대 정답을 직접 말하지 않는다.
2. 학생 답변의 맞는 부분은 인정하고, 빠지거나 틀린 부분만 꼬리 질문으로 유도한다.
3. 한 번에 하나의 질문만 간결하게 한다.
4. 현재 힌트 레벨은 {hint_level}이다. (0~3). 레벨이 높을수록 결정적인 힌트를 제공해라.
5. 학생이 스스로 객관식 문제의 정답과 원리를 깨달았다고 판단되면 status를 'complete'로 반환해라.

{format_instructions}"""),
            ("human", "[참고 지식]\n{context}\n\n[틀린 객관식 문제]\n{quiz_question}\n\n[이전 대화 기록]\n{history}\n\n[학생의 최근 답변]\n{student_answer}\n\n위 내용을 바탕으로 다음 행동을 결정해라.")
        ])
        chain = prompt | self.llm | self.socratic_parser
        return chain.invoke({
            "context": context, "quiz_question": quiz_question, 
            "history": history, "student_answer": student_answer, 
            "hint_level": hint_level, "format_instructions": self.socratic_parser.get_format_instructions()
        })

    # 🚀 기능 B-2: 이해 경로 맵 생성 에이전트
    def generate_understanding_map(self, history: str) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 학습 분석가다. 아래 소크라테스 문답 기록을 분석하여 학생의 '이해 경로 맵'을 생성해라.
{format_instructions}"""),
            ("human", "[소크라테스 문답 기록]\n{history}")
        ])
        chain = prompt | self.llm | self.map_parser
        return chain.invoke({"history": history, "format_instructions": self.map_parser.get_format_instructions()})
    
    
    # 오답노트 자동 생성 에이전트
    def generate_final_note(self, cert: str, topics: list) -> str:
        if not topics:
            return "오답 기록이 충분하지 않습니다. 모의고사를 더 풀어주세요!"

        print(f"\n[에이전트] {cert} 취약 단원({topics}) 파이널 요약노트 작성 중...")
        
        # 1. DB에서 취약 단원들의 핵심 개념(concept)만 핀셋으로 긁어모으기
        combined_context = ""
        for topic in topics:
            search_filter = {
                "$and": [
                    {"doc_type": "concept"},
                    {"cert": cert}
                ]
            }
            # 단원별로 가장 관련도 높은 지식 3개씩 추출
            docs = self.vector_db.similarity_search(topic, k=3, filter=search_filter)
            combined_context += f"\n\n### [{topic}] 파트 핵심 지식 ###\n"
            combined_context += "\n".join([doc.page_content for doc in docs])

        # 2. 일타 강사 프롬프트 작성
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 자격증 합격을 돕는 대한민국 최고의 1타 강사입니다.
아래 주어지는 [취약 단원]의 지식 데이터를 바탕으로 내일 시험장에 들고 갈 '파이널 요약 오답노트'를 작성해야 합니다.

[오답노트 작성 규칙 - 반드시 지킬 것]
1. 핵심 구조화: 지식을 단순히 나열하지 말고, 핵심 키워드 위주로 굵은 글씨(**)와 글머리 기호(-, *)를 사용해 직관적으로 정리하세요.
2. 1타 강사의 암기 비법 (필수 포함): 각 단원마다 수험생의 뇌리에 박히는 암기 팁을 반드시 하나 이상 제공하세요. 지식의 성격에 따라 다음 두 가지 방식 중 하나를 선택하세요.
   - [두음문자형]: 여러 종류, 순서, 특징을 나열해야 한다면 핵심 키워드의 앞글자를 딴 기발한 단어를 만드세요.
     (예: 정보처리기사 럼바우 모델링 -> '객동기' / 결합도 순서 -> '내공외제스자')
   - [스토리텔링형]: 법령, 판례, 인과관계, 원리 등이라면 상황을 연상할 수 있는 재미있는 한 줄 스토리를 만드세요.
     (예: 공인중개사 비진의표시 -> "비밀리에 진짜 의도를 숨겼으니 원칙적으로 유효!")
3. 시험장 주의사항: 출제자가 헷갈리게 내는 '오답 함정 패턴'이나 다른 개념과 혼동하지 말아야 할 포인트를 명시하세요.
4. 마크다운(Markdown) 형식을 엄격하게 준수하여 모바일 화면에서도 보기 좋게 출력하세요.
"""),
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": combined_context})