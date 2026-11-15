import pymupdf


MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_TEXT_LENGTH = 100_000


LENGTH_GUIDE = {
    "short": "핵심 개념만 간단하게 정리하세요.",
    "medium": "핵심 개념과 중요한 설명을 적절한 분량으로 정리하세요.",
    "long": "핵심 개념, 특징, 관계와 중요한 내용을 자세하게 정리하세요.",
}


def extract_pdf_text(pdf_bytes: bytes):
    """
    텍스트 PDF에서 글자를 추출합니다.
    """

    if not pdf_bytes:
        raise ValueError("비어 있는 PDF 파일입니다.")

    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise ValueError(
            "현재 프로토타입에서는 20MB 이하의 PDF만 지원합니다."
        )

    try:
        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )
    except Exception as exc:
        raise ValueError(
            "PDF 파일을 열 수 없습니다."
        ) from exc

    pages = []

    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    f"[{page_number}페이지]\n{text}"
                )
    finally:
        document.close()

    full_text = "\n\n".join(pages).strip()

    if not full_text:
        raise ValueError(
            "PDF에서 읽을 수 있는 텍스트를 찾지 못했습니다. "
            "현재는 텍스트가 포함된 PDF만 지원합니다."
        )

    if len(full_text) > MAX_TEXT_LENGTH:
        raise ValueError(
            "현재 프로토타입에서 처리하기에는 PDF 내용이 너무 깁니다."
        )

    return full_text, len(pages)


def _message_text(content):
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts = []

        for item in content:
            if isinstance(item, str):
                texts.append(item)

            elif isinstance(item, dict):
                text = item.get("text")

                if text:
                    texts.append(text)

        return "\n".join(texts).strip()

    return str(content).strip()


def generate_pdf_summary(
    pdf_bytes: bytes,
    filename: str,
    answer_length: str,
    engine_factory,
):
    """
    PDF 텍스트를 추출하고 Gemini로 개념 요약을 생성합니다.
    """

    text, text_page_count = extract_pdf_text(
        pdf_bytes
    )

    length_guide = LENGTH_GUIDE.get(
        answer_length,
        LENGTH_GUIDE["medium"]
    )

    prompt = f"""
너는 학습 자료를 정리해 주는 AI 학습 튜터입니다.

아래 자료는 사용자가 업로드한 PDF에서 직접 추출한 내용입니다.

반드시 제공된 PDF 내용만 근거로 작성하세요.
PDF에 없는 내용을 임의로 추가하지 마세요.
자료의 과목이나 자격증 종류와 상관없이 내용 자체를 분석하세요.

[요약 분량]
{length_guide}

[작성 목적]
학생이 PDF 전체를 다시 읽지 않아도
중요한 개념을 복습할 수 있는 개념 정리 노트를 작성합니다.

[작성 형식]

# 핵심 개념 요약

## 1. 핵심 개념명

- 개념의 의미
- 중요한 특징
- 자료에서 강조하는 내용

## 2. 핵심 개념명

- 개념의 의미
- 중요한 특징
- 다른 개념과의 관계가 있다면 설명

필요한 만큼 계속 작성하세요.

마지막에는 반드시 다음 항목을 작성하세요.

## 최종 핵심 정리

- 반드시 기억해야 할 내용
- 반드시 기억해야 할 내용
- 반드시 기억해야 할 내용

문장을 단순히 줄이는 요약이 아니라
시험이나 복습에 사용할 수 있는
학습용 개념 정리 형태로 작성하세요.

[파일 이름]
{filename}

[PDF 내용]

{text}
"""

    engine = engine_factory()

    response = engine.llm.invoke(prompt)

    summary = _message_text(response.content)

    if not summary:
        raise ValueError(
            "AI가 개념 요약을 생성하지 못했습니다."
        )

    return {
        "filename": filename,
        "summary": summary,
        "text_page_count": text_page_count,
    }