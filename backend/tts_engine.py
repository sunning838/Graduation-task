from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import logging
import os
import re
import time
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests
from dotenv import load_dotenv


LOG = logging.getLogger(__name__)


# ============================================================
# 경로
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

CACHE_ROOT = BACKEND_DIR / "storage" / "audio_cache"

SEGMENT_CACHE_DIR = CACHE_ROOT / "segments"
PACKAGE_CACHE_DIR = CACHE_ROOT / "packages"


# ============================================================
# Gemini TTS 설정
# ============================================================

TTS_MODEL = "gemini-3.8-flash-lite-tts"

VOICE_NAME = "Kore"

TTS_STYLE = (
    "calm and clear Korean certification instructor, "
    "slightly slow pace, natural educational narration, "
    "emphasize important technical terms gently, "
    "avoid exaggerated acting"
)

TTS_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{TTS_MODEL}:generateContent"
)


# 문단 사이 짧은 쉼
SEGMENT_GAP_SECONDS = 0.22

MAX_RETRIES = 3


# ============================================================
# 환경변수
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")

SEGMENT_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PACKAGE_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 공통 함수
# ============================================================

def _sha256(text: str) -> str:

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# Markdown 문단 분리
# ============================================================

def _split_markdown_blocks(
    markdown: str,
) -> list[str]:

    """
    강의 Markdown을 문단 단위로 나눈다.

    코드 블록 내부의 빈 줄 때문에
    코드가 잘리지 않도록 fenced code를 고려한다.
    """

    lines = (
        markdown or ""
    ).replace(
        "\r\n",
        "\n",
    ).split("\n")

    blocks: list[str] = []

    current: list[str] = []

    in_fence = False


    def flush():

        if current:

            block = "\n".join(
                current
            ).strip()

            if block:
                blocks.append(
                    block
                )

            current.clear()


    for line in lines:

        stripped = line.strip()

        # 코드 블록 시작 / 종료
        if stripped.startswith("```"):

            current.append(
                line
            )

            in_fence = (
                not in_fence
            )

            continue


        # 코드 외부의 빈 줄은 문단 구분
        if (
            not in_fence
            and not stripped
        ):

            flush()

            continue


        current.append(
            line
        )


    flush()

    return blocks


# ============================================================
# 소제목 확인
# ============================================================

def _is_heading(
    block: str,
) -> bool:

    lines = [

        line

        for line in block.splitlines()

        if line.strip()

    ]

    return (
        len(lines) == 1
        and bool(
            re.match(
                r"^\s{0,3}#{1,6}\s+",
                lines[0],
            )
        )
    )


# ============================================================
# 제목 + 다음 문단 묶기
# ============================================================

def _group_blocks(
    blocks: list[str],
) -> list[str]:

    """
    '### 캡슐화' 같은 제목만
    TTS 한 번 호출하지 않도록

    제목 + 다음 내용

    을 하나의 음성 구간으로 묶는다.
    """

    grouped = []

    i = 0

    while i < len(blocks):

        block = blocks[i]

        if (
            _is_heading(block)
            and i + 1 < len(blocks)
        ):

            grouped.append(

                block
                + "\n\n"
                + blocks[i + 1]

            )

            i += 2

        else:

            grouped.append(
                block
            )

            i += 1


    return grouped


# ============================================================
# Markdown → 음성용 텍스트
# ============================================================

def _markdown_to_speech(
    block: str,
) -> str:

    """
    화면 Markdown을
    TTS가 읽기 편한 텍스트로 바꾼다.
    """

    text = html.unescape(
        block or ""
    )


    # --------------------------------------------------------
    # 코드 블록
    # --------------------------------------------------------

    if re.fullmatch(
        r"\s*```[\s\S]*?```\s*",
        text,
    ):

        return (
            "화면에 표시된 코드 예시는 "
            "직접 확인해 주세요."
        )


    text = re.sub(

        r"```[\s\S]*?```",

        "\n화면에 표시된 코드 예시는 "
        "직접 확인해 주세요.\n",

        text,

    )


    # --------------------------------------------------------
    # Markdown 링크 / 이미지
    # --------------------------------------------------------

    text = re.sub(

        r"!\[([^\]]*)\]\([^)]+\)",

        r"\1",

        text,

    )

    text = re.sub(

        r"\[([^\]]+)\]\([^)]+\)",

        r"\1",

        text,

    )


    # --------------------------------------------------------
    # 제목
    # --------------------------------------------------------

    text = re.sub(

        r"(?m)^\s{0,3}#{1,6}\s*",

        "",

        text,

    )


    # --------------------------------------------------------
    # 인용
    # --------------------------------------------------------

    text = re.sub(

        r"(?m)^\s*>\s?",

        "",

        text,

    )


    # --------------------------------------------------------
    # 목록
    # --------------------------------------------------------

    lines = []

    for line in text.splitlines():

        line = re.sub(

            r"^\s*[-*+]\s+",

            "",

            line,

        )

        line = re.sub(

            r"^\s*\d+[.)]\s+",

            "",

            line,

        )

        line = line.strip()

        if line:

            lines.append(
                line
            )


    # 목록 사이에 자연스러운 쉼
    text = ". ".join(
        lines
    )


    # --------------------------------------------------------
    # Markdown 강조
    # --------------------------------------------------------

    text = (
        text
        .replace("**", "")
        .replace("__", "")
        .replace("`", "")
    )


    text = re.sub(

        r"(?<!\w)[*_](.+?)[*_](?!\w)",

        r"\1",

        text,

    )


    # HTML 제거
    text = re.sub(

        r"<[^>]+>",

        " ",

        text,

    )


    # URL 제거
    text = re.sub(

        r"https?://\S+",

        "",

        text,

    )


    # 공백 정리
    text = re.sub(

        r"\s+",

        " ",

        text,

    ).strip()


    # 잘못 겹친 문장부호 정리
    text = re.sub(

        r"\.\s*([.!?])",

        r"\1",

        text,

    )


    return text


# ============================================================
# 실제 TTS 구간 생성
# ============================================================

def build_segments(
    lecture_markdown: str,
) -> list[dict]:

    blocks = _split_markdown_blocks(
        lecture_markdown
    )

    blocks = _group_blocks(
        blocks
    )

    segments = []


    for block in blocks:

        speech = (
            _markdown_to_speech(
                block
            )
        )

        if not speech:
            continue


        segments.append(
            {
                "index": len(segments),
                "markdown": block,
                "speech": speech,
            }
        )


    return segments


# ============================================================
# 각 음성 구간 캐시
# ============================================================

def _segment_cache_path(
    speech: str,
    voice: str,
    style: str,
) -> Path:

    identity = json.dumps(

        {
            "model": TTS_MODEL,
            "voice": voice,
            "style": style,
            "speech": speech,
        },

        ensure_ascii=False,

        sort_keys=True,

    )


    return (

        SEGMENT_CACHE_DIR
        / f"{_sha256(identity)}.wav"

    )


# ============================================================
# 전체 강의 캐시
# ============================================================

def _package_dir(
    lesson: dict,
    variant: str,
    lecture_text: str,
) -> Path:

    identity = json.dumps(

        {
            "model": TTS_MODEL,
            "voice": VOICE_NAME,
            "style": TTS_STYLE,

            "lesson_id":
                lesson.get(
                    "id",
                    "",
                ),

            "lesson_version":
                lesson.get(
                    "version",
                    "",
                ),

            "variant":
                variant,

            "lecture_text":
                lecture_text,
        },

        ensure_ascii=False,

        sort_keys=True,

    )


    return (

        PACKAGE_CACHE_DIR
        / _sha256(identity)

    )


# ============================================================
# API Key
# ============================================================

def _api_key():

    key = (

        os.getenv(
            "GEMINI_API_KEY"
        )

        or

        os.getenv(
            "GOOGLE_API_KEY"
        )

    )


    if not key:

        raise RuntimeError(
            ".env에 GEMINI_API_KEY 또는 "
            "GOOGLE_API_KEY가 없습니다."
        )


    return key


# ============================================================
# Gemini 응답에서 음성 찾기
# ============================================================

def _extract_audio_bytes(
    payload: dict,
) -> bytes:

    for candidate in (
        payload.get("candidates")
        or []
    ):

        content = (
            candidate.get("content")
            or {}
        )

        for part in (
            content.get("parts")
            or []
        ):

            inline = (

                part.get(
                    "inlineData"
                )

                or

                part.get(
                    "inline_data"
                )

            )


            if (
                isinstance(
                    inline,
                    dict,
                )
                and inline.get("data")
            ):

                try:

                    return (
                        base64.b64decode(
                            inline["data"]
                        )
                    )

                except (
                    ValueError,
                    TypeError,
                ) as error:

                    raise RuntimeError(
                        "TTS 오디오 Base64 "
                        "디코딩에 실패했습니다."
                    ) from error


    raise RuntimeError(
        "Gemini TTS 응답에 "
        "오디오 데이터가 없습니다."
    )


# ============================================================
# WAV 검사
# ============================================================

def _validate_wav_bytes(
    data: bytes,
):

    try:

        with wave.open(
            io.BytesIO(data),
            "rb",
        ) as wav_file:

            if (
                wav_file.getnchannels()
                != 1
            ):

                raise RuntimeError(
                    "TTS WAV가 mono 형식이 아닙니다."
                )


            if (
                wav_file.getsampwidth()
                != 2
            ):

                raise RuntimeError(
                    "TTS WAV가 16-bit PCM 형식이 아닙니다."
                )


            if (
                wav_file.getframerate()
                != 24000
            ):

                raise RuntimeError(
                    "TTS WAV가 24 kHz 형식이 아닙니다."
                )


    except wave.Error as error:

        raise RuntimeError(
            "Gemini TTS가 유효한 WAV를 "
            "반환하지 않았습니다."
        ) from error


# ============================================================
# Gemini TTS 요청
# ============================================================

def _request_tts_wav(
    speech: str,
    voice: str,
    style: str,
) -> bytes:

    """
    실제 Gemini TTS API 호출.
    """

    body = {

        "contents": [

            {
                "role": "user",

                "parts": [

                    {
                        # 읽어야 할 실제 문장만 넣는다.
                        "text": speech,

                        # 말투/속도는 따로 전달
                        "speech_metadata": {
                            "style": style
                        },

                    }

                ],

            }

        ],


        "generationConfig": {

            "responseModalities": [
                "AUDIO"
            ],

            "speechConfig": {

                "voiceConfig": {

                    "voice":
                        voice,

                }

            },

        },

    }


    last_error = None


    for attempt in range(
        MAX_RETRIES
    ):

        try:

            response = requests.post(

                TTS_URL,

                headers={

                    "x-goog-api-key":
                        _api_key(),

                    "Content-Type":
                        "application/json",

                },

                json=body,

                timeout=(
                    10,
                    120,
                ),

            )


            # 서버 오류 / 사용량 제한
            if (
                response.status_code
                == 429
                or
                response.status_code
                >= 500
            ):

                raise RuntimeError(
                    "TTS 서버 일시 오류: "
                    f"HTTP {response.status_code}"
                )


            if not response.ok:

                raise RuntimeError(

                    "TTS 요청 실패: "
                    f"HTTP {response.status_code}"
                    " - "
                    f"{response.text[:500]}"

                )


            audio = (
                _extract_audio_bytes(
                    response.json()
                )
            )


            _validate_wav_bytes(
                audio
            )


            return audio


        except (
            requests.RequestException,
            RuntimeError,
            ValueError,
        ) as error:

            last_error = error


            if (
                attempt + 1
                < MAX_RETRIES
            ):

                time.sleep(

                    1.2
                    * (2 ** attempt)

                )


    raise RuntimeError(
        "AI 음성 강의를 생성하지 못했습니다."
    ) from last_error


# ============================================================
# 문단 음성 캐시 생성
# ============================================================

def _ensure_segment_wav(
    speech: str,
    voice: str,
    style: str,
) -> Path:

    path = (
        _segment_cache_path(
            speech,
            voice,
            style,
        )
    )


    # 기존 캐시
    if path.exists():

        try:

            _validate_wav_bytes(
                path.read_bytes()
            )

            return path

        except RuntimeError:

            LOG.warning(
                "Corrupt TTS segment cache removed: %s",
                path,
            )

            path.unlink(
                missing_ok=True
            )


    # 새 음성 생성
    audio = (
        _request_tts_wav(
            speech,
            voice,
            style,
        )
    )


    # 중간에 프로그램이 꺼져도
    # 손상된 파일이 캐시로 남지 않도록 임시파일 사용
    with NamedTemporaryFile(

        dir=SEGMENT_CACHE_DIR,

        suffix=".wav",

        delete=False,

    ) as temp:

        temp_path = Path(
            temp.name
        )

        temp.write(
            audio
        )


    temp_path.replace(
        path
    )


    return path


# ============================================================
# 문단 WAV들을 하나로 합치기
# 동시에 실제 시간을 계산
# ============================================================

def _combine_wavs(
    segment_paths: list[Path],
    segments: list[dict],
    output: Path,
) -> list[dict]:

    if not segment_paths:

        raise ValueError(
            "합칠 TTS 음성이 없습니다."
        )


    timeline = []

    elapsed = 0.0

    reference_format = None


    with NamedTemporaryFile(

        dir=output.parent,

        suffix=".wav",

        delete=False,

    ) as temp:

        temp_path = Path(
            temp.name
        )


    try:

        with wave.open(

            str(temp_path),

            "wb",

        ) as out:


            for index, (
                path,
                segment,
            ) in enumerate(

                zip(
                    segment_paths,
                    segments,
                )

            ):


                with wave.open(

                    str(path),

                    "rb",

                ) as source:


                    current_format = (

                        source.getnchannels(),

                        source.getsampwidth(),

                        source.getframerate(),

                    )


                    # 첫 WAV 형식 저장
                    if reference_format is None:

                        reference_format = (
                            current_format
                        )

                        out.setnchannels(
                            current_format[0]
                        )

                        out.setsampwidth(
                            current_format[1]
                        )

                        out.setframerate(
                            current_format[2]
                        )


                    elif (
                        current_format
                        != reference_format
                    ):

                        raise RuntimeError(
                            "TTS 구간들의 WAV 형식이 "
                            "서로 다릅니다."
                        )


                    # 실제 WAV 프레임
                    frames = (
                        source.readframes(
                            source.getnframes()
                        )
                    )


                    # ★ 핵심
                    # 사람이 시간을 재는 것이 아니라
                    # 실제 프레임 수로 시간을 계산
                    duration = (

                        source.getnframes()
                        /
                        source.getframerate()

                    )


                    start = elapsed

                    end = (
                        start
                        + duration
                    )


                    out.writeframes(
                        frames
                    )


                    # 타임라인 기록
                    timeline.append(
                        {
                            "index":
                                segment[
                                    "index"
                                ],

                            "start":
                                round(
                                    start,
                                    3,
                                ),

                            "end":
                                round(
                                    end,
                                    3,
                                ),

                            "markdown":
                                segment[
                                    "markdown"
                                ],

                            "speech":
                                segment[
                                    "speech"
                                ],
                        }
                    )


                    elapsed = end


                    # 마지막 문단이 아니면
                    # 짧은 무음 삽입
                    if (
                        index + 1
                        < len(
                            segment_paths
                        )
                    ):

                        (
                            channels,
                            sample_width,
                            frame_rate,

                        ) = reference_format


                        silence_frames = int(

                            frame_rate
                            * SEGMENT_GAP_SECONDS

                        )


                        out.writeframes(

                            b"\x00"
                            * silence_frames
                            * channels
                            * sample_width

                        )


                        elapsed += (
                            SEGMENT_GAP_SECONDS
                        )


        temp_path.replace(
            output
        )


    finally:

        temp_path.unlink(
            missing_ok=True
        )


    return timeline


# ============================================================
# 캐시된 전체 강의 읽기
# ============================================================

def _load_package(
    package_dir: Path,
):

    audio_path = (
        package_dir
        / "lecture.wav"
    )

    timeline_path = (
        package_dir
        / "timeline.json"
    )


    if (
        not audio_path.exists()
        or
        not timeline_path.exists()
    ):

        return None


    try:

        timeline = json.loads(

            timeline_path
            .read_text(
                encoding="utf-8"
            )

        )


        if (
            not isinstance(
                timeline,
                list,
            )
            or not timeline
        ):

            return None


        _validate_wav_bytes(
            audio_path.read_bytes()
        )


        return {

            "audio_path":
                audio_path,

            "timeline_path":
                timeline_path,

            "timeline":
                timeline,

        }


    except (
        OSError,
        json.JSONDecodeError,
        RuntimeError,
    ):

        LOG.warning(
            "Invalid TTS package cache: %s",
            package_dir,
        )

        return None


# ============================================================
# 외부용: 캐시 확인
# ============================================================

def get_cached_lesson_package(
    lesson: dict,
    variant: str,
    lecture_text: str,
):

    return _load_package(

        _package_dir(

            lesson,
            variant,
            lecture_text,

        )

    )


# ============================================================
# 외부용: 전체 음성 강의 생성
# ============================================================

def generate_lesson_package(
    lesson: dict,
    variant: str,
    lecture_text: str,
):

    """
    1. Markdown 문단 분리
    2. 문단별 TTS 생성
    3. 실제 WAV 시간 측정
    4. WAV 병합
    5. timeline.json 생성
    """

    package_dir = (
        _package_dir(
            lesson,
            variant,
            lecture_text,
        )
    )


    # 기존 강의가 있으면 재사용
    cached = (
        _load_package(
            package_dir
        )
    )

    if cached:

        return cached


    segments = (
        build_segments(
            lecture_text
        )
    )


    if not segments:

        raise ValueError(
            "음성으로 읽을 강의 내용이 없습니다."
        )


    package_dir.mkdir(

        parents=True,

        exist_ok=True,

    )


    # 각 문단별 TTS 생성
    segment_paths = [

        _ensure_segment_wav(

            segment[
                "speech"
            ],

            VOICE_NAME,

            TTS_STYLE,

        )

        for segment
        in segments

    ]


    # 최종 WAV
    audio_path = (

        package_dir
        / "lecture.wav"

    )


    # WAV 합치면서 실제 구간 시간 자동 계산
    timeline = (
        _combine_wavs(

            segment_paths,

            segments,

            audio_path,

        )
    )


    # timeline.json
    timeline_path = (

        package_dir
        / "timeline.json"

    )


    timeline_payload = json.dumps(

        timeline,

        ensure_ascii=False,

        indent=2,

    )


    with NamedTemporaryFile(

        dir=package_dir,

        suffix=".json",

        mode="w",

        encoding="utf-8",

        delete=False,

    ) as temp:

        temp_path = Path(
            temp.name
        )

        temp.write(
            timeline_payload
        )


    temp_path.replace(
        timeline_path
    )


    return {

        "audio_path":
            audio_path,

        "timeline_path":
            timeline_path,

        "timeline":
            timeline,

    }