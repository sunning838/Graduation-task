from __future__ import annotations

import base64
import json

import streamlit.components.v1 as components

from markdown_it import MarkdownIt


# Markdown → HTML
_MARKDOWN = MarkdownIt(
    "commonmark",
    {
        "html": False,
        "linkify": False,
    },
)


# ============================================================
# Markdown 하나를 HTML로 변환
# ============================================================

def _segment_html(
    markdown: str,
) -> str:

    return _MARKDOWN.render(
        markdown or ""
    )


# ============================================================
# 동기화 강의 플레이어
# ============================================================

def render_synced_lecture(
    package: dict,
    height: int = 650,
):

    """
    음성 플레이어 +
    현재 문단 강조 +
    자동 스크롤
    """

    # --------------------------------------------------------
    # WAV → Base64
    # --------------------------------------------------------

    audio_bytes = (
        package[
            "audio_path"
        ].read_bytes()
    )

    audio_b64 = (
        base64.b64encode(
            audio_bytes
        ).decode(
            "ascii"
        )
    )


    # --------------------------------------------------------
    # Timeline 준비
    # --------------------------------------------------------

    timeline = []


    for item in (
        package[
            "timeline"
        ]
    ):

        timeline.append(
            {
                "index":
                    int(
                        item[
                            "index"
                        ]
                    ),

                "start":
                    float(
                        item[
                            "start"
                        ]
                    ),

                "end":
                    float(
                        item[
                            "end"
                        ]
                    ),

                "html":
                    _segment_html(
                        item[
                            "markdown"
                        ]
                    ),
            }
        )


    timeline_json = json.dumps(

        timeline,

        ensure_ascii=False,

    ).replace(
        "</",
        "<\\/",
    )


    # --------------------------------------------------------
    # 각 문단 HTML
    # --------------------------------------------------------

    blocks = "\n".join(

        (
            '<section '
            'class="lecture-segment" '
            f'data-index="{item["index"]}">'
            f'{item["html"]}'
            '</section>'
        )

        for item
        in timeline

    )


    # ========================================================
    # HTML + CSS + Javascript
    # ========================================================

    page = f"""
<!doctype html>

<html lang="ko">

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>

<style>

:root {{
    color-scheme: light dark;
}}

* {{
    box-sizing: border-box;
}}

html,
body {{
    margin: 0;
    padding: 0;
    height: 100%;

    font-family:
        Arial,
        "Noto Sans KR",
        sans-serif;
}}

body {{
    background: transparent;
    color: #1f2937;
}}


/* =========================================================
   전체 플레이어
   ========================================================= */

.player-shell {{

    height: {height - 10}px;

    display: flex;

    flex-direction: column;

    border:
        1px solid
        rgba(128,128,128,.28);

    border-radius: 14px;

    overflow: hidden;

    background: #ffffff;

}}


/* =========================================================
   상단 음성 플레이어
   ========================================================= */

.player-top {{

    flex: 0 0 auto;

    padding:
        12px
        14px
        10px;

    border-bottom:
        1px solid
        rgba(128,128,128,.20);

    background: #ffffff;

    position: sticky;

    top: 0;

    z-index: 3;

}}


.player-row {{

    display: flex;

    align-items: center;

    justify-content: space-between;

    gap: 12px;

    margin-top: 7px;

    flex-wrap: wrap;

}}


audio {{

    width: 100%;

    height: 40px;

}}


.follow-label {{

    display: inline-flex;

    align-items: center;

    gap: 7px;

    font-size: 13px;

    white-space: nowrap;

    user-select: none;

}}


.follow-label input {{

    width: 16px;

    height: 16px;

}}

.speed-control {{

    display: inline-flex;

    align-items: center;

    gap: 7px;

    font-size: 13px;

    white-space: nowrap;

}}


.speed-control select {{

    min-width: 82px;

    padding:
        5px
        8px;

    border:
        1px solid
        rgba(128,128,128,.35);

    border-radius: 7px;

    background: transparent;

    color: inherit;

    font-size: 13px;

    cursor: pointer;

}}


/* Chrome 기본 더보기 메뉴는 숨김 */
audio::-webkit-media-controls-overflow-button {{

    display: none;

}}

.status {{

    font-size: 12px;

    opacity: .72;

    margin-top: 4px;

}}


/* =========================================================
   강의 내용 스크롤 영역
   ========================================================= */

.lecture-scroll {{

    flex: 1 1 auto;

    overflow-y: auto;

    padding:

        10px
        14px
        32px;

    scroll-behavior: smooth;

}}


/* =========================================================
   각 강의 문단
   ========================================================= */

.lecture-segment {{

    padding:

        11px
        13px;

    margin:

        3px
        0
        8px;

    border-left:

        4px solid
        transparent;

    border-radius: 10px;

    transition:

        background .22s ease,

        border-color .22s ease,

        box-shadow .22s ease;

    cursor: pointer;

}}


.lecture-segment:hover {{

    background:

        rgba(
            79,
            131,
            204,
            .06
        );

}}


/* =========================================================
   현재 읽고 있는 문단
   ========================================================= */

.lecture-segment.active {{

    background:

        rgba(
            255,
            193,
            7,
            .15
        );

    border-left-color:

        #f3a712;

    box-shadow:

        0
        1px
        6px
        rgba(
            0,
            0,
            0,
            .06
        );

}}


.lecture-segment h1,
.lecture-segment h2,
.lecture-segment h3 {{

    margin:

        .25rem
        0
        .75rem;

}}


.lecture-segment h3 {{

    font-size:

        1.28rem;

    padding-bottom:

        .45rem;

    border-bottom:

        1px solid
        rgba(
            128,
            128,
            128,
            .25
        );

}}


.lecture-segment p,
.lecture-segment li {{

    font-size:

        1.02rem;

    line-height:

        1.85;

    word-break:

        keep-all;

    overflow-wrap:

        anywhere;

}}


.lecture-segment p {{

    margin:

        .45rem
        0
        .8rem;

}}


.lecture-segment ul,
.lecture-segment ol {{

    padding-left:

        1.4rem;

}}


.lecture-segment blockquote {{

    margin:

        .8rem
        0;

    padding:

        .7rem
        .9rem;

    border-left:

        3px solid
        #4f83cc;

    background:

        rgba(
            79,
            131,
            204,
            .07
        );

}}


.lecture-segment pre {{

    overflow-x: auto;

    padding:

        .8rem;

    border-radius:

        8px;

    background:

        rgba(
            128,
            128,
            128,
            .10
        );

}}


/* =========================================================
   Dark Mode
   ========================================================= */

@media (
    prefers-color-scheme: dark
) {{

    body {{
        color: #e5e7eb;
    }}

    .player-shell,
    .player-top {{

        background:
            #0e1117;

    }}

    .lecture-segment.active {{

        background:

            rgba(
                243,
                167,
                18,
                .14
            );

        box-shadow:
            none;

    }}

}}

</style>

</head>


<body>


<div class="player-shell">


    <!-- ===================================================
         오디오 플레이어
         =================================================== -->

    <div class="player-top">


        <audio
            id="lectureAudio"
            controls
            preload="metadata"
        >

            <source
                src="data:audio/wav;base64,{audio_b64}"
                type="audio/wav"
            >

        </audio>


        <div class="player-row">

    <label class="follow-label">

        <input
            id="followToggle"
            type="checkbox"
            checked
        >

        현재 읽는 부분 자동 따라가기

    </label>


    <label class="speed-control">

        재생 속도

        <select id="speedSelect">

            <option value="0.75">
                0.75x
            </option>

            <option value="1" selected>
                1.0x
            </option>

            <option value="1.25">
                1.25x
            </option>

            <option value="1.5">
                1.5x
            </option>

            <option value="2">
                2.0x
            </option>

        </select>

    </label>

</div>


        <div
            id="status"
            class="status"
        >

            재생하면 현재 읽는 문단이 강조됩니다.

        </div>


    </div>


    <!-- ===================================================
         강의 내용
         =================================================== -->

    <div
        id="lectureScroll"
        class="lecture-scroll"
    >

        {blocks}

    </div>


</div>


<script>


// ============================================================
// Python에서 생성한 실제 음성 타임라인
// ============================================================

const timeline =
    {timeline_json};


// 음성 플레이어
const audio =
    document.getElementById(
        "lectureAudio"
    );


// 강의 스크롤 영역
const scrollBox =
    document.getElementById(
        "lectureScroll"
    );


// 자동 스크롤 ON/OFF
const followToggle =
    document.getElementById(
        "followToggle"
    );

const speedSelect =
    document.getElementById(
        "speedSelect"
    );

audio.playbackRate = 1.0;
audio.defaultPlaybackRate = 1.0;

// 상태 표시
const status =
    document.getElementById(
        "status"
    );


// 문단들
const elements =
    Array.from(
        document.querySelectorAll(
            ".lecture-segment"
        )
    );


let activeIndex = -1;


// ============================================================
// 현재 시간 → 해당 문단 찾기
// ============================================================

function findActive(
    time
) {{

    if (!timeline.length) {{

        return -1;

    }}


    // 뒤에서부터 확인
    // 문단 사이 짧은 무음에서는
    // 직전 문단이 유지된다.

    for (
        let i =
            timeline.length - 1;

        i >= 0;

        i--
    ) {{

        if (
            time
            >=
            timeline[i].start
        ) {{

            return i;

        }}

    }}


    return 0;

}}


// ============================================================
// 현재 문단을 화면 중앙으로 이동
// ============================================================

function centerInScrollBox(
    element
) {{

    const target =

        element.offsetTop

        -

        (
            scrollBox.clientHeight
            / 2
        )

        +

        (
            element.offsetHeight
            / 2
        );


    scrollBox.scrollTo({{

        top:
            Math.max(
                0,
                target
            ),

        behavior:
            "smooth",

    }});

}}


// ============================================================
// 현재 문단 강조
// ============================================================

function setActive(
    index
) {{

    if (

        index < 0

        ||

        index >=
        elements.length

        ||

        index ===
        activeIndex

    ) {{

        return;

    }}


    // 이전 강조 제거
    if (
        activeIndex >= 0
    ) {{

        elements[
            activeIndex
        ].classList.remove(
            "active"
        );

    }}


    activeIndex =
        index;


    const element =
        elements[
            index
        ];


    // 현재 문단 강조
    element.classList.add(
        "active"
    );


    status.textContent =

        `현재 강의 위치: ${{
            index + 1
        }} / ${{
            elements.length
        }}`;


    // 자동 따라가기
    if (
        followToggle.checked
    ) {{

        centerInScrollBox(
            element
        );

    }}

}}


// ============================================================
// 음성 현재 시간과 화면 동기화
// ============================================================

speedSelect.addEventListener(

    "change",

    () => {{

        const rate =
            Number(
                speedSelect.value
            );

        audio.playbackRate =
            rate;

        audio.defaultPlaybackRate =
            rate;

    }}

);

function sync() {{

    setActive(

        findActive(

            audio.currentTime
            || 0

        )

    );

}}


// ============================================================
// 음성 이벤트
// ============================================================



audio.addEventListener(
    "timeupdate",
    sync
);


audio.addEventListener(
    "seeking",
    sync
);


audio.addEventListener(

    "play",

    () => {{

        audio.playbackRate =
            Number(
                speedSelect.value
            );

        sync();

    }}

);


audio.addEventListener(
    "loadedmetadata",
    sync
);


audio.addEventListener(

    "ended",

    () => {{

        if (
            elements.length
        ) {{

            setActive(
                elements.length - 1
            );

        }}


        status.textContent =
            "강의 재생이 끝났습니다.";

    }}

);


// ============================================================
// 문단을 직접 클릭하면 해당 음성 위치로 이동
// ============================================================

elements.forEach(

    (
        element,
        index
    ) => {{

        element.addEventListener(

            "click",

            () => {{

                audio.currentTime =
                    timeline[
                        index
                    ].start;


                setActive(
                    index
                );


                audio.play()
                    .catch(
                        () => {{}}
                    );

            }}

        );

    }}

);


</script>


</body>

</html>
"""


    components.html(

        page,

        height=height,

        scrolling=False,

    )