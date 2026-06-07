# 자격증/과목 공통 설정 파일

CERT_CONFIG = {
    "EIP": {
        "label": "정보처리기사",
        "topics": {
            "software_design": "소프트웨어 설계",
            "software_development": "소프트웨어 개발",
            "database": "데이터베이스 구축",
            "programming_language": "프로그래밍 언어 활용",
            "info_system": "정보시스템 구축 관리",
        },
    },
    "LREA_1": {
        "label": "공인중개사 1차",
        "topics": {
            "civil_law": "민법 및 민사특별법",
            "housing_lease": "주택임대차보호법",
            "commercial_lease": "상가건물 임대차보호법",
            "aggregate_building": "집합건물법",
            "provisional_registration": "가등기담보법",
            "real_name_registration": "부동산실명법",
        },
    },
}



# 파생 딕셔너리


# 화면 표시용: "정보처리기사" -> "EIP"
CERT_MAP = {
    config["label"]: cert_code
    for cert_code, config in CERT_CONFIG.items()
}

# 화면 표시용 역매핑: "EIP" -> "정보처리기사"
CERT_LABEL_MAP = {
    cert_code: config["label"]
    for cert_code, config in CERT_CONFIG.items()
}

# 전체 과목 한글 매핑: "software_design" -> "소프트웨어 설계"
TOPIC_KOR_MAP = {
    topic_eng: topic_kor
    for config in CERT_CONFIG.values()
    for topic_eng, topic_kor in config["topics"].items()
}

# 출제 엔진에서 쓰기 편한 형태: "EIP" -> ["software_design", ...]
CERT_TOPICS = {
    cert_code: list(config["topics"].keys())
    for cert_code, config in CERT_CONFIG.items()
}
