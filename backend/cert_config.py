import json
from pathlib import Path
CERT_CONFIG = json.loads((Path(__file__).parent / "certifications.json").read_text(encoding="utf-8"))
CERT_MAP = {v["label"]: k for k,v in CERT_CONFIG.items()}
CERT_LABEL_MAP = {k:v["label"] for k,v in CERT_CONFIG.items()}
TOPIC_KOR_MAP = {k:v for c in CERT_CONFIG.values() for k,v in c["topics"].items()}
CERT_TOPICS = {k:list(v["topics"]) for k,v in CERT_CONFIG.items()}
