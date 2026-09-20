"""Run both Streamlit screens. Ctrl+C stops both child processes."""
import os
from pathlib import Path
import subprocess
import sys
import time
from dotenv import load_dotenv

root = Path(__file__).resolve().parent
load_dotenv(root / '.env')
children = []
try:
    for page, port in [('learning_app.py', '8511'), ('app.py', '8512')]:
        children.append(subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', str(root / 'frontend' / page), '--server.port', port, '--server.address', '127.0.0.1', '--server.headless', 'true'], cwd=root))
    print('학습실: http://localhost:8511 | 문제풀이: http://localhost:8512', flush=True)
    while all(p.poll() is None for p in children):
        time.sleep(1)
finally:
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        child.wait()
