"""Launch the local API and React dev server. Ctrl+C stops this launch's children."""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
import webbrowser

ROOT = Path(__file__).resolve().parent


def commands():
    node = shutil.which('node')
    vite = ROOT / 'react_frontend/node_modules/vite/bin/vite.js'
    if not node:
        raise RuntimeError('Node.js가 없습니다. Node.js 설치 후 다시 실행하세요.')
    if not vite.is_file():
        raise RuntimeError('React 의존성이 없습니다. react_frontend 폴더에서 npm install을 실행하세요.')
    if importlib.util.find_spec('uvicorn') is None:
        raise RuntimeError('Python 의존성이 없습니다. venv에서 pip install -r requirements.txt를 실행하세요.')
    return [([sys.executable, '-m', 'uvicorn', 'api:app', '--reload', '--host', '127.0.0.1', '--port', '8000'], ROOT),
            ([node, str(vite), '--host', '127.0.0.1', '--port', '5173', '--strictPort'], ROOT / 'react_frontend')]


def check_ports():
    for port in (8000, 5173):
        with socket.socket() as sock:
            try:
                sock.bind(('127.0.0.1', port))
            except OSError as exc:
                raise RuntimeError(f'{port}번 포트가 사용 중입니다. 기존 서버를 종료한 뒤 다시 실행하세요.') from exc


def stop(children):
    for child in children:
        if child.poll() is None:
            try:
                if os.name == 'nt':
                    child.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(child.pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
    for child in children:
        try:
            child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
            else:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check dependencies only; do not start servers or open a browser.')
    args = parser.parse_args()
    children = []
    try:
        entries = commands()
        if args.check:
            print('실행 환경 확인 완료: Python/uvicorn, Node.js, Vite')
            return 0
        check_ports()
        options = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
        for command, cwd in entries:
            children.append(subprocess.Popen(command, cwd=cwd, **options))
        print('\n서버 준비 중입니다. 종료할 때는 이 창에서 Ctrl+C를 누르세요.\n', flush=True)
        opener = build_opener(ProxyHandler({}))
        deadline = time.monotonic() + 90
        pending = {'http://127.0.0.1:8000/openapi.json', 'http://127.0.0.1:5173/'}
        while pending:
            if any(child.poll() is not None for child in children):
                raise RuntimeError('서버 실행에 실패했습니다. 위 로그를 확인하세요.')
            if time.monotonic() >= deadline:
                raise RuntimeError('서버 준비 시간이 초과되었습니다. 위 로그를 확인하세요.')
            for url in list(pending):
                try:
                    with opener.open(url, timeout=1) as response:
                        if response.status == 200:
                            pending.remove(url)
                except (URLError, OSError):
                    pass
            time.sleep(.4)
        url = 'http://127.0.0.1:5173/'
        print(f'실행 완료: {url}\n종료: Ctrl+C', flush=True)
        if not webbrowser.open(url):
            print('브라우저에서 위 주소를 직접 열어 주세요.', flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(.5)
        raise RuntimeError('서버가 종료되었습니다. 위 로그를 확인하세요.')
    except KeyboardInterrupt:
        print('\n서버를 종료합니다.', flush=True)
        return 0
    except Exception as exc:
        print(f'실행 오류: {exc}', file=sys.stderr, flush=True)
        return 1
    finally:
        stop(children)


if __name__ == '__main__':
    raise SystemExit(main())
