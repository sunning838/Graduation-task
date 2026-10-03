"""Offline assembly timings with synthetic questions in a disposable database."""
import statistics
import tempfile
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from backend.api_store import APIStore
from backend.mock_exams import add_question, create_exam
from backend.cert_config import CERT_CONFIG


def main():
    with tempfile.TemporaryDirectory() as folder:
        store = APIStore(Path(folder)/'benchmark.db')
        topics = list(CERT_CONFIG['EIP']['topics'])
        for topic in topics:
            for _ in range(30):
                quiz = dict(topic=topic,question=str(uuid4()),options=['a','b','c','d'],answer=1,explanation='Synthetic benchmark fixture')
                add_question(store,'EIP',quiz,{'kind':'synthetic'},'ready',{'kind':'test_only'})
        for count in (20,50,100):
            timings=[]
            for _ in range(10):
                start=perf_counter()
                exam=create_exam(store,'EIP',[dict(topic=t,count=count//len(topics)) for t in topics],str(uuid4()))
                timings.append((perf_counter()-start)*1000)
                assert len(exam['items'])==count
            print(f'{count} questions: median={statistics.median(timings):.2f}ms, max={max(timings):.2f}ms (10 local DB assemblies, no AI)')


if __name__=='__main__': main()
