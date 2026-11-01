"""Offline illustration lifecycle and API contracts; never calls paid APIs."""
import base64
import copy
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import api
from backend import presentation_routes as routes
from backend.lecture_presentation import GeminiIllustrator, PresentationError, PresentationManager, build_slides

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jX1sAAAAASUVORK5CYII=')
LESSON = dict(id='test-lesson', version='v1', cert='EIP', subject='database', title='스택',
              body='스택은 마지막에 들어온 데이터가 먼저 나오는 자료 구조입니다.', source='private-source.md')
TEXT = '### 스택\n\n스택은 마지막에 들어온 데이터가 먼저 나옵니다.\n\n- 삽입을 PUSH라고 합니다.\n- 삭제를 POP이라고 합니다.\n\n### 핵심 정리\n\n스택의 입출력 위치는 같습니다.'


class FakeProvider:
    image_model, text_model = 'offline-image', 'offline-review'

    def __init__(self, gate=None, reject=0):
        self.generated = self.planned = self.reviewed = 0
        self.gate, self.reject = gate, reject
        self.order, self.generated_ids = [], []

    def plan(self, slides, segments, docs, limit):
        self.planned += 1
        self.order.append(('plan', slides[0]['id']))
        if self.gate:
            self.gate.wait(10)
        return [dict(slide_id=slides[0]['id'], slide_title=slides[0]['title'], bullets=['스택의 핵심 개념입니다.'],
                     prompt='PRIVATE PROMPT', caption='스택의 데이터 처리 순서',
                     source_index=1, quote='PRIVATE EVIDENCE', required_elements=['후입선출'])]

    def generate(self, scene, repair=''):
        self.generated += 1
        self.generated_ids.append(scene['slide_id'])
        self.order.append(('generate', scene['slide_id']))
        return PNG

    def review(self, image, scene, slide, docs):
        self.reviewed += 1
        return dict(valid=self.reviewed > self.reject, reason='방향 확인')


class PresentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.managers = []

    def tearDown(self):
        for manager in self.managers:
            manager.executor.shutdown(wait=True)
        self.temp.cleanup()

    def manager(self, provider=None, root=None):
        manager = PresentationManager(root or Path(self.temp.name), provider or FakeProvider(), workers=1)
        self.managers.append(manager)
        return manager

    def finish(self, manager, result):
        manager.futures[result['package_id']].result(10)
        return manager.get(result['package_id'])

    def test_segments_match_existing_audio_and_all_text_is_retained(self):
        from backend.tts_engine import build_segments
        slides, segments = build_slides(TEXT, '스택')
        self.assertEqual(segments, [dict(index=s['index'], markdown=s['markdown']) for s in build_segments(TEXT)])
        self.assertEqual([i for slide in slides for i in slide['segment_indices']], list(range(len(segments))))
        self.assertEqual(len(slides), 2)
        self.assertIn('PUSH', ''.join(s['markdown'] for s in segments))

    def test_returns_before_generation_and_deduplicates_strict_mode_requests(self):
        gate = threading.Event()
        provider = FakeProvider(gate)
        manager = self.manager(provider)
        try:
            result = manager.start(LESSON, [LESSON], 'basic', TEXT)
            self.assertEqual(provider.generated, 0)
            duplicate = manager.start(LESSON, [LESSON], 'basic', TEXT)
            self.assertEqual(result['package_id'], duplicate['package_id'])
        finally:
            gate.set()
        done = self.finish(manager, result)
        self.assertEqual(done['status'], 'ready')
        self.assertEqual(done['images'], dict(total=2, ready=2, failed=0))
        self.assertEqual(provider.generated, 2)
        public = str(done)
        for private in ('PRIVATE', 'private-source', 'source_index', 'required_elements', '_lesson'):
            self.assertNotIn(private, public)
        self.assertEqual(manager.image_path(result['package_id'], 'slide-0').read_bytes(), PNG)

    def test_rejected_image_is_repaired_once_and_never_published_without_review(self):
        provider = FakeProvider(reject=20)
        manager = self.manager(provider)
        done = self.finish(manager, manager.start(LESSON, [LESSON], 'basic', TEXT))
        self.assertEqual(provider.generated, 4)
        self.assertEqual(done['status'], 'failed')
        self.assertEqual(done['images']['ready'], 0)
        self.assertEqual(done['images']['failed'], 2)
        self.assertTrue(done['segments'])
        with self.assertRaises(PresentationError):
            manager.image_path(done['package_id'], 'slide-0')

    def test_one_repair_can_succeed(self):
        provider = FakeProvider(reject=1)
        manager = self.manager(provider)
        done = self.finish(manager, manager.start(LESSON, [LESSON], 'basic', TEXT))
        self.assertEqual(done['status'], 'ready')
        self.assertEqual(provider.generated, 3)

    def test_exact_text_variant_and_source_version_invalidate_cache(self):
        manager = self.manager()
        result = manager.start(LESSON, [LESSON], 'basic', TEXT)
        self.finish(manager, result)
        changed_text = manager.start(LESSON, [LESSON], 'basic', TEXT + '\n\n추가 설명')
        changed_variant = manager.start(LESSON, [LESSON], 'simple', TEXT)
        changed_source = manager.start({**LESSON, 'version': 'v2'}, [{**LESSON, 'version': 'v2'}], 'basic', TEXT)
        self.assertEqual(len({r['package_id'] for r in [result, changed_text, changed_variant, changed_source]}), 4)

    def test_verified_cache_survives_restart_without_api_calls(self):
        manager = self.manager()
        done = self.finish(manager, manager.start(LESSON, [LESSON], 'basic', TEXT))
        provider = FakeProvider()
        restarted = self.manager(provider)
        cached = restarted.start(LESSON, [LESSON], 'basic', TEXT)
        self.assertEqual(cached['status'], 'ready')
        self.assertEqual(done['package_id'], cached['package_id'])
        self.assertEqual(provider.generated, 0)

    def test_retry_keeps_package_and_rejects_changed_source(self):
        manager = self.manager(FakeProvider(reject=20))
        done = self.finish(manager, manager.start(LESSON, [LESSON], 'basic', TEXT))
        manager.provider.reject = 0
        retried = manager.retry(done['package_id'], [LESSON])
        self.assertEqual(retried['package_id'], done['package_id'])
        self.assertEqual(self.finish(manager, retried)['status'], 'ready')
        with self.assertRaises(PresentationError):
            manager.retry(done['package_id'], [{**LESSON, 'version': 'v2'}])

    def test_routes_validate_ids_and_never_expose_private_data(self):
        manager = self.manager()
        with patch.object(routes, 'manager', return_value=manager), patch.object(routes, '_lesson', return_value=LESSON), patch.object(routes, '_catalog', return_value=({}, [LESSON])):
            with TestClient(api.app) as client:
                response = client.post('/api/learning/presentations', json=dict(lesson_id=LESSON['id'], text=TEXT))
                self.assertEqual(response.status_code, 200)
                done = self.finish(manager, response.json())
                self.assertEqual(client.get('/api/learning/presentations/' + done['package_id']).status_code, 200)
                self.assertEqual(client.get('/api/learning/presentations/not-a-package').status_code, 404)
                self.assertEqual(client.get(f"/api/learning/presentations/{done['package_id']}/images/slide-0").headers['content-type'], 'image/png')
                self.assertEqual(client.post('/api/learning/presentations', json=dict(lesson_id='x', variant='wrong', text=TEXT)).status_code, 422)

    def test_planner_rejects_missing_or_invented_source_quotes(self):
        provider = GeminiIllustrator()
        slides, segments = build_slides(TEXT, '스택')
        slides = slides[:1]
        scene = dict(slide_id='slide-0', slide_title='스택', bullets=['후입선출입니다.'], prompt='Draw stack', caption='스택 구조', source_index=1,
                     quote='존재하지 않는 잘못된 원문 인용입니다.', required_elements=['후입선출'])
        with patch.object(provider, '_json', return_value={'scenes': [scene]}):
            with self.assertRaises(ValueError): provider.plan(slides, segments, [LESSON], 3)
        scene['quote'] = LESSON['body']
        with patch.object(provider, '_json', return_value={'scenes': [scene]}):
            self.assertEqual(len(provider.plan(slides, segments, [LESSON], 3)), 1)
        duplicate = copy.deepcopy(scene)
        with patch.object(provider, '_json', return_value={'scenes': [scene, duplicate]}):
            with self.assertRaises(ValueError): provider.plan(slides, segments, [LESSON], 3)

    def test_evidence_handles_copy_original_text_on_server(self):
        provider = GeminiIllustrator()
        slides, segments = build_slides(TEXT, '스택')
        slides = slides[:1]
        scene = dict(slide_id='slide-0', slide_title='스택', bullets=['후입선출입니다.'], prompt='Draw a stack', caption='스택', evidence_id='1:0',
                     quote='model must not supply the original', required_elements=['후입선출'])
        with patch.object(provider, '_json', return_value={'scenes': [scene]}):
            result = provider.plan(slides, segments, [LESSON], 3)
            self.assertEqual(result[0]['quote'], LESSON['body'])
        scene['evidence_id'] = '1:9999'
        with patch.object(provider, '_json', return_value={'scenes': [scene]}):
            with self.assertRaises(ValueError): provider.plan(slides, segments, [LESSON], 3)

    def test_related_source_changes_invalidate_package_and_retry(self):
        manager = self.manager()
        other = {**LESSON, 'id': 'related', 'version': 'r1'}
        result = manager.start(LESSON, [LESSON, other], 'basic', TEXT)
        self.finish(manager, result)
        changed = {**other, 'version': 'r2'}
        new = manager.start(LESSON, [LESSON, changed], 'basic', TEXT)
        self.assertNotEqual(new['package_id'], result['package_id'])
        with self.assertRaises(PresentationError): manager.retry(result['package_id'], [LESSON, changed])

    def test_every_slide_gets_an_image_including_more_than_three(self):
        text = '\n\n'.join(f'### 개념 {i}\n\n스택의 동작을 설명하는 문단 {i}입니다.' for i in range(5))
        provider = FakeProvider()
        manager = self.manager(provider)
        queued = manager.start(LESSON, [LESSON], 'basic', text)
        self.assertEqual(queued['images']['total'], len(queued['slides']))
        done = self.finish(manager, queued)
        self.assertEqual(done['images'], dict(total=5, ready=5, failed=0))
        self.assertEqual(provider.generated_ids, [f'slide-{i}' for i in range(5)])
        self.assertTrue(all(slide['image_url'] for slide in done['slides']))
        self.assertEqual(provider.order[:4], [('plan', 'slide-0'), ('generate', 'slide-0'), ('plan', 'slide-1'), ('generate', 'slide-1')])

    def test_first_image_is_available_before_later_scene_planning_finishes(self):
        reached, release = threading.Event(), threading.Event()

        class SlowSecondScene(FakeProvider):
            def plan(self, slides, segments, docs, limit):
                if slides[0]['id'] == 'slide-1':
                    reached.set()
                    release.wait(8)
                return super().plan(slides, segments, docs, limit)

        manager = self.manager(SlowSecondScene())
        result = manager.start(LESSON, [LESSON], 'basic', TEXT)
        try:
            self.assertTrue(reached.wait(5))
            halfway = manager.get(result['package_id'])
            self.assertEqual(halfway['images'], dict(total=2, ready=1, failed=0))
            self.assertEqual(halfway['slides'][0]['image_status'], 'ready')
            self.assertEqual(halfway['slides'][1]['image_status'], 'planning')
        finally:
            release.set()
        self.assertEqual(self.finish(manager, result)['status'], 'ready')

    def test_retry_preserves_completed_images_and_only_retries_failed_slide(self):
        class SelectiveFailure(FakeProvider):
            fail = True

            def review(self, image, scene, slide, docs):
                return dict(valid=not (self.fail and slide['id'] == 'slide-1'), reason='검수')

        provider = SelectiveFailure()
        manager = self.manager(provider)
        done = self.finish(manager, manager.start(LESSON, [LESSON], 'basic', TEXT))
        self.assertEqual(done['status'], 'partial')
        self.assertEqual(done['images'], dict(total=2, ready=1, failed=1))
        provider.fail = False
        ready = self.finish(manager, manager.retry(done['package_id'], [LESSON]))
        self.assertEqual(ready['images'], dict(total=2, ready=2, failed=0))
        self.assertEqual(provider.generated_ids.count('slide-0'), 1)
        self.assertEqual(provider.planned, 2)

    def test_planner_requires_exact_coverage_and_joint_slide_summary(self):
        provider = GeminiIllustrator()
        slides, segments = build_slides(TEXT, '스택')
        first = dict(slide_id='slide-0', slide_title='스택', bullets=['후입선출입니다.'], prompt='Draw a stack',
                     caption='스택 구조', evidence_id='1:0', required_elements=['후입선출'])
        with patch.object(provider, '_json', return_value={'scenes': [first]}):
            with self.assertRaises(ValueError): provider.plan(slides, segments, [LESSON], len(slides))
        second = {**first, 'slide_id': 'slide-1'}
        with patch.object(provider, '_json', return_value={'scenes': [second, first]}):
            planned = provider.plan(slides, segments, [LESSON], len(slides))
            self.assertEqual([s['slide_id'] for s in planned], ['slide-0', 'slide-1'])
        second['bullets'] = []
        with patch.object(provider, '_json', return_value={'scenes': [first, second]}):
            with self.assertRaises(ValueError): provider.plan(slides, segments, [LESSON], len(slides))

    def test_slide_limit_keeps_all_narration_segments(self):
        text = '\n\n'.join(f'### 개념 {i}\n\n원문 설명 {i}입니다.' for i in range(18))
        slides, segments = build_slides(text, '스택')
        self.assertEqual(len(slides), 16)
        self.assertEqual([i for slide in slides for i in slide['segment_indices']], list(range(len(segments))))
        self.assertIn('원문 설명 17', segments[-1]['markdown'])


if __name__ == '__main__':
    unittest.main()
