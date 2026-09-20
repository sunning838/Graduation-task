"""Bounded, validated JSON responses for learning-room AI calls."""
import json
import logging
import re

LOG = logging.getLogger(__name__)


def validate_lesson(data):
    if type(data.get('supported')) is not bool:
        raise ValueError('supported must be a boolean')
    if data['supported'] and (not isinstance(data.get('text'), str) or not data['text'].strip()):
        raise ValueError('Missing lecture text')


def request_json(llm, message, validate=None):
    for attempt in range(2):
        instruction = message
        if attempt:
            instruction += '\n반드시 유효한 JSON 객체 하나만 반환하세요. text 안의 줄바꿈과 큰따옴표는 JSON 규칙대로 이스케이프하세요. 설명 길이를 줄이더라도 완결된 JSON을 반환하세요.'
        response = llm.invoke(instruction, response_mime_type='application/json')
        content = response.content
        if isinstance(content, list):
            content = ''.join(block.get('text', '') for block in content
                              if isinstance(block, dict) and block.get('type') in ('text', None)
                              and isinstance(block.get('text'), str) and not block.get('thought'))
        metadata = getattr(response, 'response_metadata', {}) or {}
        finish = str(metadata.get('finish_reason', 'unknown'))
        try:
            if any(reason in finish.upper() for reason in ('MAX_TOKENS', 'SAFETY', 'RECITATION', 'BLOCKLIST', 'PROHIBITED')):
                raise ValueError('Incomplete or blocked response')
            if not isinstance(content, str) or not content.strip():
                raise ValueError('Empty JSON response')
            parsed = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', content.strip()))
            if not isinstance(parsed, dict):
                raise ValueError('Expected a JSON object')
            if validate:
                validate(parsed)
            return parsed
        except (ValueError, TypeError) as error:
            LOG.warning('Learning JSON response rejected: attempt=%s finish=%s length=%s error=%s',
                        attempt + 1, finish, len(content) if isinstance(content, str) else 0, type(error).__name__)
            if attempt or any(reason in finish.upper() for reason in ('SAFETY', 'RECITATION', 'BLOCKLIST', 'PROHIBITED')):
                raise ValueError('Could not obtain a complete structured lesson response') from error

