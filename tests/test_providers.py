import json
from dataclasses import replace
from unittest.mock import Mock

import pytest
import requests
from pydantic import ValidationError

from coursework.llm_provider import ProviderConfig, GeminiProvider, ProviderError, ProcessingPaused, ENDPOINT
from coursework.normalize import SemanticEnrichment, normalize
from coursework.scraper import scrape
from coursework.storage import read_json
from coursework.pipeline import ROOT, execute, process_run

VALID = {'topic': 'Наука', 'keywords': ['исследование'], 'summary_ru': 'Описание исследования.', 'insufficient_data': False}


def config(**kwargs):
    return replace(ProviderConfig(api_key='test-only-credential', free_tier_confirmed=True, delay=1), **kwargs)


def response(value=VALID, status=200, finish='stop'):
    return Mock(status_code=status, headers={}, json=Mock(return_value={
        'model': 'gemini-3.1-flash-lite', 'choices': [{'finish_reason': finish, 'message': {'content': json.dumps(value, ensure_ascii=False)}}],
        'usage': {'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30}}))


def provider(tmp_path, session=None, cfg=None, run_id='test'):
    session = session or Mock(post=Mock(return_value=response()))
    return GeminiProvider(cfg or config(), tmp_path / 'cache.sqlite', session=session, sleep=lambda _: None, run_id=run_id)


def test_gemini_payload_endpoint_usage_and_cache(tmp_path):
    session = Mock(post=Mock(return_value=response()))
    client = provider(tmp_path, session)
    raw = {'title': 'Book', 'category_raw': 'Science', 'description_raw': 'Text', 'price_raw': '12'}
    schema = SemanticEnrichment.model_json_schema()
    assert SemanticEnrichment.model_validate(client.enrich(raw, schema)).topic == 'Наука'
    assert client.last_response['usage']['total_tokens'] == 30
    args, kwargs = session.post.call_args
    assert args == (ENDPOINT,)
    assert kwargs['allow_redirects'] is False
    assert kwargs['headers']['Authorization'] == 'Bearer test-only-credential'
    assert kwargs['json']['response_format']['type'] == 'json_schema'
    assert kwargs['timeout'] == (10, 60)
    assert 'price_raw' not in kwargs['json']['messages'][1]['content']
    assert session.trust_env is False
    assert 'test-only-credential' not in repr(client.config)
    assert 'test-only-credential' not in json.dumps(client.config.public())
    other = provider(tmp_path, session, run_id='another-run')
    assert other.enrich(raw, schema) == VALID
    assert other.last_response['cached'] and other.last_response['http_requests'] == 0
    assert session.post.call_count == 1


@pytest.mark.parametrize('bad', [{**VALID, 'topic': 'Wrong'}, {**VALID, 'keywords': ['x'] * 6},
    {**VALID, 'summary_ru': 'x' * 251}, {**VALID, 'keywords': 'word'}, {**VALID, 'extra': 'x'},
    {'topic': 'Наука'}, {**VALID, 'keywords': [3]}, {**VALID, 'keywords': [' ']}, {**VALID, 'insufficient_data': 'yes'},
    {**VALID, 'summary_ru': 'English summary only.'}])
def test_strict_schema(bad):
    with pytest.raises(ValidationError):
        SemanticEnrichment.model_validate(bad)


def test_null_topic_is_explicit_unknown():
    assert SemanticEnrichment.model_validate({**VALID, 'topic': None, 'insufficient_data': True}).topic is None


@pytest.mark.parametrize('status', [301, 400, 401, 403, 404, 500])
def test_http_error_no_unbounded_retry(tmp_path, status):
    session = Mock(post=Mock(return_value=response(status=status)))
    client = provider(tmp_path, session, config(retries=2))
    with pytest.raises(ProviderError, match=f'^http_{status}$'):
        client.enrich({}, {})
    assert session.post.call_count == 1


@pytest.mark.parametrize('status', [429, 503])
def test_transient_errors_retry_bounded_and_counted(tmp_path, status):
    session = Mock(post=Mock(side_effect=[response(status=status), response()]))
    client = provider(tmp_path, session, config(retries=1))
    assert client.enrich({}, {}) == VALID
    assert client.request_count == 2
    assert session.post.call_count == 2


def test_quota_pause_and_explicit_retry(tmp_path):
    session = Mock(post=Mock(return_value=response(status=429)))
    with pytest.raises(ProcessingPaused, match='http_429'):
        provider(tmp_path, session).enrich({}, {})
    session.post.return_value = response()
    with pytest.raises(ProviderError, match='http_429'):
        provider(tmp_path, session).enrich({}, {})
    assert session.post.call_count == 1
    assert provider(tmp_path, session, config(retry_failed=True)).enrich({}, {}) == VALID
    assert session.post.call_count == 2


@pytest.mark.parametrize('exception', [requests.Timeout('sensitive'), requests.ConnectionError('sensitive')])
def test_uncertain_outcome_never_resent(tmp_path, exception):
    session = Mock(post=Mock(side_effect=exception))
    with pytest.raises(ProcessingPaused, match='network_outcome_unknown'):
        provider(tmp_path, session).enrich({}, {})
    with pytest.raises(ProcessingPaused, match='uncertain_previous_request'):
        provider(tmp_path, session, config(retry_failed=True)).enrich({}, {})
    assert session.post.call_count == 1


def test_run_and_global_rolling_limit(tmp_path):
    session = Mock(post=Mock(return_value=response()))
    client = provider(tmp_path, session, config(max_requests=1, daily_limit=1))
    client.enrich({'title': 'a'}, {})
    with pytest.raises(ProcessingPaused, match='run_request_limit'):
        client.enrich({'title': 'b'}, {})
    with pytest.raises(ProcessingPaused, match='rolling_24h'):
        provider(tmp_path, session, config(daily_limit=1), run_id='new').enrich({'title': 'b'}, {})
    assert session.post.call_count == 1


@pytest.mark.parametrize('cfg', [config(api_key=''), config(free_tier_confirmed=False), config(delay=0),
    config(timeout=0), config(retries=3), config(max_requests=0), config(model='https://evil'), config(max_output_tokens=9999)])
def test_invalid_configuration_no_network(tmp_path, cfg):
    with pytest.raises(ProviderError):
        provider(tmp_path, cfg=cfg)


def test_malformed_answer_saved_and_not_resent(tmp_path):
    session = Mock(post=Mock(return_value=response(value='not an object')))
    with pytest.raises(ProviderError, match='invalid_json_response'):
        provider(tmp_path, session).enrich({}, {})
    with pytest.raises(ProviderError, match='invalid_json_response'):
        provider(tmp_path, session).enrich({}, {})
    assert session.post.call_count == 1


def test_semantic_fallback_and_missing_description(tmp_path):
    raw = tmp_path / 'raw.json'
    scrape('demo', 10, raw, ROOT / 'fixtures')
    session = Mock(post=Mock(return_value=response()))
    client = provider(tmp_path, session, config(max_requests=20))
    records = normalize(raw, tmp_path / 'normalized.json', 'gemini', provider=client)
    assert len(records) == 9 and all(r['enrichment_method'] == 'gemini' for r in records)
    empty = next(r for r in records if not r['description_raw'])
    assert empty['keywords'] == [] and empty['summary_ru'] == ''
    assert 'insufficient_semantic_data' in empty['quality_flags']
    other = tmp_path / 'other'
    session.post.return_value = response({**VALID, 'topic': 'Bad'})
    records = normalize(raw, other / 'normalized.json', 'gemini', provider=provider(other, session, config(max_requests=20)))
    assert all(r['enrichment_method'] == 'rules_fallback' for r in records)
    assert read_json(other / 'normalization_stats.json')['llm_validation_errors'] == 9


def test_resume_uses_successful_checkpoint_and_cumulative_cap(tmp_path, monkeypatch):
    session = Mock(post=Mock(return_value=response()))
    monkeypatch.setattr('coursework.llm_provider.requests.Session', lambda: session)
    monkeypatch.setattr('coursework.llm_provider.time.sleep', lambda _: None)
    # Pass sleep explicitly: its default was bound before monkeypatching.
    monkeypatch.setattr('coursework.normalize.create_provider', lambda cfg, ledger, run_id: GeminiProvider(cfg, ledger, session=session, sleep=lambda _: None, run_id=run_id))
    cfg = config(max_requests=1)
    run = execute(tmp_path, limit=3, enrich='gemini', config=cfg, api_confirmed=True, progress=lambda _: None)
    assert read_json(run / 'manifest.json')['status'] == 'processing_paused'
    assert len(read_json(run / 'gemini/normalized_books.json')) == 1
    process_run(run, enrich='gemini', config=replace(cfg, max_requests=3), api_confirmed=True, resume=True)
    manifest = read_json(run / 'manifest.json')
    assert manifest['status'] == 'completed'
    assert len(manifest['processing_sessions']) == 2
    assert session.post.call_count == 3
    assert manifest['metrics']['gemini_processed'] == 3
    assert manifest['metrics']['http_requests'] == 3
    assert manifest['metrics']['token_usage']['total_tokens'] == 90
    assert len(read_json(run / 'gemini/processing_checkpoint.json')['records']) == 3


def test_secret_echo_redacted_before_persistence(tmp_path):
    session = Mock(post=Mock(return_value=response({**VALID, 'summary_ru': 'test-only-credential'})))
    client = provider(tmp_path, session)
    assert client.enrich({}, {})['summary_ru'] == '[REDACTED]'
    assert b'test-only-credential' not in (tmp_path / 'cache.sqlite').read_bytes()


def test_crash_after_response_before_record_checkpoint_does_not_resend(tmp_path, monkeypatch):
    session = Mock(post=Mock(return_value=response()))
    cfg = config(max_requests=2)
    class Interrupted(GeminiProvider):
        def enrich(self, raw, schema):
            super().enrich(raw, schema)
            raise KeyboardInterrupt()
    monkeypatch.setattr('coursework.normalize.create_provider', lambda c, ledger, run_id: Interrupted(c, ledger, session=session, sleep=lambda _: None, run_id=run_id))
    with pytest.raises(KeyboardInterrupt):
        execute(tmp_path, limit=2, enrich='gemini', config=cfg, api_confirmed=True, progress=lambda _: None)
    run = next((tmp_path / 'runs').iterdir())
    assert read_json(run / 'manifest.json')['status'] == 'processing_interrupted'
    monkeypatch.setattr('coursework.normalize.create_provider', lambda c, ledger, run_id: GeminiProvider(c, ledger, session=session, sleep=lambda _: None, run_id=run_id))
    process_run(run, enrich='gemini', config=cfg, resume=True, api_confirmed=True)
    metrics = read_json(run / 'manifest.json')['metrics']
    assert metrics['gemini_processed'] == 2
    assert metrics['http_requests'] == session.post.call_count == 2
    assert metrics['token_usage']['total_tokens'] == 60
    assert metrics['cache_hits'] == 1


def test_two_clients_share_run_cap(tmp_path):
    session = Mock(post=Mock(return_value=response()))
    first = provider(tmp_path, session, config(max_requests=1))
    second = provider(tmp_path, session, config(max_requests=1))
    first.enrich({'title': 'a'}, {})
    with pytest.raises(ProcessingPaused, match='run_request_limit'):
        second.enrich({'title': 'b'}, {})
    assert session.post.call_count == 1
