import pytest

from coursework.secrets import migrate, audit, redact
from coursework.evaluation import evaluate, evaluate_facts, fact_id
from coursework.pipeline import execute


def test_secret_migration_and_audit(tmp_path):
    secret = 'AQ.' + 'x' * 50
    (tmp_path / 'GeminiApiKey.txt').write_text(secret)
    assert migrate(tmp_path)['source_file_required'] is False
    assert secret in (tmp_path / '.env').read_text()
    assert secret not in str(migrate(tmp_path))
    assert audit(tmp_path)['unexpected_secret_files'] == []
    (tmp_path / 'bad.log').write_text(secret)
    result = audit(tmp_path)
    assert result['unexpected_secret_files'] == ['bad.log']
    assert secret not in str(result)
    assert redact({'body': secret}, secret) == {'body': '[REDACTED]'}


def test_unconfirmed_api_launch_makes_no_run(tmp_path):
    with pytest.raises(ValueError, match='подтверждение'):
        execute(tmp_path, enrich='gemini')
    assert not (tmp_path / 'runs').exists()


def test_f1_is_manual_and_macro_includes_predicted_classes():
    records = [{'id': '1', 'topic': 'Наука', 'summary_ru': ''}, {'id': '2', 'topic': 'История', 'summary_ru': ''}]
    assert evaluate(records)['topic_macro_f1'] is None
    measured = evaluate(records, [{'id': '1', 'topic_gold': 'Наука', 'verified': 'yes'}, {'id': '2', 'topic_gold': 'Наука', 'verified': 'yes'}])
    assert measured['topic_accuracy'] == 0.5
    assert measured['topic_macro_f1'] == pytest.approx(1 / 3)
    assert measured['macro_f1_classes'] == ['История', 'Наука']


def test_fact_reviews_are_bound_to_exact_fact():
    fact = {'subject': 'book:1', 'predicate': 'topic', 'object': 'Наука', 'source_url': 'https://books.toscrape.com/1'}
    review = {**fact, 'fact_id': fact_id(fact), 'support': 'supported'}
    assert evaluate_facts([fact], [])['facts_supported_rate'] is None
    result = evaluate_facts([fact], [review])
    assert result['facts_supported_rate'] == 1
    assert result['facts_evaluated_count'] == 1
    with pytest.raises(ValueError):
        evaluate_facts([fact], [{**review, 'object': 'Changed'}])
