import copy
import csv
import sqlite3
from unittest.mock import Mock

import pytest

from coursework.dashboard import compare_runs, filter_records, list_runs, load_run, manual_metrics
from coursework.evaluation import evaluate
from coursework.normalize import normalize, parse_price
from coursework.pipeline import ROOT, evaluate_run, execute, latest_run, verify_raw
from coursework.report import generate_report
from coursework.storage import read_json, write_json


@pytest.fixture
def completed(tmp_path):
    directory = execute(tmp_path, progress=lambda _: None)
    entry = list_runs(tmp_path)[0]
    return directory, load_run(entry)


def test_run_roundtrip_exports_and_same_snapshot(completed, tmp_path):
    directory, run = completed
    before = (directory / "raw_books.json").read_bytes()
    second = execute(tmp_path, raw_path=directory / "raw_books.json", enrich="rules", progress=lambda _: None)
    assert directory != second
    assert before == (second / "raw_books.json").read_bytes() == (directory / "raw_books.json").read_bytes()
    assert verify_raw(directory)["raw_sha256"] == verify_raw(second)["raw_sha256"]
    assert latest_run(tmp_path) == second
    second_run = load_run(list_runs(tmp_path)[0])
    assert second_run["metrics"]["llm_processed"] == 0
    assert second_run["metrics"]["mock_processed"] == 0
    with pytest.raises(ValueError):
        compare_runs(run, second_run)
    result = run["result"]
    with (result / "knowledge_source.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    json_rows = read_json(result / "knowledge_source.json")
    with sqlite3.connect(result / "knowledge_source.sqlite") as db:
        db.row_factory = sqlite3.Row
        sql_rows = [dict(row) for row in db.execute("SELECT * FROM books")]
        assert db.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 63
    assert len(csv_rows) == len(json_rows) == len(sql_rows) == 9
    assert set(csv_rows[0]) == set(json_rows[0]) == set(sql_rows[0])
    assert [r["id"] for r in csv_rows] == [r["id"] for r in sql_rows]
    for j, s in zip(json_rows, sql_rows):
        assert j["description_raw"] == s["description_raw"]
        assert j["price_gbp"] == s["price_gbp"]
    with (result / "facts_triples.csv").open(encoding="utf-8-sig") as stream:
        facts = list(csv.DictReader(stream))
    assert len(facts) == 63 and all(f["source_url"].startswith("https://books.toscrape.com/") for f in facts)
    assert read_json(result / "schema.json")["additionalProperties"] is False
    assert manual_metrics(run)["topic_accuracy"] is None
    assert evaluate_run(directory)["summary_evaluated_count"] == 0


def test_filters_and_comparison_guard(completed):
    _, run = completed
    records = run["records"]
    assert len(filter_records(records, search="HARBOR")) == 1
    assert filter_records(records, topics=[]) == []
    assert filter_records(records, ratings=[]) == []
    assert filter_records(records, search="missing") == []
    target = copy.deepcopy(run)
    target["manifest"]["method"] = "gemini"
    target["manifest"]["configuration"] = {"provider": "gemini"}
    for record in target["records"]:
        record["enrichment_method"] = "gemini"
    target["records"][0]["topic"] = "Другое"
    comparison = compare_runs(run, target)
    assert len(comparison) == 9 and comparison["different"].sum() == 1
    target["manifest"]["raw_sha256"] = "other"
    with pytest.raises(ValueError, match="SHA-256"):
        compare_runs(run, target)


def test_raw_tamper_rejected(completed):
    directory, _ = completed
    with (directory / "raw_books.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="снимок"):
        verify_raw(directory)


def test_empty_and_invalid_raw(tmp_path):
    raw, normalized = tmp_path / "raw.json", tmp_path / "normalized.json"
    write_json(raw, [])
    assert normalize(raw, normalized) == []
    metrics = generate_report(raw, normalized, tmp_path / "reports")
    assert metrics["normalized_records"] == metrics["exported_facts"] == 0
    assert metrics["llm_success_rate"] is None
    write_json(raw, [{"title": "Broken"}, None])
    assert normalize(raw, normalized) == []
    metrics = generate_report(raw, normalized, tmp_path / "reports")
    assert metrics["invalid_raw_records"] == 2
    assert metrics["removed_duplicates"] == 0


@pytest.mark.parametrize("value, expected", [("£12,50", 12.5), ("-1", None), ("£1.999", None), ("n/a", None), ("0", 0.0), ("£12 text", None), ("9" * 400, None)])
def test_malformed_prices(value, expected):
    assert parse_price(value) == expected


def test_manual_metrics_denominator_and_stale_review():
    records = [{"id": "a", "topic": "Наука", "summary_ru": "Текст"}, {"id": "b", "topic": "Другое", "summary_ru": ""}]
    gold = [{"id": "a", "topic_gold": "Наука"}, {"id": "b", "topic_gold": "История"}, {"id": "outside", "topic_gold": "Наука"}]
    reviews = [{"id": "a", "summary_ru": "Текст", "support": "unsupported"}]
    gold = [{**row, 'verified': 'yes'} for row in gold]
    metrics = evaluate(records, gold, reviews)
    assert metrics["topic_accuracy"] == 0.5
    assert metrics["topic_labeled_count"] == 2
    assert metrics["summary_evaluated_count"] == 1
    assert metrics["summary_unsupported_rate"] == 1
    with pytest.raises(ValueError, match="повторяющиеся"):
        evaluate(records, gold + [gold[0]])
    with pytest.raises(ValueError, match="другому"):
        evaluate(records, reviews=[{**reviews[0], "summary_ru": "Старый текст"}])


def test_failed_configuration_records_run_status(tmp_path, monkeypatch):
    from coursework.llm_provider import ProviderError
    monkeypatch.setenv("GEMINI_TIMEOUT", "invalid")
    with pytest.raises(ProviderError, match="invalid_numeric_configuration"):
        execute(tmp_path, limit=1, enrich="gemini", api_confirmed=True, progress=lambda _: None)
    assert not (tmp_path / 'runs').exists()


def test_malformed_import_reports_rejected_records(tmp_path):
    raw = tmp_path / "input.json"
    write_json(raw, [{"source_mode": []}, {"source_mode": {}}, 5, None])
    run = execute(tmp_path, raw_path=raw, progress=lambda _: None)
    manifest = read_json(run / "manifest.json")
    assert manifest["status"] == "completed_with_errors"
    assert manifest["metrics"]["invalid_raw_records"] == 4
    assert manifest["metrics"]["removed_duplicates"] == 0
