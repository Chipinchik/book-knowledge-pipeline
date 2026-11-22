from pathlib import Path
import pandas as pd

from .evaluation import evaluate, read_labels, evaluate_facts
from .normalize import KnowledgeRecord
from .pipeline import verify_raw
from .storage import file_hash, read_json
from .status import event_metrics


def list_runs(output: Path) -> list[dict]:
    runs = []
    for path in sorted((output / "runs").glob("*/manifest.json"), reverse=True):
        try:
            manifest = read_json(path)
            runs.append({"directory": path.parent, "manifest": manifest,
                         "label": f"{manifest['run_id']} · {manifest.get('source')} · {manifest.get('method', 'сбор')} · {manifest['status']}"})
        except (ValueError, OSError, KeyError, TypeError):
            runs.append({"directory": path.parent, "manifest": {"status": "unreadable"},
                         "label": f"{path.parent.name} · ошибка manifest.json"})
    for directory in [output, *sorted(p for p in output.glob("*") if p.is_dir() and p.name not in ('runs', 'archive', 'verification', 'experiments'))]:
        if (directory / "knowledge_source.json").exists():
            runs.append({"directory": directory, "manifest": {"legacy": True},
                         "label": f"Прежний формат · {directory.name}"})
    return runs


def load_run(entry: dict) -> dict:
    directory = entry["directory"]
    manifest = entry["manifest"]
    if manifest.get("legacy"):
        result = directory
    else:
        manifest = verify_raw(directory)
        if manifest.get("status") not in ("completed", "completed_with_errors", "processing_paused"):
            raise ValueError("Запуск не завершён. Просмотрите manifest.json и progress.log; сбор можно продолжить через CLI.")
        if manifest.get("result_dir") not in ("rules", "gemini", "llm", "mock"):
            raise ValueError("Неизвестная папка результата")
        result = directory / manifest["result_dir"]
        if file_hash(result / "knowledge_source.json") != manifest.get("result_sha256"):
            raise ValueError("Нормализованный результат изменён; повторите обработку исходного снимка")
    records = [KnowledgeRecord.model_validate(r).model_dump() for r in read_json(result / "knowledge_source.json")]
    metrics = read_json(result / "metrics.json")
    stored_metrics = dict(metrics)
    stats_path = result / 'normalization_stats.json'
    if stats_path.exists():
        metrics = {**metrics, **event_metrics(read_json(stats_path))}
    return {"directory": directory, "result": result, "manifest": manifest, "records": records,
            "metrics": metrics, "stored_metrics": stored_metrics}


def filter_records(records: list[dict], search="", topics=None, ratings=None, methods=None, price_range=None,
                   include_missing_price=True) -> list[dict]:
    return [r for r in records if search.strip().casefold() in r["title"].casefold()
            and (topics is None or r["topic"] in topics)
            and (ratings is None or r["rating"] in ratings)
            and (methods is None or r["enrichment_method"] in methods)
            and (price_range is None or (r['price_gbp'] is None and include_missing_price)
                 or (r['price_gbp'] is not None and price_range[0] <= r['price_gbp'] <= price_range[1]))]


def compare_runs(left: dict, right: dict) -> pd.DataFrame:
    a, b = left["manifest"], right["manifest"]
    if not a.get("raw_sha256") or a["raw_sha256"] != b.get("raw_sha256"):
        raise ValueError("Сравнение разрешено только для одного исходного снимка (SHA-256)")
    if a.get('purpose') != b.get('purpose') or (str(a.get('experiment_id', '')).startswith('exp-')
            and a.get('experiment_id') != b.get('experiment_id')):
        raise ValueError('Нельзя смешивать разные эксперименты или техническую и итоговую выборки')
    if a.get("method") != "rules" or b.get("method") != "gemini" or b.get("configuration", {}).get("provider") != "gemini":
        raise ValueError("Сравнивайте только Rules и Gemini; исторические методы исключены")
    if a.get("status") not in ("completed", "completed_with_errors") or b.get("status") not in ("completed", "completed_with_errors"):
        raise ValueError("Сравнение возможно только после завершения обоих запусков")
    if not any(r["enrichment_method"] == "gemini" for r in right["records"]):
        raise ValueError("Нет успешных результатов LLM: все записи обработаны через fallback")
    if {r["id"] for r in left["records"]} != {r["id"] for r in right["records"]}:
        raise ValueError("Наборы нормализованных записей отличаются; сравнение некорректно")
    right_by_id = {r["id"]: r for r in right["records"]}
    rows = []
    for rules in left["records"]:
        llm = right_by_id[rules["id"]]
        rows.append({"id": rules["id"], "title": rules["title"], "rules_topic": rules["topic"],
                     "llm_topic": llm["topic"], "rules_keywords": rules["keywords"], "llm_keywords": llm["keywords"],
                     "rules_summary": rules["summary_ru"], "llm_summary": llm["summary_ru"],
                     "llm_status": llm["enrichment_method"],
                     "different": any(rules[k] != llm[k] for k in ("topic", "keywords", "summary_ru"))})
    return pd.DataFrame(rows)


def manual_metrics(run: dict, gold_path: Path | None = None) -> dict:
    result = run["result"]
    gold = gold_path or result / "gold_labels.csv"
    reviews = result / "summary_reviews.csv"
    metrics = evaluate(run["records"], read_labels(gold) if gold.exists() else [],
                       read_labels(reviews) if reviews.exists() else [])
    facts = result / 'facts_triples.csv'
    fact_reviews = result / 'fact_reviews.csv'
    metrics.update(evaluate_facts(read_labels(facts), read_labels(fact_reviews) if fact_reviews.exists() else []))
    return metrics
