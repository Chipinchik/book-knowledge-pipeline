import csv
import json
from hashlib import sha256
from pathlib import Path
from .normalize import TOPICS


def write_templates(records: list[dict], directory: Path):
    templates = {
        "gold_labels.csv": ["id", "title", "source_url", "topic_gold", "verified", "notes"],
        "summary_reviews.csv": ["id", "title", "source_url", "summary_ru", "support", "notes"],
    }
    for filename, fields in templates.items():
        path = directory / filename
        existing = read_labels(path) if path.exists() else []
        existing_by_id = {row['id']: row for row in existing}
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in records:
                row = {key: record.get(key, "") for key in fields}
                if "support" in fields:
                    row["support"] = "not_evaluated"
                writer.writerow(existing_by_id.get(record['id'], row))
    facts_path = directory / "facts_triples.csv"
    if facts_path.exists():
        reviews_path = directory / "fact_reviews.csv"
        existing = {r['fact_id']: r for r in read_labels(reviews_path)} if reviews_path.exists() else {}
        with reviews_path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['fact_id', 'subject', 'predicate', 'object', 'source_url', 'support', 'notes'])
            writer.writeheader()
            for fact in read_labels(facts_path):
                identifier = fact_id(fact)
                writer.writerow(existing.get(identifier, {**fact, 'fact_id': identifier, 'support': 'not_evaluated', 'notes': ''}))


def read_labels(path: Path | None) -> list[dict]:
    if path is None:
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def evaluate(records: list[dict], gold: list[dict] | None = None,
             reviews: list[dict] | None = None) -> dict:
    by_id = {record["id"]: record for record in records}
    labeled = correct = supported = unsupported = unverified = 0
    pairs = []
    for rows, kind in ((gold or [], "topic"), (reviews or [], "summary")):
        seen = set()
        for row in rows:
            identifier = row.get("id", "")
            if not identifier or identifier in seen:
                raise ValueError("В разметке отсутствует id или есть повторяющиеся id")
            seen.add(identifier)
            value = row.get("topic_gold" if kind == "topic" else "support", "").strip()
            allowed = (*TOPICS, "") if kind == "topic" else ("", "not_evaluated", "supported", "unsupported")
            if value not in allowed:
                raise ValueError("Недопустимая метка в ручной разметке")
            record = by_id.get(identifier)
            if record is None:
                continue
            if kind == "topic" and value:
                verified = row.get('verified', '').strip().lower()
                if verified not in ('', 'no', 'false', '0', 'нет', 'yes', 'true', '1', 'да'):
                    raise ValueError('В verified укажите yes/да после проверки эталонной метки')
                if verified not in ('yes', 'true', '1', 'да'):
                    unverified += 1
                    continue
                labeled += 1
                correct += record["topic"] == value
                pairs.append((value, record["topic"]))
            elif kind == "summary" and value in ("supported", "unsupported"):
                if not record["summary_ru"] or row.get("summary_ru") != record["summary_ru"]:
                    raise ValueError("Оценка аннотации относится к другому или пустому тексту")
                supported += value == "supported"
                unsupported += value == "unsupported"
    classes = sorted({label for pair in pairs for label in pair})
    f1_scores = []
    for label in classes:
        tp = sum(gold == predicted == label for gold, predicted in pairs)
        fp = sum(gold != label and predicted == label for gold, predicted in pairs)
        fn = sum(gold == label and predicted != label for gold, predicted in pairs)
        f1_scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return {"topic_accuracy": correct / labeled if labeled else None,
            "topic_macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else None,
            "macro_f1_classes": classes,
            "topic_labeled_count": labeled, "topic_correct_count": correct,
            "topic_unverified_count": unverified,
            "summary_evaluated_count": supported + unsupported,
            "summary_supported": supported, "summary_unsupported": unsupported,
            "summary_unsupported_rate": unsupported / (supported + unsupported) if supported + unsupported else None}


def fact_id(fact):
    payload = {k: fact[k] for k in ('subject', 'predicate', 'object', 'source_url')}
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def evaluate_facts(facts, reviews):
    known = {fact_id(f) for f in facts}
    seen, counts = set(), {'supported': 0, 'unsupported': 0}
    for row in reviews:
        identifier, label = row.get('fact_id'), row.get('support', '').strip()
        if identifier in seen or identifier not in known or fact_id(row) != identifier:
            raise ValueError('Оценка факта не соответствует текущему набору или повторяется')
        seen.add(identifier)
        if label not in ('', 'not_evaluated', 'supported', 'unsupported'):
            raise ValueError('Недопустимая оценка факта')
        if label in counts:
            counts[label] += 1
    total = sum(counts.values())
    return {'facts_evaluated_count': total, 'facts_supported': counts['supported'], 'facts_unsupported': counts['unsupported'],
            'facts_supported_rate': counts['supported'] / total if total else None}
