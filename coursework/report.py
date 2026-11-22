from __future__ import annotations

from pathlib import Path
from contextlib import closing
from collections import Counter
import json
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from .normalize import KnowledgeRecord
from .storage import read_json, write_json
from .status import event_metrics


def generate_report(raw_path: Path, normalized_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    records = json.loads(normalized_path.read_text(encoding="utf-8"))
    records = [KnowledgeRecord.model_validate(r).model_dump() for r in records]
    df = pd.DataFrame(records, columns=list(KnowledgeRecord.model_fields))
    csv_df = df.copy()
    for name in ("keywords", "quality_flags"):
        csv_df[name] = csv_df[name].apply(lambda value: json.dumps(value, ensure_ascii=False))
    csv_df.to_csv(out_dir / "knowledge_source.csv", index=False, encoding="utf-8-sig")
    facts = []
    for rec in records:
        for attribute in ("title", "category_raw", "topic", "price_gbp", "rating", "in_stock", "quantity", "summary_ru"):
            value = rec.get(attribute)
            if value is None or value == "":
                continue
            facts.append({
                "subject": "book:" + rec["id"], "predicate": attribute,
                "object": json.dumps(value, ensure_ascii=False) if isinstance(value, bool) else str(value),
                "source_url": rec["source_url"],
            })
        for word in rec["keywords"]:
            facts.append({
                "subject": "book:" + rec["id"], "predicate": "keyword",
                "object": word, "source_url": rec["source_url"],
            })
    facts_df = pd.DataFrame(facts, columns=["subject", "predicate", "object", "source_url"])
    facts_df.to_csv(out_dir / "facts_triples.csv", index=False, encoding="utf-8-sig")
    with closing(sqlite3.connect(out_dir / "knowledge_source.sqlite")) as connection, connection:
        csv_df.to_sql("books", connection, if_exists="replace", index=False,
                      dtype={"price_gbp": "REAL", "rating": "INTEGER", "quantity": "INTEGER", "in_stock": "INTEGER"})
        facts_df.to_sql("facts", connection, if_exists="replace", index=False)
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_books_id ON books (id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts (subject)")
    (out_dir / "knowledge_source.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    required = ("title", "category_raw", "description_raw", "upc", "price_gbp", "rating", "topic", "summary_ru")
    def nonempty(value):
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return not pd.isna(value)
    coverage = {k: round(100 * sum(nonempty(v) for v in df[k]) / len(df), 1) if len(df) else 0 for k in required}
    raw_fields = ('title', 'category_raw', 'description_raw', 'upc_raw', 'price_raw', 'rating_raw', 'availability_raw', 'source_url')
    raw_coverage = {k: round(100 * sum(isinstance(r, dict) and isinstance(r.get(k), str) and bool(r[k].strip())
                    for r in raw) / len(raw), 1) if raw else 0 for k in raw_fields}
    stats_path = normalized_path.with_name("normalization_stats.json")
    stats = read_json(stats_path) if stats_path.exists() else {}
    methods = Counter(r["enrichment_method"] for r in records)
    attempts = stats.get("llm_attempted_records", methods["gemini"] + methods["llm"] + methods["rules_fallback"])
    metrics = {
        "source_mode": ",".join(sorted({r["source_mode"] for r in records})) or "empty",
        "raw_records": len(raw),
        "normalized_records": len(records),
        "removed_duplicates": stats.get("removed_duplicates", len(raw) - len(records)),
        "invalid_raw_records": stats.get("invalid_raw_records", 0),
        "schema_valid_records": len(records),
        "completeness_pct": coverage,
        "raw_completeness_pct": raw_coverage,
        "llm_processed": methods["gemini"] + methods["llm"],
        "gemini_processed": methods["gemini"],
        "llm_fallbacks": int((df["enrichment_method"] == "rules_fallback").sum()),
        "mock_processed": methods["mock"],
        "llm_attempted_records": attempts,
        "llm_success_rate": stats.get("fresh_successes", methods["llm"]) / attempts if attempts else None,
        "cache_hits": stats.get("cache_hits", 0),
        "token_usage": stats.get("usage", {}),
        "pending_records": stats.get("pending_records", 0),
        "processing_errors": sum(bool(c.get("error")) for c in stats.get("calls", [])),
        "missing_values": sum(sum(not nonempty(v) for v in df[k]) for k in required),
        "llm_validation_errors": stats.get("llm_validation_errors", 0),
        "http_requests": stats.get("http_requests", 0),
        "fallback_pct": round(100 * methods["rules_fallback"] / len(records), 1) if records else 0,
        "annotation_pct": coverage["summary_ru"],
        "duration_seconds": stats.get("duration_seconds"),
        "quality_flag_counts": dict(Counter(flag for r in records for flag in r["quality_flags"])),
        "topic_counts": {str(k): int(v) for k, v in df["topic"].value_counts().items()},
        "avg_price_gbp": round(float(df["price_gbp"].mean()), 2) if df["price_gbp"].notna().any() else None,
        "rows_requiring_review": int(df["quality_flags"].apply(bool).sum()),
        "exported_facts": len(facts),
    }
    metrics.update(event_metrics(stats))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_json(out_dir / "schema.json", KnowledgeRecord.model_json_schema())

    plt.rcParams["font.family"] = "DejaVu Sans"
    counts = df["topic"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    if not counts.empty:
        counts.plot(kind="barh", ax=ax)
    ax.set_title("Распределение книг по нормализованным темам")
    ax.set_xlabel("Количество книг")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_dir / "topics.png", dpi=150)
    plt.close(fig)

    prices = pd.to_numeric(df["price_gbp"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    if not prices.empty:
        ax.hist(prices, bins=min(12, max(3, len(prices))))
    ax.set_title("Распределение цен (учебные данные сайта)")
    ax.set_xlabel("Цена, GBP")
    ax.set_ylabel("Количество книг")
    fig.tight_layout()
    fig.savefig(out_dir / "price_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = sorted(coverage.items(), key=lambda item: item[1])
    ax.barh([k for k, _ in ordered], [v for _, v in ordered])
    ax.set_xlim(0, 105)
    ax.set_title("Заполненность полей структурированного набора")
    ax.set_xlabel("Заполнено, %")
    for i, (_, percent) in enumerate(ordered):
        ax.text(min(percent + 1, 101), i, f"{percent}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "completeness.png", dpi=150)
    plt.close(fig)
    return metrics
