from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Literal
import json
import re
import time
import math

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from .llm_provider import ProviderConfig, ProviderError, ProcessingPaused, create_provider, PROMPT_VERSION
from .storage import read_json, write_json, file_hash

TOPICS = (
    "Детектив", "Фантастика", "Наука", "История", "Бизнес",
    "Искусство", "Путешествия", "Другое"
)
RATING = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
CATEGORY_MAP = {
    "mystery": "Детектив", "crime": "Детектив", "thriller": "Детектив", "suspense": "Детектив",
    "science fiction": "Фантастика", "fantasy": "Фантастика", "paranormal": "Фантастика",
    "science": "Наука", "psychology": "Наука", "academic": "Наука",
    "history": "История", "historical": "История", "biography": "История",
    "business": "Бизнес", "economics": "Бизнес",
    "art": "Искусство", "music": "Искусство", "poetry": "Искусство", "sequential art": "Искусство",
    "travel": "Путешествия",
}


class SemanticEnrichment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    topic: Literal["Детектив", "Фантастика", "Наука", "История", "Бизнес", "Искусство", "Путешествия", "Другое"] | None
    keywords: list[str] = Field(max_length=5)
    summary_ru: str = Field(max_length=250)
    insufficient_data: bool

    @field_validator('summary_ru')
    @classmethod
    def russian_summary(cls, value):
        if value and not re.search(r'[А-Яа-яЁё]', value):
            raise ValueError('A nonempty summary must contain Russian text')
        return value

    @field_validator("keywords")
    @classmethod
    def clean_keywords(cls, words):
        if any(not word.strip() or len(word.strip()) > 80 for word in words):
            raise ValueError("Keywords must contain 1..80 characters")
        return list(dict.fromkeys(word.strip() for word in words))


class RawRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_url: str = Field(min_length=1)
    title: str = Field(min_length=1)
    category_raw: str = ""
    description_raw: str = ""
    price_raw: str = ""
    rating_raw: str = ""
    availability_raw: str = ""
    upc_raw: str = ""
    collected_at: str = Field(min_length=1)
    source_mode: Literal["demo", "live"]

    @field_validator("title")
    @classmethod
    def nonblank_title(cls, value):
        if not value.strip():
            raise ValueError("empty title")
        return value


class KnowledgeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str
    source_url: str
    source_mode: Literal["demo", "live"]
    collected_at: str
    title: str = Field(min_length=1)
    upc: str
    category_raw: str
    description_raw: str
    price_gbp: float | None
    rating: int | None
    in_stock: bool | None
    quantity: int | None
    topic: Literal["Детектив", "Фантастика", "Наука", "История", "Бизнес", "Искусство", "Путешествия", "Другое"]
    keywords: list[str]
    summary_ru: str
    enrichment_method: Literal["rules", "gemini", "rules_fallback", "llm", "mock"]  # llm/mock are accepted for legacy imports.
    quality_flags: list[str]
    model: str | None = None
    processing_status: Literal['success', 'fallback', 'legacy'] = 'legacy'
    is_fallback: bool = False
    experiment_id: str = ''
    run_id: str = ''

    @field_validator("rating")
    @classmethod
    def rating_range(cls, rating: int | None) -> int | None:
        if rating is not None and rating not in range(1, 6):
            raise ValueError("rating outside 1..5")
        return rating

    @field_validator("price_gbp")
    @classmethod
    def price_nonnegative(cls, price: float | None) -> float | None:
        if price is not None and price < 0:
            raise ValueError("negative price")
        return price


def parse_price(raw: str) -> float | None:
    value = raw.strip().replace("£", "").replace("Â", "").strip()
    if not re.fullmatch(r"\d+(?:[.,]\d{1,2})?", value):
        return None
    price = float(value.replace(",", "."))
    return price if math.isfinite(price) else None


def parse_stock(raw: str) -> tuple[bool | None, int | None]:
    value = raw.casefold()
    if "out of stock" in value:
        return False, 0
    if "in stock" in value:
        match = re.search(r"\((\d+) available\)", value)
        return True, int(match.group(1)) if match else None
    return None, None


def topic_from_rules(category_raw: str) -> str:
    return CATEGORY_MAP.get(category_raw.casefold().strip(), "Другое")


def normalize(input_file: Path, output_file: Path, enrich: str = "rules", model: str | None = None,
              provider=None, config: ProviderConfig | None = None, progress=None,
              ledger: Path | None = None, resume=False, experiment_id='', run_id='') -> list[dict]:
    if enrich not in ("rules", "gemini"):
        raise ValueError("enrich must be rules or gemini")
    rows = read_json(input_file)
    if not isinstance(rows, list):
        raise ValueError("Expected a JSON array of raw books")
    if enrich == "gemini" and provider is None:
        config = config or ProviderConfig.from_env(model)
        provider = create_provider(config, ledger or output_file.parent / "gemini_cache.sqlite",
                                   run_id=output_file.parent.parent.name)
    checkpoint_path = output_file.with_name("processing_checkpoint.json")
    signature = {"raw_sha256": file_hash(input_file), "method": enrich, "model": config.model if config else None,
                 "max_output_tokens": config.max_output_tokens if config else None, "prompt_version": PROMPT_VERSION}
    stats = {"raw_records": len(rows), "removed_duplicates": 0, "invalid_raw_records": 0,
             "llm_attempted_records": 0, "llm_validation_errors": 0, "record_errors": [], "calls": [],
             "duration_seconds": 0, "http_requests": 0, "cache_hits": 0, "fresh_successes": 0,
             "usage": {}, "attempted_rows": [], "pending_records": len(rows)}
    normalized, next_index = [], 0
    if resume and checkpoint_path.exists():
        checkpoint = read_json(checkpoint_path)
        if checkpoint["signature"] != signature:
            raise ValueError("Нельзя продолжить обработку с другим снимком, моделью или промптом")
        normalized, stats, next_index = checkpoint["records"], checkpoint["stats"], checkpoint["next_index"]
    seen_urls = {r["source_url"] for r in normalized}
    seen_upcs = {r["upc"] for r in normalized if r["upc"]}
    tick = time.perf_counter()

    def save(index):
        nonlocal tick
        now = time.perf_counter()
        stats["duration_seconds"] += now - tick
        tick = now
        stats["http_requests"] = getattr(provider, "request_count", 0)
        stats["pending_records"] = len(rows) - index
        write_json(checkpoint_path, {"signature": signature, "next_index": index, "records": normalized, "stats": stats})
        write_json(output_file, normalized)
        write_json(output_file.with_name("normalization_stats.json"), stats)

    save(next_index)
    for index in range(next_index, len(rows)):
        try:
            row = RawRecord.model_validate(rows[index]).model_dump()
        except ValidationError:
            stats["invalid_raw_records"] += 1
            stats["record_errors"].append({"row": index, "error": "invalid_raw_schema"})
            save(index + 1)
            continue
        url, upc = row["source_url"], row["upc_raw"].strip()
        if url in seen_urls or (upc and upc in seen_upcs):
            stats["removed_duplicates"] += 1
            save(index + 1)
            continue
        price, rating = parse_price(row["price_raw"]), RATING.get(row["rating_raw"])
        in_stock, qty = parse_stock(row["availability_raw"])
        flags = []
        for missing, flag in ((price is None, "missing_price"), (rating is None, "missing_rating"),
                (not upc, "missing_upc"), (not row["description_raw"].strip(), "missing_description"),
                (in_stock is None, "missing_availability")):
            if missing:
                flags.append(flag)
        semantic = SemanticEnrichment(topic=topic_from_rules(row["category_raw"]), keywords=[], summary_ru="",
                                      insufficient_data=not bool(row["description_raw"].strip()))
        method = "rules"
        if enrich == "gemini":
            started = time.perf_counter()
            error, paused = None, False
            try:
                semantic = SemanticEnrichment.model_validate(provider.enrich(row, SemanticEnrichment.model_json_schema()))
                method = "gemini"
            except ProcessingPaused as exc:
                error, paused = str(exc), True
            except (ProviderError, ValidationError) as exc:
                error = str(exc) if isinstance(exc, ProviderError) else "invalid_semantic_schema"
                if error in ("http_400", "http_401", "http_403", "http_404"):
                    paused = True
                stats["llm_validation_errors"] += isinstance(exc, ValidationError)
                flags.append("llm_processing_failed")
                method = "rules_fallback"
            except BaseException:
                save(index)
                raise
            response = getattr(provider, "last_response", {})
            requests_made = response.get("http_requests", 0)
            cached = bool(response.get("cached"))
            charged_here = not cached or response.get("origin_run") == getattr(provider, 'run_id', None)
            if (requests_made or (cached and charged_here)) and index not in stats["attempted_rows"]:
                stats["attempted_rows"].append(index)
            stats["llm_attempted_records"] = len(stats["attempted_rows"])
            stats["cache_hits"] += cached
            stats["fresh_successes"] += method == "gemini" and charged_here and not paused
            if charged_here:
                for key, value in response.get("usage", {}).items():
                    stats["usage"][key] = stats["usage"].get(key, 0) + value
            event = {"row": index, "status": "paused" if paused else method, "error": error,
                     "duration_seconds": round(time.perf_counter() - started, 6), **response}
            stats["calls"].append(event)
            if paused:
                save(index)
                raise ProcessingPaused(error)
        if semantic.insufficient_data or semantic.topic is None or not row['description_raw'].strip():
            flags.append("insufficient_semantic_data")
        if semantic.topic is None:
            flags.append("unknown_topic")
        keywords, summary = semantic.keywords, semantic.summary_ru
        if not row["description_raw"].strip():
            keywords, summary = [], ""
        record = KnowledgeRecord(
            id=upc or sha256(url.encode()).hexdigest()[:16], source_url=url,
            source_mode=row["source_mode"], collected_at=row["collected_at"],
            title=" ".join(row["title"].split()), upc=upc,
            category_raw=" ".join(row["category_raw"].split()),
            description_raw=row["description_raw"], price_gbp=price, rating=rating,
            in_stock=in_stock, quantity=qty, topic=semantic.topic or "Другое",
            keywords=keywords, summary_ru=summary, enrichment_method=method, quality_flags=flags,
            model=(config.model if config else getattr(getattr(provider, 'config', None), 'model', model)) if enrich == 'gemini' else None,
            processing_status='fallback' if method == 'rules_fallback' else 'success',
            is_fallback=method == 'rules_fallback', experiment_id=experiment_id or run_id, run_id=run_id,
        )
        normalized.append(record.model_dump())
        seen_urls.add(url)
        if upc:
            seen_upcs.add(upc)
        save(index + 1)
        if progress:
            progress(f"Обработка: {index + 1}/{len(rows)} · {method} · API-запросов: {stats['http_requests']}")
    return normalized
