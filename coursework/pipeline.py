from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
from uuid import uuid4

from .evaluation import evaluate, read_labels, write_templates, evaluate_facts
from .llm_provider import ProviderConfig, ProcessingPaused
from .normalize import normalize
from .report import generate_report
from .scraper import scrape
from .storage import file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "3.0"


def now():
    return datetime.now(timezone.utc).isoformat()


def new_run(output: Path, source: str, limit: int, delay: float) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:6]
    directory = output / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "manifest.json", {
        "run_id": run_id, "created_at": now(), "schema_version": SCHEMA_VERSION,
        "source": source, "limit": limit, "delay": delay, "status": "created",
    })
    return directory


def update_manifest(directory: Path, **changes):
    manifest = read_json(directory / "manifest.json")
    manifest.update(changes, updated_at=now())
    write_json(directory / "manifest.json", manifest)
    return manifest


def verify_raw(directory: Path):
    manifest = read_json(directory / "manifest.json")
    if not manifest.get("raw_sha256") or file_hash(directory / "raw_books.json") != manifest["raw_sha256"]:
        raise ValueError("Исходный снимок изменён или не завершён; сравнение запрещено")
    return manifest


def latest_run(output: Path) -> Path | None:
    pointer = output / "latest.json"
    if pointer.exists():
        run = output / "runs" / read_json(pointer)["run_id"]
        if run.is_dir() and run.parent.resolve() == (output / "runs").resolve():
            return run
    return None


def execute(output: Path = ROOT / "outputs", *, source="demo", limit=10, delay=1.5,
            enrich="rules", model: str | None = None, config: ProviderConfig | None = None,
            raw_path: Path | None = None, scrape_only=False, resume: Path | None = None,
            progress=None, process_limit: int | None = None, api_confirmed=False,
            experiment_id=None, purpose='technical', on_created=None) -> Path:
    if enrich not in ("rules", "gemini"):
        raise ValueError("Допустимы только rules и gemini")
    if enrich == "gemini" and not scrape_only:
        if not api_confirmed:
            raise ValueError("Нужно явное подтверждение API-запросов Gemini")
        config = config or ProviderConfig.from_env(model)
        config.validate()
    log_callback = progress or (lambda message: print(message, flush=True))
    if resume:
        directory = resume.resolve()
        manifest = read_json(directory / "manifest.json")
        if manifest["status"] not in ("collecting", "collection_failed", "interrupted") or manifest.get("raw_sha256"):
            raise ValueError("Продолжать можно только незавершённый сбор")
        source, limit, delay = manifest["source"], manifest["limit"], manifest["delay"]
    else:
        directory = new_run(output, source, limit, delay)
        update_manifest(directory, experiment_id=experiment_id or directory.name,
                        purpose='demo' if source == 'demo' and raw_path is None and purpose == 'technical' else purpose)
        if on_created:
            on_created(directory)

    def log(message):
        with (directory / "progress.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{now()} {message}\n")
        log_callback(message)

    log(f"Запуск: {directory}")
    raw = directory / "raw_books.json"
    try:
        if raw_path is not None:
            # Reject modified snapshots before importing.
            if (raw_path.parent / "manifest.json").exists():
                verify_raw(raw_path.parent)
            rows = read_json(raw_path)
            if not isinstance(rows, list):
                raise ValueError("Ожидался JSON-массив сырых записей")
            if process_limit is not None:
                if not 1 <= process_limit <= len(rows):
                    raise ValueError("Размер подвыборки должен быть от 1 до размера снимка")
                rows = rows[:process_limit]
            if process_limit is None or process_limit == len(read_json(raw_path)):
                shutil.copyfile(raw_path, raw)
            else:
                write_json(raw, rows)
            snapshots = raw_path.parent / "html_snapshots"
            if snapshots.is_dir():
                shutil.copytree(snapshots, directory / "html_snapshots")
            modes = sorted({r.get("source_mode") if r.get("source_mode") in ("demo", "live") else "unknown"
                            for r in rows if isinstance(r, dict)})
            update_manifest(directory, source=",".join(modes) or "empty", imported_raw=True,
                            parent_raw_sha256=file_hash(raw_path), limit=len(rows), selection_first_n=process_limit)
        else:
            update_manifest(directory, status="collecting")
            rows = scrape(source, limit, raw, ROOT / "fixtures", delay, progress=log, resume=bool(resume))
        scrape_errors = read_json(directory / "scrape_errors.json") if (directory / "scrape_errors.json").exists() else []
        update_manifest(directory, status="collected", raw_sha256=file_hash(raw), raw_records=len(rows),
                        scrape_error_count=len(scrape_errors), collection_completed_at=now())
    except KeyboardInterrupt:
        update_manifest(directory, status="interrupted")
        log("Сбор прерван. Checkpoint сохранён; используйте --resume с папкой этого запуска.")
        raise
    except Exception as exc:
        update_manifest(directory, status="collection_failed", error_type=type(exc).__name__)
        log("Ошибка сбора/импорта; накопленные записи сохранены.")
        raise
    if not scrape_only:
        process_run(directory, enrich=enrich, model=model, config=config, progress=log, api_confirmed=api_confirmed)
    write_json(output / "latest.json", {"run_id": directory.name})
    return directory


def process_run(directory: Path, *, enrich="rules", model=None, config=None, progress=None,
                api_confirmed=False, resume=False):
    manifest = verify_raw(directory)
    if enrich == 'rules':
        config = None
    if enrich not in ("rules", "gemini"):
        raise ValueError("Допустимы только rules и gemini")
    if enrich == "gemini":
        if not api_confirmed:
            raise ValueError("Нужно явное подтверждение API-запросов Gemini")
        config = config or ProviderConfig.from_env(model)
        config.validate()
    method = enrich
    result = directory / method
    if resume and (manifest.get("method") != method or manifest.get("status") not in
                   ("processing_paused", "processing_interrupted", "processing_failed", "processing")):
        raise ValueError("Продолжать можно только незавершённую обработку того же метода")
    if resume and config:
        previous = manifest.get("configuration", {})
        if previous.get("model") != config.model or previous.get("max_output_tokens") != config.max_output_tokens:
            raise ValueError("При продолжении нельзя менять модель или размер ответа")
    result.mkdir(exist_ok=resume)
    sessions = manifest.get('processing_sessions', []) + [{
        'started_at': now(), 'resume': resume,
        'configuration': config.public() if config else {'provider': 'rules', 'model': None}}]
    update_manifest(directory, status="processing", method=method, result_dir=method,
                    processing_sessions=sessions,
                    configuration=config.public() if config else {"provider": "rules", "model": None})
    try:
        records = normalize(directory / "raw_books.json", result / "normalized_books.json",
                            enrich=enrich, config=config, progress=progress, resume=resume,
                            experiment_id=manifest.get('experiment_id', directory.name), run_id=directory.name,
                            ledger=directory.parent.parent / "gemini_cache.sqlite")
        metrics = generate_report(directory / "raw_books.json", result / "normalized_books.json", result)
        write_templates(records, result)
        write_json(result / "evaluation.json", evaluate(records))
        status = "completed_with_errors" if (metrics["llm_fallbacks"] or metrics["invalid_raw_records"]
            or read_json(directory / "manifest.json").get("scrape_error_count")) else "completed"
        update_manifest(directory, status=status, completed_at=now(), metrics=metrics,
                        result_sha256=file_hash(result / "knowledge_source.json"))
    except ProcessingPaused as exc:
        # Export completed records while keeping the run paused.
        metrics = generate_report(directory / "raw_books.json", result / "normalized_books.json", result)
        records = read_json(result / "normalized_books.json")
        write_templates(records, result)
        update_manifest(directory, status="processing_paused", pause_reason=str(exc), metrics=metrics,
                        result_sha256=file_hash(result / "knowledge_source.json"))
    except BaseException as exc:
        update_manifest(directory, status="processing_interrupted" if isinstance(exc, KeyboardInterrupt) else "processing_failed",
                        error_type=type(exc).__name__)
        raise
    return result


def evaluate_run(directory: Path, gold_path: Path | None = None, reviews_path: Path | None = None) -> dict:
    manifest = verify_raw(directory)
    result = directory / manifest["result_dir"]
    if file_hash(result / "knowledge_source.json") != manifest.get("result_sha256"):
        raise ValueError("Нормализованный результат изменён")
    gold_path = gold_path or result / "gold_labels.csv"
    reviews_path = reviews_path or result / "summary_reviews.csv"
    metrics = evaluate(read_json(result / "knowledge_source.json"), read_labels(gold_path), read_labels(reviews_path))
    facts_reviews = result / 'fact_reviews.csv'
    metrics.update(evaluate_facts(read_labels(result / 'facts_triples.csv'), read_labels(facts_reviews) if facts_reviews.exists() else []))
    write_json(result / "evaluation.json", {**metrics, "evaluated_at": now(),
        "fact_reviews_sha256": file_hash(facts_reviews) if facts_reviews.exists() else None,
        "gold_sha256": file_hash(gold_path), "reviews_sha256": file_hash(reviews_path)})
    return metrics
