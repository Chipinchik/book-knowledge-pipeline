from dataclasses import replace
from pathlib import Path
import argparse
import json
import sys

from .llm_provider import ProviderConfig, ProviderError
from .pipeline import ROOT, evaluate_run, execute, latest_run, process_run, verify_raw
from .report import generate_report
from .storage import read_json, file_hash


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Первичный источник знаний: Rules + Google Gemini")
    parser.add_argument("stage", choices=("scrape", "normalize", "all", "report", "evaluate", "resume-processing"))
    parser.add_argument("--source", choices=("demo", "live"), default="demo")
    parser.add_argument("--enrich", choices=("rules", "gemini"), default="rules")
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--process-limit", type=int, help="Первые N исходных записей для обоих методов")
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--api-delay", type=float)
    parser.add_argument("--max-requests", type=int, help="Общий лимит HTTP-попыток этого запуска, включая повторы")
    parser.add_argument("--confirm-gemini", action="store_true", help="Разрешить реальные запросы Google API")
    parser.add_argument("--confirm-free-tier", action="store_true", help="Я проверил Free tier и отсутствие биллинга")
    parser.add_argument("--retry-failed", action="store_true", help="Явно повторить ранее отклонённые запросы 429/503")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--resume", type=Path, help="Продолжить сбор; для обработки используйте resume-processing")
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()
    if args.raw and args.stage not in ("normalize", "report"):
        parser.error("--raw допустим только для normalize/report")
    if args.resume and args.stage not in ("scrape", "all"):
        parser.error("--resume допустим только для scrape/all")
    try:
        selected = args.run or latest_run(args.out)
        method = args.enrich
        if args.stage == "resume-processing":
            if selected is None:
                raise ValueError("Укажите --run")
            saved = read_json(selected / "manifest.json")
            method = saved["method"]
            args.model = saved.get("configuration", {}).get("model")
        config = None
        if method == "gemini" and args.stage in ("all", "normalize", "resume-processing"):
            config = ProviderConfig.from_env(args.model)
            config = replace(config, max_requests=args.max_requests if args.max_requests is not None else config.max_requests,
                delay=args.api_delay if args.api_delay is not None else config.delay,
                free_tier_confirmed=args.confirm_free_tier or config.free_tier_confirmed, retry_failed=args.retry_failed)
        if args.stage in ("scrape", "all", "normalize"):
            raw = args.raw
            if args.stage == "normalize" and raw is None:
                raw = selected / "raw_books.json" if selected else args.out / "raw_books.json"
            directory = execute(args.out, source=args.source, limit=args.limit, delay=args.delay,
                enrich=method, config=config, raw_path=raw, scrape_only=args.stage == "scrape",
                resume=args.resume, process_limit=args.process_limit, api_confirmed=args.confirm_gemini)
            manifest = read_json(directory / "manifest.json")
            print(json.dumps({"status": manifest["status"], "metrics": manifest.get("metrics"),
                              "pause_reason": manifest.get("pause_reason")}, ensure_ascii=False, indent=2))
            print(f"Результаты: {directory}")
        elif args.stage == "resume-processing":
            result = process_run(selected, enrich=method, config=config, resume=True,
                                 api_confirmed=args.confirm_gemini, progress=print)
            print(f"Результаты: {result}")
        elif args.stage == "evaluate":
            if selected is None:
                raise ValueError("Укажите --run")
            print(json.dumps(evaluate_run(selected, args.gold, args.annotations), ensure_ascii=False, indent=2))
        elif selected:
            manifest = verify_raw(selected)
            if args.raw and file_hash(args.raw) != manifest["raw_sha256"]:
                raise ValueError("Снимки raw не совпадают")
            print(json.dumps(read_json(selected / manifest["result_dir"] / "metrics.json"), ensure_ascii=False, indent=2))
        else:
            print(json.dumps(generate_report(args.raw or args.out / "raw_books.json",
                args.out / "normalized_books.json", args.out), ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        print("Остановлено. Данные сохранены; продолжите сбор или обработку существующего запуска.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        message = str(exc) if isinstance(exc, (ValueError, ProviderError, FileExistsError)) else type(exc).__name__
        print(f"Ошибка: {message}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
