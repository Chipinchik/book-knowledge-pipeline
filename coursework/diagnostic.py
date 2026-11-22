from dataclasses import replace
from pathlib import Path
from .llm_provider import ProviderConfig
from .pipeline import ROOT, execute
from .storage import read_json, write_json


def main():
    output = ROOT / "outputs"
    marker = output / "verification" / "gemini_diagnostic.json"
    if marker.exists() or any((output / 'archive').glob('*/files/verification/gemini_diagnostic.json')):
        print("Diagnostic already recorded; no additional API request made.")
        return
    config = replace(ProviderConfig.from_env(), max_requests=1, retries=0)
    config.validate()
    sources = []
    for path in (output / "runs").glob("*/manifest.json"):
        m = read_json(path)
        if m.get("source") == "live" and m.get("raw_sha256") and m.get("raw_records", 0) >= 1:
            sources.append(path.parent / "raw_books.json")
    if not sources:
        raise ValueError("No saved live snapshot for diagnostic")
    write_json(marker, {"status": "started", "maximum_requests": 1, "model": config.model})
    try:
        directory = execute(output, raw_path=sorted(sources)[-1], process_limit=1,
                            enrich="gemini", config=config, api_confirmed=True)
        manifest = read_json(directory / "manifest.json")
        write_json(marker, {"status": manifest["status"], "run_id": directory.name,
            "model": config.model, "pause_reason": manifest.get("pause_reason"), "metrics": manifest.get("metrics")})
        print("Diagnostic:", manifest["status"], "successful records:", manifest.get("metrics", {}).get("gemini_processed", 0))
        if manifest.get("pause_reason"):
            print("Safe error code:", manifest["pause_reason"])
    except Exception as exc:
        write_json(marker, {"status": "failed", "error_type": type(exc).__name__, "maximum_requests": 1})
        print("Diagnostic failed; credentials and response bodies withheld.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
