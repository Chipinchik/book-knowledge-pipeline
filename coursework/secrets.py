from pathlib import Path
import argparse
import json
import os
import re
import subprocess

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_-]{35}")


def redact(value, secret=""):
    if isinstance(value, str):
        if secret:
            value = value.replace(secret, "[REDACTED]")
        return KEY_PATTERN.sub("[REDACTED]", value)
    if isinstance(value, dict):
        return {k: redact(v, secret) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secret) for v in value]
    return value


def migrate(root=ROOT):
    source, target = root / "GeminiApiKey.txt", root / ".env"
    text = source.read_text(encoding="utf-8-sig").strip()
    keys = {text} if re.fullmatch(r"[A-Za-z0-9_./+=-]{20,256}", text) else set(KEY_PATTERN.findall(text))
    if len(keys) != 1:
        raise ValueError("Файл должен содержать ровно один ключ Google API")
    key = keys.pop()
    existing = target.read_text(encoding="utf-8") if target.exists() else ""
    configured = dotenv_values(target).get("GEMINI_API_KEY") if target.exists() else None
    if configured and configured != key:
        raise ValueError("В .env уже другой ключ; автоматическая замена запрещена")
    lines = [line for line in existing.splitlines() if not re.match(r"\s*GEMINI_API_KEY\s*=", line)]
    lines.append("GEMINI_API_KEY=" + key)
    defaults = {"GEMINI_MODEL": "gemini-3.1-flash-lite", "GEMINI_TIMEOUT": "60",
                "GEMINI_DELAY": "7", "GEMINI_MAX_REQUESTS": "5", "GEMINI_DAILY_LIMIT": "20",
                "GEMINI_RETRIES": "0", "GEMINI_MAX_OUTPUT_TOKENS": "512",
                "GEMINI_FREE_TIER_CONFIRMED": "false"}
    present = dotenv_values(target) if target.exists() else {}
    lines.extend(f"{k}={v}" for k, v in defaults.items() if k not in present)
    temporary = target.with_name(".env.migrating")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    assert dotenv_values(target).get("GEMINI_API_KEY") == key
    return {"migration": "verified", "source_file_required": False, "source_file_preserved": True}


def audit(root=ROOT):
    values = dotenv_values(root / ".env")
    secret = values.get("GEMINI_API_KEY") or ""
    matches = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git", ".venv", "__pycache__", ".pytest_cache")]
        for filename in files:
            path = Path(base) / filename
            if path in (root / ".env", root / "GeminiApiKey.txt"):
                continue
            data = path.read_bytes()
            if (secret and secret.encode() in data) or KEY_PATTERN.search(data.decode("utf-8", errors="ignore")):
                matches.append(str(path.relative_to(root)))
    probe = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root, capture_output=True)
    git = {"repository_present": probe.returncode == 0, "tracked_secret_file": None, "secret_path_in_history": None}
    if git["repository_present"]:
        tracked = subprocess.run(["git", "ls-files", "--", "GeminiApiKey.txt", ".env"], cwd=root, capture_output=True)
        history = subprocess.run(["git", "log", "--all", "--format=%H", "--", "GeminiApiKey.txt", ".env"], cwd=root, capture_output=True)
        git.update(tracked_secret_file=bool(tracked.stdout.strip()), secret_path_in_history=bool(history.stdout.strip()))
    return {"key_configured": bool(secret), "unexpected_secret_files": matches, "git": git}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("migrate", "audit"))
    args = parser.parse_args()
    try:
        print(json.dumps(migrate() if args.action == "migrate" else audit(), ensure_ascii=True, indent=2))
    except Exception:
        print("Secret operation failed; credential contents withheld.")
        raise SystemExit(1) from None
