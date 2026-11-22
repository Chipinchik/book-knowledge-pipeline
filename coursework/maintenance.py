from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import zipfile

from .storage import file_hash, read_json, write_json

LEGACY_FILES = ('raw_books.json', 'normalized_books.json', 'knowledge_source.csv', 'knowledge_source.json',
                'knowledge_source.sqlite', 'facts_triples.csv', 'metrics.json', 'completeness.png',
                'price_distribution.png', 'topics.png', 'html_snapshots', 'latest.json')


def archive_development(output: Path):
    root = output.resolve(strict=True)
    candidates = [p for p in (root / 'runs').iterdir() if p.is_dir()] if (root / 'runs').exists() else []
    linked = {rid for p in (root / 'experiments').glob('*/experiment.json')
              for rid in read_json(p).get('runs', {}).values()}
    # Preserve linked runs and final experiments.
    candidates = [p for p in candidates if p.name not in linked and (not (p / 'manifest.json').exists()
                  or read_json(p / 'manifest.json').get('purpose') != 'final')]
    candidates += [root / name for name in LEGACY_FILES if (root / name).exists()]
    latest = root / 'latest.json'
    if latest in candidates and (root / 'runs' / read_json(latest).get('run_id', '')) not in candidates:
        candidates.remove(latest)
    candidates += [p for p in (root / 'verification',) if p.exists()]
    if not candidates:
        return None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target = root / 'archive' / stamp
    target.mkdir(parents=True)
    files = []
    for source in candidates:
        resolved = source.resolve(strict=True)
        if not resolved.is_relative_to(root) or resolved == root or source.is_symlink():
            raise ValueError('Archive target escapes the outputs workspace')
        for path in source.rglob('*') if source.is_dir() else [source]:
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError('Archive refuses links outside the workspace')
            if path.is_file():
                files.append({'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size, 'sha256': file_hash(path)})
    inventory = {'created_at': datetime.now(timezone.utc).isoformat(), 'reason': 'Предфинальная очистка результатов разработки',
                 'status': 'planned', 'items': [p.relative_to(root).as_posix() for p in candidates], 'files': files,
                 'preserved_active': ['gemini_cache.sqlite', 'experiments/ (если существуют)']}
    write_json(target / 'inventory.json', inventory)
    # Verify the backup before moving active files.
    with zipfile.ZipFile(target / 'backup.zip', 'x', compression=zipfile.ZIP_DEFLATED) as backup:
        for item in files:
            backup.write(root / item['path'], item['path'])
    from hashlib import sha256
    with zipfile.ZipFile(target / 'backup.zip') as backup:
        for item in files:
            if sha256(backup.read(item['path'])).hexdigest() != item['sha256']:
                raise ValueError('Backup verification failed; active files were not moved')
    inventory['status'] = 'backup_verified'
    write_json(target / 'inventory.json', inventory)
    moved = target / 'files'
    for source in candidates:
        destination = moved / source.relative_to(root)
        # Recheck path containment before moving directories.
        if not source.resolve().is_relative_to(root) or not destination.resolve().is_relative_to(target.resolve()):
            raise ValueError('Unsafe archive move')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    inventory['status'] = 'archived'
    write_json(target / 'inventory.json', inventory)
    return target


if __name__ == '__main__':
    from .pipeline import ROOT
    target = archive_development(ROOT / 'outputs')
    print(json.dumps({'archive': str(target) if target else None}, ensure_ascii=True))
