from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import os
import re
import shutil

from .llm_provider import ProviderConfig
from .pipeline import execute, verify_raw
from .storage import file_hash, read_json, write_json


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError('Операция уже выполняется. После аварийной остановки проверьте процесс перед удалением файла .lock.') from None
    try:
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def list_experiments(output):
    result = []
    for path in sorted((output / 'experiments').glob('*/experiment.json'), reverse=True):
        value = read_json(path)
        result.append(value)
    return result


def experiment_path(output, identifier):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', identifier):
        raise ValueError('Некорректный идентификатор эксперимента')
    return output / 'experiments' / identifier


def load_experiment(output, identifier):
    directory = experiment_path(output, identifier)
    value = read_json(directory / 'experiment.json')
    if file_hash(directory / 'raw_books.json') != value['raw_sha256']:
        raise ValueError('Исходный снимок эксперимента изменён')
    return value


def prepare_experiment(output: Path, raw_path: Path, count: int, config: ProviderConfig,
                       purpose='final', title='Сравнение Rules и Gemini', *, allow_new=False):
    if purpose not in ('final', 'technical'):
        raise ValueError('Неизвестный тип эксперимента')
    config.validate_parameters()
    if purpose == 'technical' and count > 2:
        raise ValueError('Техническая проверка ограничена 1–2 записями')
    rows = read_json(raw_path)
    if not isinstance(rows, list) or not 1 <= count <= min(30, len(rows)):
        raise ValueError('Нужно выбрать от 1 до 30 существующих исходных записей')
    if purpose == 'final' and (count < 15 or any(not isinstance(r, dict) or r.get('source_mode') != 'live' for r in rows[:count])):
        raise ValueError('Итоговый эксперимент требует 15–30 реальных live-записей')
    if (raw_path.parent / 'manifest.json').exists():
        verify_raw(raw_path.parent)
    with exclusive(output / 'experiments' / '.prepare.lock'):
        if purpose == 'final' and not allow_new and any(e.get('purpose') == 'final' for e in list_experiments(output)):
            raise ValueError('Итоговый эксперимент уже подготовлен. Используйте его; повторный итоговый набор не создаётся.')
        identifier = 'exp-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid4().hex[:6]
        directory = experiment_path(output, identifier)
        directory.mkdir(parents=True)
        if count == len(rows):
            shutil.copyfile(raw_path, directory / 'raw_books.json')
        else:
            write_json(directory / 'raw_books.json', rows[:count])
        if (raw_path.parent / 'html_snapshots').exists():
            shutil.copytree(raw_path.parent / 'html_snapshots', directory / 'html_snapshots')
        document = {'experiment_id': identifier, 'title': title, 'purpose': purpose,
                    'created_at': datetime.now(timezone.utc).isoformat(), 'raw_sha256': file_hash(directory / 'raw_books.json'),
                    'parent_raw_sha256': file_hash(raw_path), 'raw_records': count, 'configuration': config.public(),
                    'runs': {}, 'source': ','.join(sorted({r['source_mode'] for r in rows[:count]}))}
        write_json(directory / 'experiment.json', document)
        return document


def run_method(output, identifier, method, *, api_confirmed=False, config=None, progress=None):
    if method not in ('rules', 'gemini'):
        raise ValueError('Допустимы Rules и Gemini')
    directory = experiment_path(output, identifier)
    with exclusive(directory / '.launch.lock'):
        experiment = load_experiment(output, identifier)
        if method in experiment['runs']:
            return output / 'runs' / experiment['runs'][method]
        if method == 'gemini':
            config = config or ProviderConfig.from_env()
            frozen = experiment['configuration']
            config = replace(config, **{k: frozen[k] for k in
                ('model', 'timeout', 'delay', 'max_requests', 'daily_limit', 'retries', 'max_output_tokens')})
        else:
            config = None

        def bind(run):
            experiment['runs'][method] = run.name
            write_json(directory / 'experiment.json', experiment)

        return execute(output, raw_path=directory / 'raw_books.json', enrich=method,
            config=config, api_confirmed=api_confirmed, progress=progress,
            experiment_id=identifier, purpose=experiment['purpose'], on_created=bind)
