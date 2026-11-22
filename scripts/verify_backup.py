import argparse
from hashlib import sha256
import json
from pathlib import Path
import zipfile


def verify(backup, root=None):
    inventory=json.loads((backup/'inventory.json').read_text(encoding='utf-8'))
    archive=backup/'experiment.zip'
    if sha256(archive.read_bytes()).hexdigest()!=inventory['archive_sha256']:
        raise ValueError('Контрольная сумма архива не совпала')
    cache_inventory = backup/'cache_inventory.json'
    if cache_inventory.exists():
        cache = json.loads(cache_inventory.read_text(encoding='utf-8'))
        if sha256((backup/'shared-cache.sqlite').read_bytes()).hexdigest() != cache['sha256']:
            raise ValueError('Контрольная сумма резервной копии кеша не совпала')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Архив повреждён')
        for item in inventory['files']:
            if sha256(z.read(item['path'])).hexdigest()!=item['sha256']:
                raise ValueError('Файл в копии изменён: '+item['path'])
            if root is not None:
                path=(root/item['path']).resolve()
                if not path.is_relative_to(root.resolve()) or sha256(path.read_bytes()).hexdigest()!=item['sha256']:
                    raise ValueError('Исходный файл изменён: '+item['path'])
    return len(inventory['files'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Проверка резервной копии без распаковки и изменения файлов')
    parser.add_argument('backup',type=Path)
    parser.add_argument('--root',type=Path,help='Дополнительно сравнить исходные файлы в корне проекта')
    args=parser.parse_args()
    print(f'Проверено файлов: {verify(args.backup,args.root)}. Все SHA-256 совпадают.')
