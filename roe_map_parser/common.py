"""Shared export paths and provenance records."""
import json
from pathlib import Path

from .assets import sha256

WORLDS = {'europa': ('EU_KILLZONE_4096', 16014), 'dione': ('EU_ICELAND_4096', 16010)}


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def safe_output(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out == root or any(out.is_relative_to(root/p) for p in ('SFC', 'Cache', 'XVersion', 'Data', 'data_bas_0')):
        raise ValueError('output must be separate from original game resources')
    out.mkdir(parents=True, exist_ok=True)
    return out


def manifest(out):
    dump(out/'manifest.json', [{'path': p.relative_to(out).as_posix(), 'bytes': p.stat().st_size,
                               'sha256': sha256(p)} for p in sorted(out.rglob('*'))
                              if p.is_file() and p != out/'manifest.json'])
