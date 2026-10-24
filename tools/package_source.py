"""Audit and package source-only files. Never include game or generated data."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '.venv', '__pycache__', '.pytest_cache', 'output', 'dist', 'build'}
ROOT_FILES = {'README.md', 'pyproject.toml', 'MANIFEST.in', '.gitignore', '.gitattributes'}
BANNED_IMPORTS = {'matplotlib', 'scipy', 'flask', 'fastapi', 'http', 'socket', 'webbrowser',
                  'roe_map_tools', 'roe_server', 'uvicorn', 'tkinter'}
SENSITIVE = [re.compile(r'(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{20,}'),
             re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
             re.compile(r'[A-Za-z]:[\\/](?:Users[\\/]|SteamLibrary[\\/])', re.I)]


def source_files(root=ROOT):
    result = []
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if any(p in SKIP or p.endswith('.egg-info') for p in rel.parts):
            continue
        if path.is_dir():
            continue
        allowed = (rel.as_posix() in ROOT_FILES or
                   rel.parts[0] in {'roe_map_parser', 'tests', 'tools'} and path.suffix == '.py' or
                   len(rel.parts) == 3 and rel.parts[:2] == ('.github', 'workflows') and path.suffix == '.yml')
        if path.is_symlink() or not allowed:
            raise ValueError(f'not in source whitelist: {rel}')
        text = path.read_text(encoding='utf-8')
        if any(pattern.search(text) for pattern in SENSITIVE):
            raise ValueError(f'possible credential/local absolute path: {rel}')
        if path.suffix == '.py':
            for node in ast.walk(ast.parse(text, filename=str(rel))):
                names = ([n.name.split('.')[0] for n in node.names] if isinstance(node, ast.Import)
                         else [(node.module or '').split('.')[0]] if isinstance(node, ast.ImportFrom) and not node.level
                         else [])
                for name in names:
                    if name in BANNED_IMPORTS:
                        raise ValueError(f'viewer/server dependency {name}: {rel}')
                    if name and name not in sys.stdlib_module_names and name not in {'numpy', 'lz4', 'roe_map_parser'}:
                        raise ValueError(f'unexpected dependency {name}: {rel}')
        result.append(path)
    config = tomllib.loads((root/'pyproject.toml').read_text())
    if config['project']['dependencies'] != ['numpy>=1.24', 'lz4>=4.3']:
        raise ValueError('unexpected package dependencies')
    if not (root/'roe_map_parser/cli.py').is_file():
        raise ValueError('missing CLI')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--out', type=Path, default=ROOT/'dist/roe-map-parser-source.zip')
    args = ap.parse_args()
    paths = source_files()
    report = {'status': 'passed', 'files': len(paths), 'source_bytes': sum(p.stat().st_size for p in paths)}
    if not args.check:
        out = args.out.resolve()
        if out.suffix.lower() != '.zip' or not out.is_relative_to(ROOT/'dist'):
            raise ValueError('source archives must be ZIP files under this project dist/')
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, 'roe-map-parser/'+path.relative_to(ROOT).as_posix())
        report.update(archive=str(out), archive_bytes=out.stat().st_size,
                      sha256=hashlib.sha256(out.read_bytes()).hexdigest())
        (out.with_suffix('.zip.sha256')).write_text(report['sha256']+'  '+out.name+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
