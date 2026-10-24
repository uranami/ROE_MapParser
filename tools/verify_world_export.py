"""Recheck full-world exports against the original local game assets."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roe_map_parser.assets import GameAssets, Sfc, path_hash, sha256
from roe_map_parser.world_instances import load_tables
from roe_map_parser.transforms import instance_matrix
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--game-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
    checked, containers = 0, {}
    try:
        with GameAssets(args.game_root) as game:
            for source in summary['inputs']:
                if 'record_offset' in source:
                    if source['archive'] not in containers:
                        containers[source['archive']] = Sfc(source['archive'])
                    archive = containers[source['archive']]
                    assert archive.index[path_hash(source['resource'])] == source['record_offset']
                    raw = archive.read(source['resource'])
                else:
                    raw, _ = game.read(source['resource'])
                assert len(raw) == source['bytes'] and hashlib.sha256(raw).hexdigest() == source['sha256']
                checked += 1
            terrain = json.loads((out/'terrain/summary.json').read_text())
            with gzip.open(out/f"terrain/terrain-surface{summary['map']}.bin.gz", 'rb') as stream:
                assert struct.unpack('<8s4I', stream.read(24)) == (b'ROEHGT1\0', terrain['peak'], 200, 257, 16)
                for y in range(16):
                    for x in range(16):
                        raw, _ = game.read(f"DATA/WORLD/{summary['world']}/TERRAINHEIGHTTILE_{x}_{y}.QSHM")
                        assert stream.read(257*257*2) == raw[12:]
                assert not stream.read(1)
    finally:
        for archive in containers.values():
            archive.close()
    entries = json.loads((out/'manifest.json').read_text())
    for entry in entries:
        path = (out/entry['path']).resolve()
        assert path.is_relative_to(out)
        assert path.stat().st_size == entry['bytes'] and sha256(path) == entry['sha256']
    tables, _ = load_tables(args.game_root.resolve(), summary['world'])
    catalog = json.loads((out/'models.json').read_text())
    unsupported = json.loads((out/'unsupported.json').read_text())
    assert len(unsupported) == summary['unsupported_models']
    count = Counter()
    for kind in ('static', 'tree'):
        with (out/(kind+'_instances.jsonl')).open(encoding='utf-8') as f:
            for record, original in zip(map(json.loads, f), tables[kind]['rows'], strict=True):
                assert record['original'] == original
                model = catalog[record['model_id']]
                assert model['resource'] == original['mModelFullPath'] and record['model_status'] == model['status']
                np.testing.assert_allclose(record['model_to_world_rows'], instance_matrix(original, kind), rtol=0, atol=0)
                count[record['model_id']] += 1
    for model in catalog:
        assert count[model['model_id']] == model['instances']
        if 'file' in model:
            data = json.loads((out/model['file']).read_text(encoding='utf-8'))
            assert data['model_id'] == model['model_id'] and data['source']['sha256'] == model['source']['sha256']
    assert sum(count.values()) == summary['total_instances']
    report = {'status': 'passed', 'original_source_hashes': checked, 'exact_qshm_tiles': 256,
              'instances_and_transforms': sum(count.values()), 'models': len(catalog),
              'unsupported_models': len(unsupported), 'output_hashes': len(entries)}
    # Kept outside the export manifest, so verification does not mutate its inputs.
    (out.parent/(out.name+'_verification.json')).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
