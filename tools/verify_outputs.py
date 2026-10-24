"""Independent checks of terrain, layout and collision exports from game assets."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np

from roe_map_parser.assets import GameAssets, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--game-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    base = args.output.resolve()
    source_checks = 0
    summaries = {}
    with GameAssets(args.game_root) as game:
        for name in ['europa_terrain', 'europa_layouts', 'dione_rigid', 'building_collision', 'rock_collision']:
            out = base / name
            summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
            assert summary['source'] == 'game_installation_only'
            summaries[name] = summary
            for source in summary['inputs']:
                raw, meta = game.read(source['resource'])
                assert meta['sha256'] == source['sha256'] and len(raw) == source['bytes']
                source_checks += 1
            for entry in json.loads((out / 'manifest.json').read_text()):
                assert sha256(out / entry['path']) == entry['sha256']
        terrain = summaries['europa_terrain']
        with gzip.open(base / 'europa_terrain/terrain-surface16014.bin.gz', 'rb') as f:
            assert struct.unpack('<8s4I', f.read(24)) == (b'ROEHGT1\0', terrain['peak'], 200, 257, 16)
            for ty in range(16):
                for tx in range(16):
                    raw, _ = game.read(fr'DATA\WORLD\EU_KILLZONE_4096\TERRAINHEIGHTTILE_{tx}_{ty}.QSHM')
                    assert len(raw) == 12 + 257 * 257 * 2
                    assert f.read(257 * 257 * 2) == raw[12:]
            assert f.read(1) == b''
    vertices, triangles = [], []
    with (base / 'europa_terrain/terrain.obj').open() as f:
        for line in f:
            values = line.split()
            if values[0] == 'v':
                vertices.append([float(n) for n in values[1:]])
            elif values[0] == 'f':
                triangles.append([int(n) - 1 for n in values[1:]])
    v, faces = np.asarray(vertices), np.asarray(triangles)
    assert np.isfinite(v).all() and faces.min() >= 0 and faces.max() < len(v)
    assert len(v) == terrain['mesh']['vertices'] and len(faces) == terrain['mesh']['triangles']
    points = v[faces]
    assert (np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])[:, 2] > 0).all()
    assert (v[:, :2].min(axis=0) == [0, 0]).all() and (v[:, :2].max(axis=0) == [819200, 819200]).all()
    # Count the original collision elements, not the parser's serialized counts.
    root = ET.fromstring((base / 'building_collision/source.xml').read_bytes())
    counts = {kind: len(root.findall(f'.//mActors/mActors/{tag}/{tag}')) for tag, kind in
              [('mBoxes', 'box'), ('mSpheres', 'sphere'), ('mCapsules', 'capsule'), ('mConvexes', 'convex_points')]}
    counts = {k: v for k, v in counts.items() if v}
    assert counts == summaries['building_collision']['counts']
    model = json.loads((base / 'building_collision/collision.json').read_text())
    assert Counter(s['kind'] for a in model['actors'] for s in a['shapes']) == counts
    repx = ET.fromstring((base / 'rock_collision/source.xml').read_bytes())
    model = json.loads((base / 'rock_collision/collision.json').read_text())
    for source, result in zip(repx.findall('PxTriangleMesh'), model['triangle_meshes'], strict=True):
        np.testing.assert_array_equal(np.fromstring(source.findtext('Points'), sep=' ').reshape(-1, 3), result['vertices'])
        np.testing.assert_array_equal(np.fromstring(source.findtext('Triangles'), sep=' ', dtype=int).reshape(-1, 3), result['triangles'])
    layouts = json.loads((base / 'europa_layouts/layout_instances.json').read_text())
    definitions = json.loads((base / 'europa_layouts/layout_definitions.json').read_text())
    unresolved = 0
    for instance in layouts['instances']:
        nodes = {n['attributes']['mName'].lower() for n in definitions[instance['attributes']['mParentName'].lower()]['nodes']}
        unresolved += sum(l['mNodeName'].rsplit('.', 1)[-1].lower() not in nodes for l in instance['entity_links'])
    assert unresolved == len(summaries['europa_layouts']['unresolved_node_links'])
    report = {'status': 'PASS', 'source_records_rechecked': source_checks, 'qshm_payloads_byte_identical': 256,
              'terrain_vertices': len(v), 'terrain_triangles': len(faces), 'original_collision_counts': counts,
              'unresolved_prefab_links_retained': unresolved,
              'checks': ['original resource SHA-256', 'output manifests', 'all raw QSHM payloads',
                         'OBJ coordinates/indices/winding/extent', 'original physical XML shape counts',
                         'RepX source mesh coordinates/indices', 'explicit unresolved prefab references']}
    (base / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
