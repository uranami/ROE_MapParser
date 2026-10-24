"""Headless full-world export: original terrain, model shapes and final instances."""
from collections import Counter
import json
from pathlib import Path

from . import formats as fmt
from .common import WORLDS, dump, manifest
from .transforms import instance_matrix
from .world_instances import load_tables, intersects


def export_tables(root, world, out, scope='ClientTable', roi=None):
    tables, sources = load_tables(root, world, scope)
    summary = {'source': 'game_installation_only', 'world': world, 'scope': scope,
               'inputs': sources, 'roi_xy': roi, 'tables': {}}
    for kind, table in tables.items():
        selected = [r for r in table['rows'] if intersects(r, roi)]
        with (out/(kind+'_instances.jsonl')).open('w', encoding='utf-8') as f:
            for row in selected:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        summary['tables'][kind] = {'class': table['class'], 'name': table['name'], 'flags': table['flags'],
                                  'total_instances': len(table['rows']), 'selected_instances': len(selected),
                                  'selected_model_paths': len({r['mModelFullPath'] for r in selected})}
    summary['limitations'] = ['Authored bounds are broadphase only, not collision geometry.',
                              'Raw flags/enums do not imply collision or visibility policies.']
    return summary


def parse_model(raw):
    root = fmt.xml_root(raw)
    if root.tag == 'PhysX30Collection':
        parsed = fmt.repx(raw)
        try:
            parsed['resolved_static_shapes'] = fmt.identity_repx_shapes(raw)
            return parsed, None
        except ValueError as error:
            # Raw data is still exported; unsupported poses are never applied.
            return parsed, str(error)
    parsed = fmt.phys_collection(raw)
    if any(a['attributes'].get('mUsage') != 'PhysActorDesc::eSTATIC' for a in parsed['actors']):
        return parsed, 'nonstatic actor: raw model exported without static resolution'
    return parsed, None


def export_world(game, args, out, export_terrain):
    if any(out.iterdir()):
        raise ValueError('world export requires a new/empty output directory to avoid stale assets')
    world, map_id = WORLDS[args.world]
    tables, table_sources = load_tables(game.root, world)
    paths = sorted({r['mModelFullPath'] for table in tables.values() for r in table['rows']})
    ids = {path: i for i, path in enumerate(paths)}
    counts = Counter(ids[r['mModelFullPath']] for t in tables.values() for r in t['rows'])
    model_dir = out/'models'
    model_dir.mkdir()
    catalog, unsupported, inputs = [], [], list(table_sources)
    for mid, path in enumerate(paths):
        record = {'model_id': mid, 'resource': path, 'instances': counts[mid]}
        try:
            raw, source = game.read(path)
            inputs.append(source)
            parsed, limitation = parse_model(raw)
            filename = f'models/{mid:06d}.json'
            dump(out/filename, {'model_id': mid, 'resource': path, 'source': source, 'data': parsed})
            record.update(file=filename, format=parsed['format'], source=source,
                          status='unsupported_static_resolution' if limitation else 'resolved')
            if limitation:
                record['reason'] = limitation
                unsupported.append(dict(record))
        except (ValueError, OSError, KeyError, fmt.ET.ParseError) as error:
            record.update(status='unparsed', reason=str(error))
            unsupported.append(dict(record))
        catalog.append(record)
    failed = {r['model_id'] for r in unsupported}
    for kind, table in tables.items():
        with (out/(kind+'_instances.jsonl')).open('w', encoding='utf-8') as f:
            for row in table['rows']:
                mid = ids[row['mModelFullPath']]
                record = {'model_id': mid, 'model_status': catalog[mid]['status'], 'original': row,
                          'model_to_world_rows': instance_matrix(row, kind).tolist()}
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
    terrain_dir = out/'terrain'
    terrain_dir.mkdir()
    terrain = export_terrain(game, args, terrain_dir)
    dump(terrain_dir/'summary.json', terrain)
    manifest(terrain_dir)
    inputs.extend(terrain['inputs'])
    dump(out/'models.json', catalog)
    dump(out/'unsupported.json', unsupported)
    total = sum(counts.values())
    return {'source': 'game_installation_only', 'world': world, 'map': map_id,
            'status': 'partial' if unsupported else 'resolved_static_resources',
            'inputs': inputs, 'terrain_tiles': terrain['tiles'],
            'static_instances': len(tables['static']['rows']), 'tree_instances': len(tables['tree']['rows']),
            'total_instances': total, 'models': len(paths), 'resolved_models': len(paths)-len(failed),
            'resolved_instances': total-sum(counts[i] for i in failed),
            'unsupported_models': len(failed), 'unsupported_instances': sum(counts[i] for i in failed),
            'coordinate_policy': 'Original XYZ; Z up. world = model_to_world @ shape_local @ vertex.',
            'limitations': ['Collision primitives/point clouds and source triangles only; no viewer or visual meshes.',
                            'Unsupported/malformed resources are listed, never repaired or substituted.',
                            'Dynamic/destructible state, filter semantics and native-client alignment remain unverified.',
                            'Terrain export diagonal and welded seam ownership are explicit export policies.',
                            'No navmesh, pathfinding or BOT runtime asset is generated.']}
