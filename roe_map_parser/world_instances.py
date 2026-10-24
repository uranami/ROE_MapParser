"""Read shipped world-cache static/tree collision instance tables.

The reflection schemas are registered by XVersion 140832AE0 and 140833240.
These are original client cache records, not third-party server exports.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from .assets import Sfc, path_hash, sha256
from .common import WORLDS, dump, safe_output


class BinaryReader:
    def __init__(self, raw):
        self.raw, self.offset = raw, 0

    def take(self, count):
        if count < 0 or self.offset + count > len(self.raw):
            raise ValueError(f'truncated world table at {self.offset}')
        value = self.raw[self.offset:self.offset+count]
        self.offset += count
        return value

    def u8(self): return self.take(1)[0]
    def u16(self): return struct.unpack('<H',self.take(2))[0]
    def u32(self): return struct.unpack('<I',self.take(4))[0]
    def f32(self):
        value = struct.unpack('<f',self.take(4))[0]
        if not np.isfinite(value): raise ValueError('nonfinite table float')
        return value

    def expect(self, value):
        if self.take(len(value)) != value:
            raise ValueError(f'world table schema mismatch at {self.offset-len(value)}')

    def text(self):
        # Observed interned-string length encoding: C0..FF for 0..63;
        # 80..BF followed by a high byte for longer strings. Unobserved tags
        # and larger encodings are rejected, never searched around or patched.
        first = self.u8()
        if first < 128: raise ValueError('unsupported interned-string tag')
        size = first & 63
        if not first & 64:
            high = self.u8()
            if high >= 128: raise ValueError('unsupported large string encoding')
            size += high << 6
            if size < 64: raise ValueError('noncanonical long string')
        value = self.take(size).decode('utf-8')
        if '\0' in value: raise ValueError('embedded NUL in table string')
        return value

    def type_name(self):
        return self.take(self.u8()).decode('ascii')


STATIC_FLOATS = ('mPosX','mPosY','mPosZ','mScale','mRotateX','mRotateY','mRotateZ',
                 'mBoundingX','mBoundingY','mBoundingZ','mBoundingRadius')
TREE_FLOATS = ('mPosX','mPosY','mPosZ','mScale','mQuatX','mQuatY','mQuatZ','mQuatW',
               'mBoundingX','mBoundingY','mBoundingZ','mBoundingRadius')


def decode_table(raw, kind):
    if kind not in ('static','tree'): raise ValueError('unknown world table kind')
    r = BinaryReader(raw)
    class_name = 'QSRigidStaticDescription' if kind == 'static' else 'QSRigidSpeedTreeServerDescription'
    if r.type_name() != class_name or r.type_name() != class_name or r.u16() != 1:
        raise ValueError('unsupported world table class/version')
    count = r.u32()
    if count > 1000000: raise ValueError('world table instance budget exceeded')
    rows, ids = [], set()
    for i in range(count):
        start = r.offset
        if r.u16() != (17 if kind == 'static' else 15):
            raise ValueError(f'unsupported field count in row {i}')
        row = {'mID':r.u32()}
        if row['mID'] in ids: raise ValueError('duplicate collision instance ID')
        ids.add(row['mID'])
        if kind == 'static': row['mName'] = r.text()
        row['mModelName'], row['mModelFullPath'] = r.text(), r.text()
        for field in STATIC_FLOATS if kind == 'static' else TREE_FLOATS:
            row[field] = r.f32()
        if row['mScale'] <= 0 or row['mBoundingRadius'] < 0:
            raise ValueError(f'invalid scale/bounds in row {i}')
        if kind == 'static':
            row['mClientGppTag'] = r.u32()  # Raw enum; no visibility policy inferred.
            value = r.u8()
            if value not in (0,1): raise ValueError('nonboolean LOD flag')
            row['mHasClientLodRes'] = bool(value)
        else:
            norm = np.linalg.norm([row[k] for k in ('mQuatX','mQuatY','mQuatZ','mQuatW')])
            if not .99 < norm < 1.01: raise ValueError(f'nonunit tree quaternion in row {i}')
        row['binary_offset'] = start
        rows.append(row)
    r.expect(b'\x07GameObj\x02\x00')
    name, flags = r.text(), r.u32()
    if r.offset != len(raw): raise ValueError('trailing bytes in world table')
    return {'class':class_name,'name':name,'flags':flags,'rows':rows}


def load_tables(root, world, scope='ClientTable'):
    if scope not in ('ClientTable','ServerData'): raise ValueError('unknown shipped table scope')
    folder = root / 'Cache' / 'Data' / 'world' / world
    path = folder / (world + '.sfc')
    result, sources = {}, []
    with Sfc(path) as archive:
        for kind, table in (('static','RigidStaticInPxScene'),('tree','RigidSpeedTreeInPxScene')):
            resource = f'CACHE/DATA/WORLD/{world}/{scope}/{table}.BIN'
            raw = archive.read(resource)
            decoded = decode_table(raw,kind)
            expected = f'DATA.WORLD.{world}.{scope}.{table}'
            if decoded['name'].upper() != expected.upper():
                raise ValueError('world table identity mismatch')
            result[kind] = decoded
            sources.append({'resource':resource,'archive':str(path.resolve()),
                            'record_offset':archive.index[path_hash(resource)],'bytes':len(raw),
                            'sha256':hashlib.sha256(raw).hexdigest()})
    return result, sources


def intersects(row, roi):
    if roi is None: return True
    # Exact circle/rectangle broadphase on the authored bounding sphere's XY
    # projection, not an invented obstacle box or a collision test.
    p = np.array([row['mBoundingX'],row['mBoundingY']])
    nearest = np.clip(p,np.array(roi[:2]),np.array(roi[2:]))
    return float(np.sum((p-nearest)**2)) <= row['mBoundingRadius']**2


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--game-root',type=Path,required=True)
    ap.add_argument('--world',choices=WORLDS,default='europa')
    ap.add_argument('--scope',choices=('ClientTable','ServerData'),default='ClientTable')
    ap.add_argument('--roi',nargs=4,type=float,metavar=('XMIN','YMIN','XMAX','YMAX'))
    ap.add_argument('--out',type=Path,default=Path('output/europa_world_instances'))
    args=ap.parse_args();root=args.game_root.resolve();out=safe_output(root,args.out)
    if args.roi is not None and (not np.isfinite(args.roi).all() or args.roi[0]>=args.roi[2] or args.roi[1]>=args.roi[3]):
        raise ValueError('invalid ROI')
    tables,sources=load_tables(root,WORLDS[args.world][0],args.scope)
    summary={'source':'game_installation_only','world':WORLDS[args.world][0],'scope':args.scope,'inputs':sources,'roi_xy':args.roi,'tables':{}}
    for kind,table in tables.items():
        selected=[r for r in table['rows'] if intersects(r,args.roi)]
        with (out/(kind+'_instances.jsonl')).open('w',encoding='utf-8') as f:
            for row in selected: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        summary['tables'][kind]={'class':table['class'],'name':table['name'],'flags':table['flags'],
                                'total_instances':len(table['rows']),'selected_instances':len(selected),
                                'selected_model_paths':len(set(r['mModelFullPath'] for r in selected)),
                                'physical_formats':dict(Counter(Path(r['mModelFullPath']).suffix.upper() for r in selected))}
    summary['limitations']=['Authored bounds are broadphase only; they are not collision geometry.',
                           'Physical asset poses, shape flags and dynamic entity state need separate interpretation.',
                           'Static Euler angles remain raw radians; tree rotations are explicit xyzw quaternions.']
    dump(out/'summary.json',summary)
    dump(out/'manifest.json',[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'])
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
