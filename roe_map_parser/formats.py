"""Original client resource formats; named XML fields stay named, unknowns retained."""
from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET

import numpy as np


def xml_root(raw):
    if len(raw) > 128 * 1024 * 1024 or b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("XML size/DTD budget rejected")
    # Client reflection XML allows bracketed editor attribute names. Keep their
    # values, spelling the two observed names without brackets for stdlib XML.
    raw = re.sub(rb'\[(XlsProp|_EditorOnlyObj)\](\s*=)', rb'\1\2', raw)
    return ET.fromstring(raw)


def array(node, tag):
    container = node.find(tag)
    if container is None:
        raise ValueError(f"missing array: {tag}")
    rows = list(container)
    if len(rows) > 1000000 or int(container.get("ArrayCount", "-1")) != len(rows):
        raise ValueError(f"incorrect ArrayCount: {tag}")
    if any(n.tag != tag for n in rows):
        raise ValueError(f"unexpected child in {tag}")
    return rows


def vector(node, axes="xyz"):
    if node is None:
        raise ValueError("missing vector")
    value = [float(node.attrib[k]) for k in axes]
    if not np.isfinite(value).all():
        raise ValueError("nonfinite vector")
    return value


def numeric_text(node, dtype=float, columns=3):
    if node is None:
        raise ValueError("missing numeric array")
    tokens = (node.text or "").split()
    if len(tokens) > 12000000 or len(tokens) % columns:
        raise ValueError("invalid numeric array length")
    values = np.asarray([dtype(v) for v in tokens])
    if not np.isfinite(values).all():
        raise ValueError("nonfinite numeric array")
    return values.reshape(-1, columns)


def qshm(raw):
    if len(raw) != 12 + 257 * 257 * 2 or struct.unpack_from("<III", raw) != (0xFF000001, 257, 257):
        raise ValueError("unsupported QSHM header or length")
    return np.frombuffer(raw, "<u2", offset=12).reshape(257, 257)


def weld(tiles):
    if tiles.shape != (16, 16, 257, 257):
        raise ValueError("expected 16x16 QSHM tiles")
    grid = np.empty((4097, 4097), dtype="<u2")
    for y in range(16):
        for x in range(16):
            ny, nx = (257 if y == 15 else 256), (257 if x == 15 else 256)
            grid[y * 256:y * 256 + ny, x * 256:x * 256 + nx] = tiles[y, x, :ny, :nx]
    return grid


def terrain_mesh(grid, peak, step=16):
    if step not in (8, 16, 32, 64):
        raise ValueError("mesh step must be 8/16/32/64")
    indices = np.arange(0, 4097, step)
    xx, yy = np.meshgrid(indices * 200, indices * 200)
    zz = grid[np.ix_(indices, indices)].astype(float) * peak / 65535
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    a = np.arange(len(indices) ** 2).reshape(len(indices), -1)[:-1, :-1].ravel()
    faces = np.concatenate([np.column_stack([a, a + 1, a + len(indices) + 1]),
                            np.column_stack([a, a + len(indices) + 1, a + len(indices)])])
    return vertices, faces


def tree(node):
    return {"tag": node.tag, "attributes": dict(node.attrib),
            **({"text": node.text.strip()} if node.text and node.text.strip() else {}),
            **({"children": [tree(n) for n in node]} if len(node) else {})}


def phys_collection(raw):
    root = xml_root(raw)
    desc = root.find("PhysCollectionDesc")
    if desc is None:
        raise ValueError("expected original PhysCollectionDesc")
    actors = []
    kinds = {"mBoxes": "box", "mSpheres": "sphere", "mCapsules": "capsule", "mConvexes": "convex_points", "mMeshes": "triangle_mesh"}
    for actor in array(desc, "mActors"):
        shapes = []
        for tag, kind in kinds.items():
            if actor.find(tag) is None:
                continue  # Original sparse XML omits default empty arrays.
            for shape in array(actor, tag):
                param = shape.find("mParam")
                if param is None:
                    raise ValueError("missing shape mParam")
                item = {"kind": kind, "parameters": dict(param.attrib),
                        "position": vector(param.find("mLocalPosition")),
                        "quaternion_xyzw": vector(param.find("mLocalRotation"), "xyzw"),
                        "attributes": dict(shape.attrib)}
                if not 0.99 < np.linalg.norm(item["quaternion_xyzw"]) < 1.01:
                    raise ValueError("nonunit shape quaternion")
                if kind == "box":
                    item["half_extents"] = vector(shape.find("mHalfExtents"))
                    if min(item["half_extents"]) <= 0:
                        raise ValueError("nonpositive box")
                elif kind in ("convex_points", "triangle_mesh"):
                    item["points"] = [vector(n) for n in array(shape, "mPoints")]
                    if kind == 'triangle_mesh':
                        item['triangles'] = [[int(n.attrib[k]) for k in 'xyz'] for n in array(shape,'mTriangles')]
                        ids = np.asarray(item['triangles'],dtype=int)
                        if not len(ids) or ids.min()<0 or ids.max()>65535 or ids.max()>=len(item['points']):
                            raise ValueError('invalid original triangle mesh indices')
                    elif len(item["points"]) < 4:
                        raise ValueError("incomplete convex point cloud")
                else:
                    item["radius"] = float(shape.attrib["mRadius"])
                    if kind == "capsule":
                        item["half_height"] = float(shape.attrib["mHalfHeight"])
                    if not np.isfinite([item["radius"], item.get("half_height", 0)]).all() or item["radius"] <= 0 or item.get("half_height", 0) < 0:
                        raise ValueError("invalid sphere/capsule dimensions")
                # Preserve the whole original element, including future fields.
                item["source_fields"] = tree(shape)
                shapes.append(item)
        actors.append({"attributes": dict(actor.attrib), "shapes": shapes,
                       "omitted_shape_arrays": [tag for tag in kinds if actor.find(tag) is None],
                       "uninterpreted_children": [tree(n) for n in actor if n.tag not in kinds]})
    return {"format": "PhysCollectionDesc", "object": root.get("ObjName"), "actors": actors,
            "coordinate_scope": "model local; not placed in a world", "source_fields": dict(desc.attrib)}


def repx(raw):
    root = xml_root(raw)
    if root.tag != "PhysX30Collection":
        raise ValueError("expected PhysX30Collection")
    meshes = []
    for mesh in root.findall("PxTriangleMesh"):
        vertices = numeric_text(mesh.find("Points"))
        triangles = numeric_text(mesh.find("Triangles"), int)
        if not len(vertices) or (len(triangles) and (triangles.min() < 0 or triangles.max() >= len(vertices))):
            raise ValueError("invalid RepX triangle indices")
        meshes.append({"id": mesh.findtext("Id"), "vertices": vertices.tolist(),
                       "triangles": triangles.tolist()})
    return {"format": "PhysX30Collection", "version": root.get("version"),
            "triangle_meshes": meshes,
            "other_objects": [tree(n) for n in root if n.tag != "PxTriangleMesh"],
            "coordinate_scope": "raw mesh coordinates; actor/shape/mesh poses are preserved, not applied"}


def identity_repx_shapes(raw):
    """Resolve referenced static triangle meshes with explicitly identity poses.

    TDM's three referenced RepX resources have this form. Nonidentity actor,
    shape and mesh poses, other geometry and actor classes remain unsupported.
    """
    result = repx(raw)
    root = xml_root(raw)
    meshes = {m['id']:m for m in result['triangle_meshes']}
    if len(meshes)!=len(result['triangle_meshes']): raise ValueError('duplicate RepX mesh ID')
    shapes = []
    def expect_numbers(node,path,expected):
        values = [float(v) for v in (node.findtext(path) or '').split()]
        if values != expected: raise ValueError('unsupported nonidentity/missing RepX '+path)
    for actor in root:
        if actor.tag in ('UpVector','Scale','PxMaterial','PxTriangleMesh'): continue
        if actor.tag!='PxRigidStatic': raise ValueError('unsupported RepX actor '+actor.tag)
        expect_numbers(actor,'GlobalPose',[0,0,0,1,0,0,0])
        container=actor.find('Shapes')
        if container is None: raise ValueError('missing RepX shapes')
        for shape in container:
            if shape.tag!='PxShape': raise ValueError('unsupported RepX shape')
            expect_numbers(shape,'LocalPose',[0,0,0,1,0,0,0])
            geometry=shape.find('Geometry')
            if geometry is None or len(geometry)!=1 or geometry[0].tag!='PxTriangleMeshGeometry':
                raise ValueError('unsupported RepX geometry')
            geometry=geometry[0]
            expect_numbers(geometry,'Scale/Scale',[1,1,1])
            expect_numbers(geometry,'Scale/Rotation',[0,0,0,1])
            mesh=meshes.get(geometry.findtext('TriangleMesh'))
            if mesh is None: raise ValueError('unresolved RepX mesh reference')
            shapes.append({'kind':'triangle_mesh','position':[0,0,0],'quaternion_xyzw':[0,0,0,1],
                           'points':mesh['vertices'],'triangles':mesh['triangles'],
                           'parameters':{k:shape.findtext(k) for k in ('Flags','SimulationFilterData','QueryFilterData','Name')},
                           'repx_mesh_id':mesh['id']})
    if not shapes: raise ValueError('no supported RepX shapes')
    return shapes


def layout_transform(node, position):
    matrix = array(node.find("mRotMatrix"), "m")
    if len(matrix) != 3:
        raise ValueError("layout rotation must be 3x3")
    return {"position": vector(node.find(position)), "matrix_rows_as_stored": [vector(n) for n in matrix],
            "scale": vector(node.find("mScale"))}


def layout_manager(raw):
    root = xml_root(raw)
    manager = root.find("LayoutManager")
    if manager is None:
        raise ValueError("expected LayoutManager")
    definitions = [n.attrib["v"] for n in array(manager, "mLayoutNames")]
    instances = []
    for node in manager.findall(".//LayoutInstance"):
        instances.append({"attributes": dict(node.attrib),
                          "transform": layout_transform(node, "mPosition"),
                          "entity_links": [dict(n.attrib) for n in array(node, "mEntityNames")]})
    return {"definitions": definitions, "instances": instances,
            "coordinate_policy": "Raw named transforms only; no inferred matrix composition or collision binding."}


def layout_definition(raw):
    root = xml_root(raw)
    layout = root.find("LayoutBaseObject")
    if layout is None:
        raise ValueError("expected LayoutBaseObject")
    return {"object": root.attrib["ObjName"], "attributes": dict(layout.attrib),
            "nodes": [{"attributes": dict(n.attrib), "transform": layout_transform(n, "mRelativePos")}
                      for n in array(layout, "mNodes")],
            "uninterpreted_children": [tree(n) for n in layout if n.tag not in ("mNodes",)]}


def rigid_instances(raw):
    root = xml_root(raw)
    desc = root.find("QSRigidStaticDescription")
    if desc is None:
        raise ValueError("expected QSRigidStaticDescription")
    records = []
    ids = set()
    for n in array(desc, "mItems"):
        row = dict(n.attrib)
        id_ = int(row["mID"])
        if id_ in ids:
            raise ValueError("duplicate rigid instance ID")
        ids.add(id_)
        floats = [float(row[k]) for k in ("mPosX", "mPosY", "mPosZ", "mScale", "mRotateX", "mRotateY", "mRotateZ")]
        if not np.isfinite(floats).all() or floats[3] <= 0:
            raise ValueError("invalid rigid instance transform")
        records.append(row)
    return records
