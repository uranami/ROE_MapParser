"""Read original map resources from a preserved game installation."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import struct
import sys

import numpy as np

from .assets import GameAssets, sha256
from . import formats as fmt
from .common import WORLDS, dump, safe_output, manifest


def obj(path, vertices, faces):
    if not np.isfinite(vertices).all() or (len(faces) and (faces.min() < 0 or faces.max() >= len(vertices))):
        raise ValueError("invalid export mesh")
    with path.open("w", encoding="utf-8") as f:
        f.write("# Game-only source. Original XYZ units, Z up.\n")
        np.savetxt(f, vertices, fmt="v %.6f %.6f %.6f")
        np.savetxt(f, faces + 1, fmt="f %d %d %d")


def terrain(game, args, out):
    world, map_id = WORLDS[args.world]
    prefix = "DATA\\WORLD\\" + world + "\\"
    raw, source = game.read(prefix + "CONFIG\\EDITORTERRAINBRUSHCONFIG.XML")
    node = fmt.xml_root(raw).find("QSTerrainEditor")
    if node is None:
        raise ValueError("missing QSTerrainEditor")
    peak = int(node.attrib["mLevelPeakHeight"])
    if not 1000 <= peak <= 1000000:
        raise ValueError("invalid terrain peak")
    sources = [source]
    tiles = np.empty((16, 16, 257, 257), dtype="<u2")
    for ty in range(16):
        for tx in range(16):
            raw, meta = game.read(prefix + f"TERRAINHEIGHTTILE_{tx}_{ty}.QSHM")
            tiles[ty, tx] = fmt.qshm(raw)
            sources.append({**meta, "tile": [tx, ty]})
    grid = fmt.weld(tiles)
    # This is our compatibility export, not a file read from another server.
    with (out / f"terrain-surface{map_id}.bin.gz").open("wb") as f:
        with gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0) as z:
            z.write(struct.pack("<8s4I", b"ROEHGT1\0", peak, 200, 257, 16))
            z.write(tiles.tobytes())
    vertices, faces = fmt.terrain_mesh(grid, peak, args.step)
    obj(out / "terrain.obj", vertices, faces)
    rows = []
    for axis in ("x", "y"):
        for strip in range(16):
            for boundary in range(1, 16):
                a = tiles[strip, boundary - 1, :, -1] if axis == "x" else tiles[boundary - 1, strip, -1, :]
                b = tiles[strip, boundary, :, 0] if axis == "x" else tiles[boundary, strip, 0, :]
                for i in np.flatnonzero(a != b):
                    rows.append({"axis": axis, "strip": strip, "boundary": boundary, "sample": int(i),
                                 "negative_u16": int(a[i]), "positive_u16": int(b[i]),
                                 "delta_z": (int(b[i]) - int(a[i])) * peak / 65535})
    with (out / "seams.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["axis", "strip", "boundary", "sample", "negative_u16", "positive_u16", "delta_z"])
        writer.writeheader()
        writer.writerows(rows)
    return {"world": world, "map": map_id, "source": "game_installation_only", "inputs": sources,
            "tiles": 256, "tile_samples": [257, 257], "sample_spacing": 200, "peak": peak,
            "sample_count": int(tiles.size), "unique_grid": list(grid.shape),
            "sample_u16_min_max": [int(tiles.min()), int(tiles.max())],
            "seam_mismatches": len(rows), "max_seam_z_difference": max((abs(r["delta_z"]) for r in rows), default=0),
            "mesh": {"step_samples": args.step, "vertices": len(vertices), "triangles": len(faces)},
            "policies": ["Original XYZ units; Z up; height=u16*editor_peak/65535.",
                         "Positive-side tile owns duplicate interior coordinates for welded export; CSV retains both values.",
                         "OBJ uses a fixed cell diagonal as an export choice, not recovered renderer topology."],
            "limitations": ["Bare terrain only; this command supplies no buildings, water or navigation."]}


def collision(game, args, out):
    raw, source = game.read(args.resource)
    root = fmt.xml_root(raw)
    result = fmt.repx(raw) if root.tag == "PhysX30Collection" else fmt.phys_collection(raw)
    dump(out / "collision.json", result)
    (out / "source.xml").write_bytes(raw)
    if result["format"] == "PhysX30Collection":
        for i, mesh in enumerate(result["triangle_meshes"]):
            obj(out / f"mesh_{i}.obj", np.asarray(mesh["vertices"]), np.asarray(mesh["triangles"], dtype=int).reshape(-1, 3))
        counts = {"triangle_meshes": len(result["triangle_meshes"]),
                  "vertices": sum(len(m["vertices"]) for m in result["triangle_meshes"]),
                  "triangles": sum(len(m["triangles"]) for m in result["triangle_meshes"])}
    else:
        counts = dict(Counter(s["kind"] for a in result["actors"] for s in a["shapes"]))
    return {"source": "game_installation_only", "inputs": [source], "format": result["format"], "counts": counts,
            "coordinate_scope": result["coordinate_scope"],
            "limitations": ["Model-local parse; world placement and client query/filter semantics are not inferred.",
                             "Convexes remain source point clouds; original fields/flags and unhandled children are retained."]}


def layouts(game, args, out):
    world, map_id = WORLDS[args.world]
    raw, source = game.read(f"DATA\\WORLD\\{world}\\CONFIG\\LAYOUTMANAGER.XML")
    manager = fmt.layout_manager(raw)
    dump(out / "layout_instances.json", manager)
    sources, missing, definitions = [source], [], {}
    for key in manager["definitions"]:
        path = key.replace(".", "\\") + ".XML"
        try:
            raw, meta = game.read(path)
        except FileNotFoundError:
            missing.append(path)
            continue
        definitions[key.lower()] = fmt.layout_definition(raw)
        sources.append(meta)
    dump(out / "layout_definitions.json", definitions)
    missing_links = []
    for inst in manager["instances"]:
        parent = inst["attributes"]["mParentName"].lower()
        definition = definitions.get(parent)
        if definition is None:
            continue
        nodes = {n["attributes"]["mName"].lower() for n in definition["nodes"]}
        for link in inst["entity_links"]:
            node = link["mNodeName"].rsplit(".", 1)[-1].lower()
            if node not in nodes:
                missing_links.append(link)
    return {"source": "game_installation_only", "world": world, "map": map_id, "inputs": sources,
            "referenced_definitions": len(manager["definitions"]), "parsed_definitions": len(definitions),
            "instances": len(manager["instances"]),
            "entity_links": sum(len(i["entity_links"]) for i in manager["instances"]),
            "missing_definitions": missing, "unresolved_node_links": missing_links,
            "limitations": ["Prefab metadata covers only listed instances; it is not a complete world collision catalog.",
                             "Raw matrices/transforms retained. No inferred composition, mesh-to-physics name substitution or BOT data generated."]}


def rigid(game, args, out):
    world, map_id = WORLDS[args.world]
    resource = f"DATA\\WORLD\\{world}\\CLIENTTABLE\\RIGIDSTATICINPXSCENE.XML"
    raw, source = game.read(resource)
    rows = fmt.rigid_instances(raw)
    with (out / "rigid_instances.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    models = sorted(set(r["mModelFullPath"] for r in rows))
    dump(out / "model_paths.json", models)
    return {"source": "game_installation_only", "world": world, "map": map_id, "inputs": [source],
            "instances": len(rows), "models": len(models),
            "limitations": ["Named source transforms and model references only; no generated world collision.",
                             "Absent resources cause an error; no substitution from another map or server."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("terrain", "collision", "layouts", "rigid-instances", "world-instances", "world"):
        p = sub.add_parser(command)
        p.add_argument("--game-root", required=True, type=Path)
        p.add_argument("--out", required=True, type=Path)
        if command == "collision":
            p.add_argument("--resource", required=True)
        else:
            p.add_argument("--world", choices=tuple(WORLDS), default="europa")
        if command in ("terrain", "world"):
            p.add_argument("--step", type=int, choices=(8, 16, 32, 64), default=16)
        if command == "world-instances":
            p.add_argument('--scope', choices=('ClientTable', 'ServerData'), default='ClientTable')
            p.add_argument('--roi', nargs=4, type=float, metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'))
    args = parser.parse_args(argv)
    try:
        out = safe_output(args.game_root, args.out)
        if args.command == 'world-instances':
            from .world import export_tables
            if args.roi is not None and (not np.isfinite(args.roi).all() or args.roi[0]>=args.roi[2] or args.roi[1]>=args.roi[3]):
                raise ValueError('invalid ROI')
            result = export_tables(args.game_root.resolve(), WORLDS[args.world][0], out, args.scope, args.roi)
        else:
            with GameAssets(args.game_root) as game:
                if args.command == 'world':
                    from .world import export_world
                    result = export_world(game, args, out, terrain)
                else:
                    result = {"terrain": terrain, "collision": collision, "layouts": layouts,
                              "rigid-instances": rigid}[args.command](game, args, out)
        dump(out / "summary.json", result)
        manifest(out)
        compact = {k: v for k, v in result.items() if k not in ("inputs", "policies", "limitations", "missing_definitions", "unresolved_node_links")}
        print(json.dumps({"out": str(out), **compact}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, fmt.ET.ParseError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
