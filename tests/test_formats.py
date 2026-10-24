import struct
from pathlib import Path
import tempfile
import unittest

import lz4.block
import numpy as np

from roe_map_parser.assets import Sfc, path_hash
from roe_map_parser import formats as f


def array(tag, content, count=1):
    return f'<{tag} ArrayCount="{count}">{content}</{tag}>'


def transform(position="mPosition", x=1):
    return (f'<{position} x="{x}" y="2" z="3"/>'
            '<mRotMatrix><m ArrayCount="3"><m x="1" y="0" z="0"/>'
            '<m x="0" y="1" z="0"/><m x="0" y="0" z="1"/></m></mRotMatrix>'
            '<mScale x="1" y="2" z="3"/>')


PARAM = ('<mParam mWalkThrough="false" FutureFlag="keep">'
         '<mLocalPosition x="11" y="22" z="33"/>'
         '<mLocalRotation x="0" y="0" z="0" w="1"/></mParam>')


class ContainerTests(unittest.TestCase):
    def test_path_hash_normalization(self):
        self.assertEqual(path_hash(r'data//world\a.xml'), path_hash('DATA/WORLD/A.XML'))
        self.assertNotEqual(path_hash('A'), path_hash('B'))
        self.assertEqual(path_hash(''), 0x1712E27F4D2960DB)

    def test_original_sfc_raw_and_compressed(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'synthetic.sfc'
                raw = b'original-data-' * 100
                packed = lz4.block.compress(raw, store_size=False) if compressed else raw
                blob = (b'SFC\x1a\x03\0\0\0' + struct.pack('<I', 1) +
                        struct.pack('<QI', path_hash('DATA/A'), 24) + struct.pack('<I', len(packed)) +
                        packed + struct.pack('<I', len(raw)))
                path.write_bytes(blob)
                with Sfc(path) as archive:
                    self.assertEqual(archive.read('data/a'), raw)
                    with self.assertRaises(ValueError):
                        archive.read('data/a', maximum=10)
                path.write_bytes(blob[:-1])
                with Sfc(path) as archive, self.assertRaises(ValueError):
                    archive.read('data/a')

    def test_sfc_rejects_bad_index_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.sfc'
            for data in [b'wrong', b'SFC\x1a\x03\0\0\0' + struct.pack('<I', 2000001),
                         b'SFC\x1a\x03\0\0\0' + struct.pack('<I', 1) + struct.pack('<QI', 1, 0)]:
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    Sfc(path)


class TerrainTests(unittest.TestCase):
    def test_qshm_exact_layout_and_rejects_truncation(self):
        expected = np.arange(257 * 257, dtype='<u2').reshape(257, 257)
        raw = struct.pack('<III', 0xFF000001, 257, 257) + expected.tobytes()
        np.testing.assert_array_equal(f.qshm(raw), expected)
        for bad in (raw[:-1], raw + b'\0', b'wrong' + raw[5:]):
            with self.assertRaises(ValueError):
                f.qshm(bad)

    def test_weld_owns_positive_duplicate_but_keeps_last_outer_edge(self):
        tiles = np.zeros((16, 16, 257, 257), dtype='<u2')
        tiles[0, 0, :, -1] = 10
        tiles[0, 1, :, 0] = 20
        tiles[-1, -1, -1, -1] = 123
        grid = f.weld(tiles)
        self.assertEqual(grid[0, 256], 20)
        self.assertEqual(tiles[0, 0, 0, -1], 10)  # raw negative side is not modified
        self.assertEqual(grid[4096, 4096], 123)

    def test_mesh_topology_spacing_and_positive_z_winding(self):
        grid = np.broadcast_to(np.arange(4097, dtype='<u2'), (4097, 4097))
        v, triangles = f.terrain_mesh(grid, 120000, 64)
        self.assertEqual(v.shape, (65 * 65, 3))
        self.assertEqual(len(triangles), 64 * 64 * 2)
        self.assertEqual(v[:, 0].max(), 819200)
        points = v[triangles]
        self.assertTrue((np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])[:, 2] > 0).all())


class OriginalXmlTests(unittest.TestCase):
    def test_sparse_tree_xml_omits_empty_arrays_without_inventing_shapes(self):
        raw=('<NewObj><PhysCollectionDesc>'+array('mActors','<mActors>'+array('mCapsules','<mCapsules mRadius="3" mHalfHeight="4">'+PARAM+'</mCapsules>')+'</mActors>')+'</PhysCollectionDesc></NewObj>').encode()
        actor=f.phys_collection(raw)['actors'][0]
        self.assertEqual(len(actor['shapes']),1)
        self.assertEqual(actor['shapes'][0]['kind'],'capsule')
        self.assertEqual(set(actor['omitted_shape_arrays']),{'mBoxes','mSpheres','mMeshes','mConvexes'})
        with self.assertRaises(ValueError):f.phys_collection(raw.replace(b'ArrayCount="1"',b'ArrayCount="2"',1))

    def test_original_triangle_mesh_points_indices_and_pose(self):
        empty=''.join(array(tag,'',0) for tag in ('mBoxes','mSpheres','mCapsules','mConvexes'))
        mesh='<mMeshes>'+PARAM+array('mPoints','<mPoints x="0" y="0" z="0"/><mPoints x="1" y="0" z="0"/><mPoints x="0" y="1" z="0"/>',3)+array('mTriangles','<mTriangles x="0" y="2" z="1"/>')+'</mMeshes>'
        raw=('<NewObj><PhysCollectionDesc>'+array('mActors','<mActors>'+empty+array('mMeshes',mesh)+'</mActors>')+'</PhysCollectionDesc></NewObj>').encode()
        shape=f.phys_collection(raw)['actors'][0]['shapes'][0]
        self.assertEqual(shape['kind'],'triangle_mesh');self.assertEqual(shape['triangles'],[[0,2,1]])
        self.assertEqual(shape['position'],[11,22,33])
        with self.assertRaises(ValueError):f.phys_collection(raw.replace(b'x="0" y="2" z="1"',b'x="0" y="3" z="1"'))

    def test_repx_identity_reference_resolution_rejects_unsupported_pose(self):
        raw=(b'<PhysX30Collection><PxTriangleMesh><Id>7</Id><Points>0 0 0 1 0 0 0 1 0</Points><Triangles>0 1 2</Triangles></PxTriangleMesh>'
             b'<PxRigidStatic><GlobalPose>0 0 0 1 0 0 0</GlobalPose><Shapes><PxShape><Geometry><PxTriangleMeshGeometry><Scale><Scale>1 1 1</Scale><Rotation>0 0 0 1</Rotation></Scale><TriangleMesh>7</TriangleMesh></PxTriangleMeshGeometry></Geometry>'
             b'<LocalPose>0 0 0 1 0 0 0</LocalPose><Flags>eSIMULATION_SHAPE</Flags></PxShape></Shapes></PxRigidStatic></PhysX30Collection>')
        shapes=f.identity_repx_shapes(raw)
        self.assertEqual(shapes[0]['triangles'],[[0,1,2]])
        self.assertEqual(shapes[0]['parameters']['Flags'],'eSIMULATION_SHAPE')
        for bad in (raw.replace(b'<GlobalPose>0 0 0 1 0 0 0',b'<GlobalPose>0 0 0 1 10 0 0'),raw.replace(b'<Scale>1 1 1',b'<Scale>2 1 1'),raw.replace(b'<TriangleMesh>7',b'<TriangleMesh>8')):
            with self.assertRaises(ValueError):f.identity_repx_shapes(bad)

    def test_editor_attributes_preserved_and_dtd_rejected(self):
        root = f.xml_root(b'<a [XlsProp]="0" [_EditorOnlyObj]="1"/>')
        self.assertEqual(root.attrib, {'XlsProp': '0', '_EditorOnlyObj': '1'})
        with self.assertRaises(ValueError):
            f.xml_root(b'<!DOCTYPE a><a/>')
        with self.assertRaises(ValueError):
            f.array(f.xml_root(b'<a><b ArrayCount="2"><b/></b></a>'), 'b')

    def test_all_four_original_collision_shapes_and_unknown_fields(self):
        boxes = array('mBoxes', '<mBoxes>' + PARAM + '<mHalfExtents x="1" y="2" z="3"/></mBoxes>')
        spheres = array('mSpheres', '<mSpheres mRadius="2">' + PARAM + '</mSpheres>')
        capsules = array('mCapsules', '<mCapsules mRadius="3" mHalfHeight="4">' + PARAM + '</mCapsules>')
        convexes = array('mConvexes', '<mConvexes>' + PARAM + array('mPoints',
            ''.join(f'<mPoints x="{x}" y="{y}" z="{z}"/>' for x, y, z in
                    [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]), 4) + '</mConvexes>')
        raw = ('<NewObj><PhysCollectionDesc>' + array('mActors', '<mActors mUsage="static">' +
               boxes + spheres + capsules + convexes + '<FutureValue v="keep"/></mActors>') + '</PhysCollectionDesc></NewObj>').encode()
        result = f.phys_collection(raw)
        shapes = result['actors'][0]['shapes']
        self.assertEqual({s['kind'] for s in shapes}, {'box', 'sphere', 'capsule', 'convex_points'})
        self.assertEqual(shapes[0]['half_extents'], [1, 2, 3])
        self.assertEqual(shapes[0]['position'], [11, 22, 33])
        self.assertEqual(shapes[0]['parameters']['FutureFlag'], 'keep')
        self.assertEqual(result['actors'][0]['uninterpreted_children'][0]['tag'], 'FutureValue')
        with self.assertRaises(ValueError):
            f.phys_collection(raw.replace(b'w="1"', b'w="0"'))

    def test_repx_mesh_exact_indices_and_preserves_shape_pose(self):
        raw = (b'<PhysX30Collection version="3.3.4"><PxTriangleMesh><Id>7</Id>'
               b'<Points>0 0 0 1 0 0 0 1 0</Points><Triangles>0 1 2</Triangles></PxTriangleMesh>'
               b'<PxRigidStatic><GlobalPose>0 0 0 1 5 6 7</GlobalPose></PxRigidStatic></PhysX30Collection>')
        result = f.repx(raw)
        self.assertEqual(result['triangle_meshes'][0]['triangles'], [[0, 1, 2]])
        self.assertEqual(result['other_objects'][0]['children'][0]['text'], '0 0 0 1 5 6 7')
        for bad in (raw.replace(b'0 1 2', b'0 1 3'), raw.replace(b'0 1 2', b'0 -1 2'),
                    raw.replace(b'0 0 0 1 0 0', b'nan 0 0 1 0 0')):
            with self.assertRaises(ValueError):
                f.repx(bad)

    def test_prefab_instance_raw_transform_and_entity_identity(self):
        body = ('<LayoutInstance mName="instance" mParentName="Data.prefab">' + transform() +
                array('mEntityNames', '<mEntityNames mEntityName="Data.world.a" mNodeName="Data.prefab.node_0"/>') + '</LayoutInstance>')
        raw = ('<NewObj><LayoutManager>' + array('mLayoutNames', '<mLayoutNames v="Data.prefab"/>') +
               body + '</LayoutManager></NewObj>').encode()
        result = f.layout_manager(raw)
        self.assertEqual(result['instances'][0]['transform']['position'], [1, 2, 3])
        self.assertEqual(result['instances'][0]['entity_links'][0]['mEntityName'], 'Data.world.a')
        with self.assertRaises(ValueError):
            f.layout_manager(raw.replace(b'x="1" y="2" z="3"', b'x="nan" y="2" z="3"', 1))

    def test_rigid_instance_catalog_preserves_ids_paths_and_rejects_duplicates(self):
        row = ('<mItems mID="42" mModelFullPath="Data/model.xml" mPosX="1" mPosY="2" mPosZ="3" '
               'mScale="1" mRotateX="0" mRotateY="0" mRotateZ="1.5"/>')
        raw = ('<NewObj><QSRigidStaticDescription>' + array('mItems', row) + '</QSRigidStaticDescription></NewObj>').encode()
        self.assertEqual(f.rigid_instances(raw)[0]['mModelFullPath'], 'Data/model.xml')
        bad = ('<NewObj><QSRigidStaticDescription>' + array('mItems', row + row, 2) + '</QSRigidStaticDescription></NewObj>').encode()
        with self.assertRaises(ValueError):
            f.rigid_instances(bad)


if __name__ == '__main__':
    unittest.main()
