import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from roe_map_parser.transforms import instance_matrix
from roe_map_parser.world import export_world, parse_model


class WorldExportTests(unittest.TestCase):
    def test_static_and_tree_matrices_agree_and_compose_source_positions(self):
        row = {'mScale': 2, 'mPosX': 123, 'mPosY': 456, 'mPosZ': 789,
               'mRotateX': 0, 'mRotateY': 0, 'mRotateZ': np.pi/2,
               'mQuatX': 0, 'mQuatY': 0, 'mQuatZ': np.sqrt(.5), 'mQuatW': np.sqrt(.5)}
        a, b = [instance_matrix(row, kind) for kind in ('static', 'tree')]
        np.testing.assert_allclose(a, b, atol=1e-14)
        np.testing.assert_allclose(a@np.array([1, 0, 0, 1]), [123, 458, 789, 1])
        with self.assertRaises(ValueError):
            instance_matrix({**row, 'mScale': -1}, 'tree')

    def test_repx_unknown_pose_is_retained_without_resolving_fake_shapes(self):
        raw = (b'<PhysX30Collection version="3.3.4"><PxTriangleMesh><Id>7</Id>'
               b'<Points>0 0 0 1 0 0 0 1 0</Points><Triangles>0 1 2</Triangles></PxTriangleMesh>'
               b'<PxRigidStatic><GlobalPose>0 0 0 1 5 6 7</GlobalPose></PxRigidStatic></PhysX30Collection>')
        parsed, reason = parse_model(raw)
        self.assertIn('GlobalPose', reason)
        self.assertNotIn('resolved_static_shapes', parsed)
        self.assertEqual(parsed['other_objects'][0]['children'][0]['text'], '0 0 0 1 5 6 7')

    def test_world_keeps_instances_with_malformed_resources_and_exports_model_once(self):
        row = {'mID': 7, 'mModelFullPath': 'missing.xml', 'mScale': 1,
               'mPosX': 1, 'mPosY': 2, 'mPosZ': 3, 'mRotateX': 0, 'mRotateY': 0, 'mRotateZ': 0}
        tables = {'static': {'rows': [row, {**row, 'mID': 8}]}, 'tree': {'rows': []}}
        class Game:
            root = Path('.')
            reads = 0
            def read(self, path):
                self.reads += 1
                raise FileNotFoundError('missing original resource')
        game = Game()
        with tempfile.TemporaryDirectory() as directory, patch('roe_map_parser.world.load_tables', return_value=(tables, [])):
            out = Path(directory)
            terrain = lambda *args: {'inputs': [], 'tiles': 256}
            result = export_world(game, SimpleNamespace(world='europa'), out, terrain)
            self.assertEqual(game.reads, 1)
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['unsupported_instances'], 2)
            self.assertEqual(result['resolved_instances'], 0)
            rows = [json.loads(s) for s in (out/'static_instances.jsonl').read_text().splitlines()]
            self.assertEqual([r['original']['mID'] for r in rows], [7, 8])
            self.assertTrue(all(r['model_status'] == 'unparsed' for r in rows))
            with self.assertRaises(ValueError):
                export_world(game, SimpleNamespace(world='europa'), out, terrain)


if __name__ == '__main__':
    unittest.main()
