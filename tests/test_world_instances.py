import struct
import unittest

from roe_map_parser.world_instances import BinaryReader,decode_table,intersects


def text(value):
    data=value.encode('utf-8');n=len(data)
    return (bytes([192+n]) if n<64 else bytes([128+(n&63),n>>6]))+data


def table(kind='static'):
    name='QSRigidStaticDescription' if kind=='static' else 'QSRigidSpeedTreeServerDescription'
    tag=bytes([len(name)])+name.encode()
    row=struct.pack('<HI',17 if kind=='static' else 15,7)
    if kind=='static':row+=text('Data.world.example.rock7')
    row+=text('rock_collision.xml')+text('Data/Scene/general/stone/'+('rock_'*12)+'collision.xml')
    values=[100,200,300,1.25,.1,.2,.3,101,202,303,42] if kind=='static' else [100,200,300,1.25,0,0,0,1,101,202,303,42]
    row+=struct.pack('<'+str(len(values))+'f',*values)
    if kind=='static':row+=struct.pack('<IB',0x01020304,1)
    return tag+tag+struct.pack('<HI',1,1)+row+b'\x07GameObj\x02\0'+text('DATA.WORLD.example.ClientTable.Rigid')+struct.pack('<I',2098176)


class WorldTableTests(unittest.TestCase):
    def test_short_and_long_interned_strings_and_invalid_framing(self):
        for value in ('','x'*63,'y'*64,'z'*129,'树木'):
            self.assertEqual(BinaryReader(text(value)).text(),value)
        for raw in (b'\x01x',b'\x80\x80',b'\x80\0',b'\xc5abc'):
            with self.assertRaises(ValueError):BinaryReader(raw).text()

    def test_original_static_schema_keeps_paths_enums_lod_and_raw_offset(self):
        decoded=decode_table(table(),'static');row=decoded['rows'][0]
        self.assertEqual(row['mID'],7)
        self.assertEqual(row['mName'],'Data.world.example.rock7')
        self.assertTrue(row['mModelFullPath'].endswith('collision.xml'))
        self.assertEqual([row[k] for k in ('mPosX','mPosY','mPosZ','mScale')],[100,200,300,1.25])
        self.assertEqual(row['mClientGppTag'],0x01020304)
        self.assertTrue(row['mHasClientLodRes'])
        self.assertEqual(table()[row['binary_offset']:row['binary_offset']+2],b'\x11\0')

    def test_tree_schema_has_explicit_quaternion_and_no_invented_entity_name(self):
        row=decode_table(table('tree'),'tree')['rows'][0]
        self.assertEqual([row[k] for k in ('mQuatX','mQuatY','mQuatZ','mQuatW')],[0,0,0,1])
        self.assertNotIn('mName',row)

    def test_truncated_trailing_and_wrong_class_records_are_rejected(self):
        raw=table()
        for bad in (raw[:-1],raw+b'\0',raw.replace(b'\x11\0\x07\0\0\0',b'\x10\0\x07\0\0\0'),raw.replace(b'QSRigidStaticDescription',b'QSRigidSpeedTreeServerDescription',1)):
            with self.assertRaises(ValueError):decode_table(bad,'static')

    def test_roi_uses_sphere_intersection_instead_of_origin_only(self):
        row={'mBoundingX':-2,'mBoundingY':5,'mBoundingRadius':3}
        self.assertTrue(intersects(row,[0,0,10,10]))
        self.assertFalse(intersects(row,[2,0,10,10]))


if __name__=='__main__':unittest.main()
