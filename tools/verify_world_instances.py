"""Compare binary cache tables with independent original Dione XML tables."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from roe_map_parser.assets import GameAssets,path_hash
from roe_map_parser.formats import xml_root,array
from roe_map_parser.world_instances import load_tables,STATIC_FLOATS,TREE_FLOATS


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--game-root',type=Path,required=True)
    args=ap.parse_args();root=args.game_root.resolve()
    tables,sources=load_tables(root,'EU_ICELAND_4096');report={}
    with GameAssets(root) as game:
        for kind,name in [('static','RIGIDSTATICINPXSCENE'),('tree','RIGIDSPEEDTREEINPXSCENE')]:
            raw,source=game.read('DATA/WORLD/EU_ICELAND_4096/CLIENTTABLE/'+name+'.XML')
            desc=xml_root(raw).find(tables[kind]['class'])
            rows=[n.attrib for n in array(desc,'mItems')];binary=tables[kind]['rows']
            assert len(rows)==len(binary)
            differences=[];spelling=0
            for a,b in zip(rows,binary):
                assert int(a['mID'])==b['mID']
                for key in ['mModelName','mModelFullPath']+(['mName'] if kind=='static' else []):
                    assert path_hash(a[key])==path_hash(b[key]),(key,a[key],b[key])
                    spelling+=a[key]!=b[key]
                diff=[abs(float(a[k])-b[k]) for k in (STATIC_FLOATS if kind=='static' else TREE_FLOATS)]
                differences.append(max(diff));assert max(diff)<1e-5,(a['mID'],diff)
                if kind=='static':assert (a['mHasClientLodRes']=='true')==b['mHasClientLodRes']
            report[kind]={'rows':len(rows),'max_float_error':max(differences),
                          'same_normalized_game_hash':True,'case_or_slash_spelling_differences':spelling,
                          'xml_source':source,'binary_source':sources[0 if kind=='static' else 1]}
    print(json.dumps({'status':'passed','tables':report},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
