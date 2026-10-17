from pathlib import Path
import sys,json,importlib.util,argparse
ROOT=Path(__file__).resolve().parent
import trimesh
import numpy as np
spec=importlib.util.spec_from_file_location('case_geometry',ROOT/'CameraSnapCase'/'geometry.py')
geo=importlib.util.module_from_spec(spec); spec.loader.exec_module(geo)

def solid(kind,d):
    if kind=='box':
        m=trimesh.creation.box(extents=[d[1]-d[0],d[3]-d[2],d[5]-d[4]])
        m.apply_translation([(d[0]+d[1])/2,(d[2]+d[3])/2,(d[4]+d[5])/2])
    elif kind=='cylinder':
        x,y,z,r,h=d
        m=trimesh.creation.cylinder(radius=r,height=h,sections=64)
        m.apply_translation([x,y,z+h/2])
    else:
        vertices=[[x,y,z] for y in d[3:] for x,z in d[:3]]
        faces=[[0,2,1],[3,4,5],[0,1,4],[0,4,3],[1,2,5],[1,5,4],[2,0,3],[2,3,5]]
        m=trimesh.Trimesh(vertices=vertices,faces=faces)
        if m.volume < 0: m.invert()
    return m

def count_bodies(m):
    parents=list(range(len(m.vertices)))
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]]; i=parents[i]
        return i
    for a,b in m.edges: parents[find(int(a))]=find(int(b))
    return len({find(i) for i in range(len(parents))})

def build(thickness):
    data=geo.geometry(thickness)
    out=ROOT/'snap-case-v2'; out.mkdir(exist_ok=True)
    results=[]; meshes={}
    for part in ['back','lid']:
        adds=[solid(k,d) for op,k,n,d in data[part] if op=='add']
        cuts=[solid(k,d) for op,k,n,d in data[part] if op=='cut']
        m=trimesh.boolean.union(adds,engine='manifold')
        if cuts: m=trimesh.boolean.difference([m]+cuts,engine='manifold')
        assert m.is_watertight and m.is_winding_consistent and count_bodies(m)==1
        meshes[part]=m.copy()
        m.export(out/f'{part}-assembled.ply')
        # Front frame is printed front-face-down, latch legs pointing upward.
        if part=='lid': m.apply_transform(trimesh.transformations.rotation_matrix(np.pi,[1,0,0]))
        m.apply_translation(-m.bounds[0])
        name=f'camera-{part}-v2-pcb-{thickness:g}mm.stl'
        m.export(out/name)
        results.append({'file':name,'bounds_mm':m.bounds.tolist(),'volume_mm3':float(m.volume),'watertight':True,'connected_bodies':1})
    interference=trimesh.boolean.intersection([meshes['back'],meshes['lid']],engine='manifold')
    assert abs(interference.volume)<0.001, f'Assembled interference: {interference.volume}'
    (out/'validation.json').write_text(json.dumps({'parameters':{k:v for k,v in data.items() if k not in ['back','lid']},'assembled_interference_mm3':float(interference.volume),'parts':results},indent=2))
    (out/'geometry.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(results,indent=2))
    return meshes,data

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pcb-thickness',type=float,required=True)
    args=parser.parse_args()
    build(args.pcb_thickness)
