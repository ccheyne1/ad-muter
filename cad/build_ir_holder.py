"""Two open optical bays, M2-retained frame; dimensions in millimetres."""
from build_threaded_mount import trimesh,np,solid,count_bodies
from pathlib import Path
import math,json
OUT=Path(__file__).resolve().parent/'ir-holder';OUT.mkdir(exist_ok=True)
# User measurements. Clearance is total, not per side.
BOARD_W,BOARD_L,ENVELOPE_H=19.5,23,7
W,L=BOARD_W+1,BOARD_L+1
bay_x=[4,4+W+2.4]; outer=8+2*W+2.4; length=L+8
floor=2; seat=floor+ENVELOPE_H+1
boxes=[solid('box',[0,outer,0,length,0,floor])]
for a,b in [(2.2,4),(4+W,6.4+W),(outer-4,outer-2.2)]:
    boxes.append(solid('box',[a,b,0,length,1.9,seat]))
# End stops only at board corners; full-height centre openings clear optics and wires.
for x in bay_x:
    for a,b in [(x,x+2.5),(x+W-2.5,x+W)]:
        for c,d in [(2.2,4),(4+L,5.8+L)]:boxes.append(solid('box',[a,b,c,d,1.9,seat]))
screws=[(-.5,3),(-.5,length-3),(outer+.5,3),(outer+.5,length-3)]
for x,y in screws:boxes.append(solid('cylinder',[x,y,0,3.2,seat]))
back=trimesh.boolean.union(boxes,engine='manifold');cuts=[]
for x,y in screws:
    pocket=trimesh.creation.cylinder(radius=4.25/math.sqrt(3),height=2.1,sections=6)
    pocket.apply_translation([x,y,seat-.95]);cuts.append(pocket)
    cuts.append(solid('cylinder',[x,y,seat-4,1.15,4.1]))
back=trimesh.boolean.difference([back]+cuts,engine='manifold')
# Flat retaining lattice: broad windows expose both components and optical ends.
lid=solid('box',[-3.7,outer+3.7,0,length,seat,seat+2.4])
cuts=[]
for x in bay_x:
    cuts.append(solid('box',[x+.6,x+W-.6,4.6,3.4+L,seat-.1,seat+2.5]))
    # Central exits run through the ends, while side tabs capture board corners.
    for a,b in [(-.1,4.7),(3.3+L,length+.1)]:
        cuts.append(solid('box',[x+2.5,x+W-2.5,a,b,seat-.1,seat+2.5]))
for x,y in screws:cuts.append(solid('cylinder',[x,y,seat-.1,1.15,2.6]))
lid=trimesh.boolean.difference([lid]+cuts,engine='manifold')
lid=trimesh.boolean.union([lid,solid('box',[-3.7,outer+3.7,length/2-1.5,length/2+1.5,seat,seat+2.4])],engine='manifold')
report={'measured_module_mm':[BOARD_W,BOARD_L,ENVELOPE_H],'bay_mm':[W,L,ENVELOPE_H+1],'hardware':'four M2 x 6 mm screws and four M2 nuts','parts':[]}
for name,m in [('back',back),('lid',lid)]:
    assert m.is_watertight and m.is_winding_consistent and count_bodies(m)==1
    m.export(OUT/f'{name}-assembled.ply')
    p=m.copy();p.apply_translation(-p.bounds[0]);p.export(OUT/f'ir-holder-{name}.stl')
    assert trimesh.load_mesh(OUT/f'ir-holder-{name}.stl').is_watertight
    report['parts'].append({'name':name,'size_mm':p.extents.tolist(),'watertight':True})
assert abs(trimesh.boolean.intersection([back,lid],engine='manifold').volume)<1e-5
(OUT/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
