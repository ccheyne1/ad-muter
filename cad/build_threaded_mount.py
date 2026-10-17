"""Printed right-hand 1/4-20 socket and compatible camera rear shell.

The internal thread uses a truncated 60-degree basic profile, not a certified
UNC tolerance class. A diametral print allowance is applied to both radii.
"""
from pathlib import Path
import sys,json,math
ROOT=Path(__file__).resolve().parent
import numpy as np,trimesh
from build_snap_case import count_bodies,solid

PITCH=25.4/20
MAJOR=6.35
MINOR=MAJOR-1.0825317547305484*PITCH
OUT=ROOT/'threaded-mount'; OUT.mkdir(exist_ok=True)

def thread_tool(z0,z1,allowance,entry_z,exit_z=None):
    ntheta=128
    nz=math.ceil((z1-z0)/(PITCH/40))+1
    z=np.linspace(z0,z1,nz)
    theta=np.arange(ntheta)*2*np.pi/ntheta
    zz,tt=np.meshgrid(z,theta,indexing='ij')
    # z increases with positive azimuth: a right-hand helix.
    phase=(zz/PITCH-tt/(2*np.pi)+.5)%1-.5
    distance=np.abs(phase)*PITCH
    groove=np.clip((3*PITCH/8-distance)/(5*PITCH/16),0,1)
    low=(MINOR+allowance)/2; high=(MAJOR+allowance)/2
    radius=low+(high-low)*groove
    # 45-degree entry lead-in, 0.65 mm beyond the groove-root radius.
    radius=np.maximum(radius,high+.65-np.maximum(zz-entry_z,0))
    if exit_z is not None:
        radius=np.maximum(radius,high+.65-np.maximum(exit_z-zz,0))
    vertices=np.stack([radius*np.cos(tt),radius*np.sin(tt),zz],axis=-1).reshape(-1,3)
    vertices=np.vstack([vertices,[0,0,z0],[0,0,z1]])
    faces=[]
    for k in range(nz-1):
        for j in range(ntheta):
            a=k*ntheta+j; b=k*ntheta+(j+1)%ntheta
            c=b+ntheta; d=a+ntheta
            faces.extend([[a,b,c],[a,c,d]])
    for j in range(ntheta):
        faces.append([nz*ntheta,(j+1)%ntheta,j])
        faces.append([nz*ntheta+1,(nz-1)*ntheta+j,(nz-1)*ntheta+(j+1)%ntheta])
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0
    return mesh

def verify_export(mesh,name,normalize=True):
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0
    assert count_bodies(mesh)==1
    mesh=mesh.copy()
    if normalize:mesh.apply_translation(-mesh.bounds[0])
    mesh.export(OUT/name)
    loaded=trimesh.load_mesh(OUT/name)
    assert loaded.is_watertight and loaded.is_winding_consistent
    return {'file':name,'bounds_mm':mesh.bounds.tolist(),'volume_mm3':float(mesh.volume),'watertight':True,'connected_bodies':1}

def main():
    results=[]
    for allowance in [.35,.55]:
        block=solid('box',[-7,7,-7,7,0,8])
        cutter=thread_tool(-.3,8.3,allowance,0,8)
        coupon=trimesh.boolean.difference([block,cutter],engine='manifold')
        results.append(verify_export(coupon,f'quarter-20-fit-allowance-{allowance:.2f}mm.stl'))
    existing=trimesh.load_mesh(ROOT/'snap-case-v2/back-assembled.ply')
    # Preserve the original camera seat and side latches. Extend the rear
    # downward so its bed-facing surface is coplanar across the shell.
    thickening=[solid('box',[-.4,29.4,-.4,18,-11,.1]),
                solid('box',[-.4,1.6,17.9,28.4,-11,.1]),
                solid('box',[27.4,29.4,17.9,28.4,-11,.1])]
    blank=trimesh.boolean.union([existing]+thickening,engine='manifold')
    for allowance in [.35,.55]:
        cutter=thread_tool(-11.3,-1.6,allowance,-11)
        cutter.apply_translation([14.5,8,0])
        back=trimesh.boolean.difference([blank,cutter],engine='manifold')
        # No original material is removed; all socket modifications are
        # behind the original shell and stop below its original rear face.
        lost=trimesh.boolean.difference([existing,back],engine='manifold')
        assert len(lost.faces)==0
        lid=trimesh.load_mesh(ROOT/'snap-case-v2/lid-assembled.ply')
        clash=trimesh.boolean.intersection([back,lid],engine='manifold')
        assert abs(clash.volume)<1e-5, f'Lid collision: {clash.volume} mm3'
        filename=f'camera-back-quarter-20-allowance-{allowance:.2f}mm.stl'
        results.append(verify_export(back,filename))
        back.export(OUT/f'back-assembled-{allowance:.2f}.ply')
    # Analytical thread sanity checks: pitch, radial depth and 60° flanks.
    assert abs(PITCH-1.27)<1e-10
    flank_angle=math.degrees(math.atan(((MAJOR-MINOR)/2)/(5*PITCH/16)))
    assert abs(flank_angle-60)<1e-8
    report={'thread':'1/4-20 right-hand, basic 60-degree internal profile with print allowance',
            'pitch_mm':PITCH,'nominal_major_mm':MAJOR,'basic_minor_mm':MINOR,
            'socket_depth_mm':9.4,'recommended_max_stud_projection_mm':8,
            'camera_seat_and_latches_unchanged':True,'assembled_lid_interference':False,
            'parts':results}
    (OUT/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
