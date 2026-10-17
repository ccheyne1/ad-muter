"""TSAL6200 enclosure: support-free split shell and adjustable M2 cable clamp.

All geometry is in mm. Run with Python 3.12 and the packages in requirements-cad.txt.
The supplied STL files are already oriented on their flat exterior faces.
"""
from pathlib import Path
import sys, math, json, zipfile
from itertools import combinations
from xml.etree.ElementTree import Element, SubElement, tostring, fromstring

ROOT = Path(__file__).resolve().parent
import numpy as np
import trimesh

OUT = ROOT / 'ir-diode-case-v1'
W, L, H = 28.0, 50.0, 16.0
WALL, FLOOR, SPLIT = 2.4, 2.0, 8.0
LED_X = W / 2
BODY_D, FLANGE_D, LEAD_D = 5.4, 6.2, 4.6
FLANGE_FRONT, FLANGE_BACK = 2.0, 3.6
SCREW_D, NUT_AF = 2.3, 4.4
CASE_SCREWS = [(3.6, 7.5), (W-3.6, 7.5), (3.6, L-3.6), (W-3.6, L-3.6)]
CLAMP_SCREWS = [(7.0, 40.0), (21.0, 40.0)]
CLAMP_Z = 5.0  # Preview position: underside of clamp, adjustable in real assembly.
CLAMP_T = 2.8


def box(x0, x1, y0, y1, z0, z1):
    m = trimesh.creation.box([x1-x0, y1-y0, z1-z0])
    m.apply_translation([(x0+x1)/2, (y0+y1)/2, (z0+z1)/2])
    return m


def cylinder(x, y, z0, z1, radius, sections=96):
    m = trimesh.creation.cylinder(radius=radius, height=z1-z0, sections=sections)
    m.apply_translation([x, y, (z0+z1)/2])
    return m


def axial_y(x, z, y0, y1, diameter):
    m = trimesh.creation.cylinder(radius=diameter/2, height=y1-y0, sections=96)
    m.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [1, 0, 0]))
    m.apply_translation([x, (y0+y1)/2, z])
    return m


def union(items):
    return trimesh.boolean.union(items, engine='manifold')


def subtract(base, cuts):
    return trimesh.boolean.difference([base] + cuts, engine='manifold')


def intersection(a, b):
    return trimesh.boolean.intersection([a, b], engine='manifold')


def count_bodies(mesh):
    parents = list(range(len(mesh.vertices)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for a, b in mesh.edges:
        parents[find(int(a))] = find(int(b))
    return len({find(i) for i in range(len(parents))})


def rounded_box(x0, x1, y0, y1, z0, z1, radius):
    return union([
        box(x0+radius, x1-radius, y0, y1, z0, z1),
        box(x0, x1, y0+radius, y1-radius, z0, z1),
        *[cylinder(x, y, z0, z1, radius)
          for x in (x0+radius, x1-radius) for y in (y0+radius, y1-radius)]
    ])


def nut_tool(x, y, z0, z1):
    return cylinder(x, y, z0, z1, NUT_AF / math.sqrt(3), sections=6)


def print_orientation(mesh, flip=False):
    m = mesh.copy()
    if flip:
        m.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]))
    m.apply_translation(-m.bounds[0])
    return m


def export_layout(parts):
    """Standard model-only 3MF, explicitly millimetres; no slicer settings."""
    namespace = 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'
    model = Element('model', {'unit': 'millimeter', 'xmlns': namespace})
    resources = SubElement(model, 'resources')
    build = SubElement(model, 'build')
    for i, (name, mesh) in enumerate(parts.items(), 1):
        obj = SubElement(resources, 'object', {'id': str(i), 'type': 'model', 'name': name})
        mnode = SubElement(obj, 'mesh')
        vertices = SubElement(mnode, 'vertices')
        triangles = SubElement(mnode, 'triangles')
        for vertex in mesh.vertices:
            SubElement(vertices, 'vertex', dict(zip(('x','y','z'), (f'{x:.6f}' for x in vertex))))
        for face in mesh.faces:
            SubElement(triangles, 'triangle', dict(zip(('v1','v2','v3'), (str(int(x)) for x in face))))
        SubElement(build, 'item', {'objectid': str(i), 'transform': f'1 0 0 0 1 0 0 0 1 {10+(i-1)*36} 10 0'})
    with zipfile.ZipFile(OUT / 'ir-diode-v1-print-layout.3mf', 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        archive.writestr('_rels/.rels', '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        archive.writestr('3D/3dmodel.model', tostring(model, encoding='utf-8', xml_declaration=True))


def build():
    OUT.mkdir(exist_ok=True)
    shell = subtract(rounded_box(0, W, 0, L, 0, H, 3.0), [
        box(WALL, W-WALL, 5, L-WALL, FLOOR, H-FLOOR)
    ])
    shell = union([shell] + [cylinder(x,y,0,H,3.5) for x,y in CASE_SCREWS])
    shell = subtract(shell, [
        axial_y(LED_X, SPLIT, -.2, FLANGE_FRONT+.01, BODY_D),
        axial_y(LED_X, SPLIT, FLANGE_FRONT, FLANGE_BACK+.01, FLANGE_D),
        axial_y(LED_X, SPLIT, FLANGE_BACK, 5.2, LEAD_D),
        box(LED_X-4.5, LED_X+4.5, L-WALL-.1, L+.1, SPLIT-2.5, SPLIT+2.5),
        *[cylinder(x,y,3.5,H+.1,SCREW_D/2) for x,y in CASE_SCREWS]
    ])
    base = intersection(shell, box(-1,W+1,-1,L+1,-1,SPLIT))
    lid = intersection(shell, box(-1,W+1,-1,L+1,SPLIT,H+1))
    base = subtract(base, [nut_tool(x,y,5.6,SPLIT+.1) for x,y in CASE_SCREWS])

    # Low captive-nut posts and a raised, broad wire saddle. No PCB required.
    base = union([base,
        rounded_box(9.5,18.5,36.5,43.5,FLOOR-.1,3.0,.8),
        *[cylinder(x,y,FLOOR-.1,4.2,3.2) for x,y in CLAMP_SCREWS]
    ])
    base = subtract(base, [
        *[nut_tool(x,y,2.0,4.3) for x,y in CLAMP_SCREWS],
        *[cylinder(x,y,-.1,4.3,SCREW_D/2) for x,y in CLAMP_SCREWS]
    ])
    clamp = subtract(rounded_box(3.7,24.3,37.0,43.0,CLAMP_Z,CLAMP_Z+CLAMP_T,1.2),
                     [cylinder(x,y,CLAMP_Z-.1,CLAMP_Z+CLAMP_T+.1,SCREW_D/2) for x,y in CLAMP_SCREWS])
    assembled = {'base': base, 'lid': lid, 'cable-clamp': clamp}
    printed = {}
    report = {
        'version': 1, 'units': 'mm', 'body_envelope_mm': [W,L,H],
        'component_space_mm': [20,24,12], 'component_space_bounds_mm': [[4,11,2],[24,35,14]],
        'led': {'part': 'Vishay TSAL6200', 'body_bore_mm': BODY_D, 'flange_pocket_diameter_mm': FLANGE_D,
                'flange_pocket_depth_mm': FLANGE_BACK-FLANGE_FRONT, 'lead_passage_mm': LEAD_D,
                'axis_xyz_mm': [LED_X,0,SPLIT], 'datasheet': 'https://www.vishay.com/docs/81010/tsal6200.pdf'},
        'cable_exit_mm': [9,5], 'clamp_clear_width_mm': 7.6,
        'nut_pocket_across_flats_mm': NUT_AF,
        'hardware': {'lid': '4 x M2 x 12 mm + 4 M2 hex nuts',
                     'clamp': '2 x M2 x 6 mm for flat wires; 8 mm for a thicker bundle + 2 M2 hex nuts'},
        'physical_fit_tested': False, 'parts': [], 'collision_checks': []
    }
    for name, mesh in assembled.items():
        assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0, name
        assert count_bodies(mesh) == 1, name
        mesh.export(OUT / f'{name}-assembled.ply')
        p = print_orientation(mesh, flip=(name=='lid'))
        filename = f'ir-diode-{name}-v1.stl'
        p.export(OUT / filename)
        reload = trimesh.load_mesh(OUT / filename)
        assert reload.is_watertight and reload.is_winding_consistent and count_bodies(reload) == 1
        # All significant downward facets must be the flat bed face: no unsupported ledges.
        down = (p.face_normals[:,2] < -.7072) & (p.triangles_center[:,2] > .001)
        overhang_area = float(p.area_faces[down].sum())
        assert overhang_area < .01, (name, overhang_area)
        printed[name] = p
        report['parts'].append({'name': name, 'file': filename, 'size_mm': p.extents.tolist(),
                                'volume_mm3': float(p.volume), 'watertight': True, 'connected_bodies': 1,
                                'faces': len(p.faces), 'unsupported_downward_area_mm2': overhang_area})
    for (a,ma),(b,mb) in combinations(assembled.items(),2):
        clash = abs(float(intersection(ma,mb).volume))
        assert clash < .001, (a,b,clash)
        report['collision_checks'].append({'parts':[a,b], 'overlap_mm3':clash})
    # A conservative maximum-size emitter envelope, with 1.2 mm flange thickness.
    led = union([axial_y(LED_X,SPLIT,-5.5,2.2,5.15), axial_y(LED_X,SPLIT,2.2,3.4,5.95)])
    led.export(OUT / 'led-envelope-reference.ply')
    for name in ('base','lid'):
        clash = abs(float(intersection(assembled[name],led).volume))
        assert clash < .001, ('LED envelope',name,clash)
        report['collision_checks'].append({'parts':['maximum_LED_envelope',name], 'overlap_mm3':clash})
    # Verify room for two individually insulated leads, nut engagement and the driver bundle.
    leads = union([axial_y(LED_X+dx,SPLIT,3.4,10,1.8) for dx in (-1.27,1.27)])
    component_space = box(4,24,11,35,2.01,13.99)
    for obstacle_name, obstacle in [('insulated_LED_leads',leads),('driver_space',component_space)]:
        for name in ('base','lid'):
            clash = abs(float(intersection(assembled[name],obstacle).volume))
            assert clash < .001, (obstacle_name,name,clash)
            report['collision_checks'].append({'parts':[obstacle_name,name], 'overlap_mm3':clash})
    for x,y in CASE_SCREWS:
        screw_shaft = cylinder(x,y,4,H,1.0)
        nut = cylinder(x,y,5.6,7.2,4.0/math.sqrt(3), sections=6)
        nut = subtract(nut,[cylinder(x,y,5.5,7.3,1.0)])
        assert abs(intersection(base,screw_shaft).volume) < .001
        assert abs(intersection(lid,screw_shaft).volume) < .001
        assert abs(intersection(base,nut).volume) < .001
    report['case_screw_check'] = '12 mm shafts fit; full 1.6 mm nominal nut engagement; 0.5 mm tip clearance'
    # Clamp screws at the preview position (2 mm-high wire bundle), with full nut engagement.
    for x,y in CLAMP_SCREWS:
        shaft = cylinder(x,y,CLAMP_Z+CLAMP_T-6,CLAMP_Z+CLAMP_T,1.0)
        assert abs(intersection(base,shaft).volume) < .001
        assert abs(intersection(clamp,shaft).volume) < .001
    export_layout(printed)
    # Round-trip the 3MF to verify all print parts and their orientation survived export.
    with zipfile.ZipFile(OUT / 'ir-diode-v1-print-layout.3mf') as archive:
        root = fromstring(archive.read('3D/3dmodel.model'))
    ns = {'m': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
    assert root.attrib['unit'] == 'millimeter'
    objects = root.findall('m:resources/m:object', ns)
    assert len(objects) == 3 and len(root.findall('m:build/m:item', ns)) == 3
    for obj in objects:
        vertices = [[float(v.attrib[c]) for c in ('x','y','z')]
                    for v in obj.findall('m:mesh/m:vertices/m:vertex',ns)]
        faces = [[int(t.attrib[c]) for c in ('v1','v2','v3')]
                 for t in obj.findall('m:mesh/m:triangles/m:triangle',ns)]
        reloaded = trimesh.Trimesh(vertices=vertices, faces=faces)
        assert reloaded.is_watertight and count_bodies(reloaded) == 1
        assert reloaded.bounds[0,2] >= -.00001
    report['model_3mf_roundtrip'] = {'objects': 3, 'all_watertight': True, 'units': 'millimeter'}
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return assembled, led


if __name__ == '__main__':
    build()
