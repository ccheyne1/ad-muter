"""Camera back v5: blind M2 pilot holes, existing v4 cover and tested tripod socket."""
from pathlib import Path
import argparse
import json
import math
from build_threaded_mount import trimesh, np, solid, count_bodies

ROOT = Path(__file__).resolve().parent
HOLES = [(4, 4), (25, 4), (4, 16), (25, 16)]
SEAT_Z = 4.6
PILOT_FLOOR_Z = -0.8
HEAD_SEAT_Z = 9.75


def frustum(x, y, bottom, top, lower_radius, upper_radius):
    segments = 64
    angles = np.arange(segments) * 2 * np.pi / segments
    vertices = [[x + radius * math.cos(a), y + radius * math.sin(a), z]
                for radius, z in ((lower_radius, bottom), (upper_radius, top)) for a in angles]
    vertices.extend([[x, y, bottom], [x, y, top]])
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend([[i, j, j + segments], [i, j + segments, i + segments],
                      [2 * segments, j, i], [2 * segments + 1, i + segments, j + segments]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    return mesh


def build(pilot_diameter=1.7):
    if not 1.5 <= pilot_diameter <= 1.85:
        raise ValueError('M2 pilot diameter must be between 1.5 and 1.85 mm')
    out = ROOT / 'screw-case-v5-direct'
    out.mkdir(exist_ok=True)
    original = trimesh.load_mesh(ROOT / 'threaded-mount/back-assembled-0.55.ply')
    # Use the uncut, physically tested rear shell, rather than filling nut cavities.
    pin_cuts = [solid('cylinder', [x, y, SEAT_Z, 1, 1]) for x, y in HOLES]
    back = trimesh.boolean.difference([original] + pin_cuts, engine='manifold')
    # Preserve v4's 6 mm diameter bearing faces. Continuous material underneath
    # permits longer engagement without raising the PCB or touching its components.
    bosses = [solid('cylinder', [x, y, -1, 3, SEAT_Z + 1]) for x, y in HOLES]
    back = trimesh.boolean.union([back] + bosses, engine='manifold')
    cuts = []
    for x, y in HOLES:
        cuts.append(solid('cylinder', [x, y, PILOT_FLOOR_Z, pilot_diameter / 2,
                                       SEAT_Z - PILOT_FLOOR_Z + .15]))
        cuts.append(frustum(x, y, SEAT_Z - .35, SEAT_Z + .15,
                            pilot_diameter / 2, 1.15))
    back = trimesh.boolean.difference([back] + cuts, engine='manifold')
    lid = trimesh.load_mesh(ROOT / 'screw-case-v4/lid-assembled.ply')
    clash = trimesh.boolean.intersection([back, lid], engine='manifold')
    assert abs(clash.volume) < 1e-5, 'Cover collides with back'

    # Verify the tested threaded socket's whole surrounding region is unchanged.
    socket_region = solid('cylinder', [14.5, 8, -11.1, 4.2, 9.6])
    before = trimesh.boolean.intersection([original, socket_region], engine='manifold')
    after = trimesh.boolean.intersection([back, socket_region], engine='manifold')
    socket_difference = (abs(trimesh.boolean.difference([before, after], engine='manifold').volume)
                         + abs(trimesh.boolean.difference([after, before], engine='manifold').volume))
    assert socket_difference < 1e-5, 'Tripod socket geometry changed'

    engagement = {}
    for length in (8, 10):
        tip = HEAD_SEAT_Z - length
        assert tip > PILOT_FLOOR_Z + .3, 'Screw would bottom out'
        engagement[str(length)] = {'engagement_mm': round(SEAT_Z - tip, 2),
                                   'tip_clearance_mm': round(tip - PILOT_FLOOR_Z, 2)}
    report = {'revision': 5, 'hole_spacing_mm': [21, 12], 'pilot_diameter_mm': pilot_diameter,
              'blind_pilot_depth_mm': SEAT_Z - PILOT_FLOOR_Z,
              'boss_diameter_mm': 6, 'straight_pilot_radial_wall_mm': 3 - pilot_diameter / 2,
              'screws': 'four M2 x 10 mm recommended; M2 x 8 mm also compatible',
              'engagement_by_screw_length_mm': engagement, 'nuts_required': 0,
              'seat_z_mm': SEAT_Z, 'pcb_thickness_mm': 1.7, 'cover_clearance_mm': .25,
              'quarter_20_allowance_mm': .55, 'socket_region_difference_mm3': socket_difference,
              'assembled_interference_mm3': abs(clash.volume), 'cover': 'exact v4 geometry', 'parts': []}
    for name, mesh in (('back', back), ('lid', lid)):
        assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
        assert count_bodies(mesh) == 1
        mesh.export(out / f'{name}-assembled.ply')
        printable = mesh.copy()
        if name == 'lid':
            printable.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        printable.apply_translation(-printable.bounds[0])
        filename = f'camera-{name}-m2-direct-v5.stl'
        printable.export(out / filename)
        exported = trimesh.load_mesh(out / filename)
        assert exported.is_watertight and exported.is_winding_consistent and count_bodies(exported) == 1
        report['parts'].append({'file': filename, 'size_mm': printable.extents.tolist(),
                                'volume_mm3': float(printable.volume), 'watertight': True,
                                'connected_bodies': 1, 'print_orientation': 'rear/outer flat face down'})
    (out / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-diameter', type=float, default=1.7)
    build(parser.parse_args().pilot_diameter)
