"""Common geometry specification for STL generation and Autodesk Fusion, mm.

The PCB thickness must be measured on the bare board at a mounting hole.
This first enclosure prototype tests retention; the tripod adapter is separate.
"""
def geometry(pcb_thickness):
    if not 0.5 <= pcb_thickness <= 6:
        raise ValueError('Confirm PCB thickness in mm (0.5 to 6).')
    clearance = 0.25
    seat_z = 4.6
    lid_z = seat_z + pcb_thickness + clearance
    # [operation, kind, name, dimensions]
    back, lid = [], []
    def box(part,name,x0,x1,y0,y1,z0,z1,op='add'):
        part.append([op,'box',name,[x0,x1,y0,y1,z0,z1]])
    def cyl(part,name,x,y,z,r,h,op='add'):
        part.append([op,'cylinder',name,[x,y,z,r,h]])
    box(back,'Back plate',-.4,29.4,-.4,28.4,0,1.6)
    for i,(x,y) in enumerate([(4,4),(25,4),(4,16),(25,16)],1):
        cyl(back,f'Support {i}',x,y,1.5,1.5,3.1)
        cyl(back,f'Locating pin {i}',x,y,4.5,.8,.9)
    box(back,'Left wall',-.4,1.6,-.4,28.4,1.5,lid_z)
    box(back,'Right wall',27.4,29.4,-.4,28.4,1.5,lid_z)
    # Keep the upper edge below the PCB front for an overhanging lens.
    box(back,'Low upper wall',-.4,29.4,-.4,1.6,1.5,4.1)
    # Broad bottom opening above this wall clears the ribbon connector.
    box(back,'Low ribbon wall',-.4,29.4,26.4,28.4,1.5,2.6)
    box(back,'Left latch window',-.5,.85,13.7,18.3,1.9,3.65,'cut')
    box(back,'Right latch window',28.15,29.5,13.7,18.3,1.9,3.65,'cut')
    # Connector-inclusive thickness is 6 mm, greater than the 3 mm support
    # clearance. Open the rear below the lower hole row to avoid trapping it.
    box(back,'Ribbon connector rear opening',1.6,27.4,18,28.5,-.1,4.2,'cut')

    # U-shaped front frame. Open centre and open top avoid requiring an
    # estimated lens-centre coordinate. Only four hole-area pads touch PCB.
    frame_thickness = 3.2
    box(lid,'Lower frame bridge',-.4,29.4,26.7,31.2,lid_z,lid_z+frame_thickness)
    box(lid,'Left frame rail',-.4,2.5,-4.4,31.2,lid_z,lid_z+frame_thickness)
    box(lid,'Right frame rail',26.5,29.4,-4.4,31.2,lid_z,lid_z+frame_thickness)
    for y in [4,16]:
        box(lid,f'Left retention pad {y}',1.5,5.5,y-1.5,y+1.5,lid_z,lid_z+frame_thickness)
        box(lid,f'Right retention pad {y}',23.5,27.5,y-1.5,y+1.5,lid_z,lid_z+frame_thickness)
    # Cantilevers flex sideways in the print's XY plane. Hooks descend at
    # their free ends; pulling both free ends outward releases the lid.
    box(lid,'Left spring root',-3,1,-4.4,-2,lid_z,lid_z+frame_thickness)
    box(lid,'Right spring root',28,32,-4.4,-2,lid_z,lid_z+frame_thickness)
    box(lid,'Left spring beam',-3,-1.4,-3,18,lid_z,lid_z+frame_thickness)
    box(lid,'Right spring beam',30.4,32,-3,18,lid_z,lid_z+frame_thickness)
    box(lid,'Left latch leg',-3,-1.4,14,18,2.3,lid_z+.1)
    box(lid,'Right latch leg',30.4,32,14,18,2.3,lid_z+.1)
    # XZ triangular ramp extruded along Y. The short flat upper ledge
    # catches in the wall recess with 0.15 mm axial clearance.
    lid.append(['add','wedge','Left latch ramp',[[-1.5,2.3],[.1,3.5],[-1.5,3.5],14,18]])
    lid.append(['add','wedge','Right latch ramp',[[30.5,2.3],[28.9,3.5],[30.5,3.5],14,18]])
    # Buttress the print-down underside of each hook at 45 degrees. This
    # reduces its unsupported horizontal ledge from 1.5 mm to 0.5 mm while
    # retaining the original tip and latch ledge positions.
    lid.append(['add','wedge','Left hook print buttress',[[-1.5,4.6],[-.4,3.5],[-1.5,3.5],14,18]])
    lid.append(['add','wedge','Right hook print buttress',[[30.5,4.6],[29.4,3.5],[30.5,3.5],14,18]])
    return {'pcb_thickness_mm':pcb_thickness,'lid_bottom_z':lid_z,
            'clearance_mm':clearance,'revision':2,'frame_thickness_mm':frame_thickness,
            'back':back,'lid':lid}
