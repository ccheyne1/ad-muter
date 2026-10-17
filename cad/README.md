# Printable enclosures

Dimensions are millimetres. Import the STLs at 100% scale into your slicer or Fusion. Review printer/material settings yourself; no machine-specific profiles or G-code are distributed.

- `screw-case-v5-direct`: camera back and rigid cover, 21 x 12 mm PCB hole spacing, 1.7 mm PCB thickness, 17 mm lens diameter, 16 mm lens projection. The back uses blind 1.7 mm M2 pilot holes rather than nuts. Use four M2 x 10 mm screws (8 mm also fits); longer screws can bottom out. Form threads gently in the empty housing before inserting the board. PETG, 0.2 mm layers, four walls, 20% infill, supplied flat orientations, supports off. The preserved printed 1/4-20 tripod socket uses 0.55 mm diametral allowance; keep the mount stud projection at or below 8 mm. Actual fit depends on your printer and camera.
- `ir-diode-case-v1`: TSAL6200 LED pod with insulated driver space and cable clamp. Use four M2 x 12 mm case screws, two M2 x 6 mm clamp screws and six M2 nuts. PETG, 0.2 mm layers, four walls, supports off. First-print fit is unverified.
- `ir-holder`: loose dual holder for 19.5 x 23 x 7 mm receiver/transmitter modules. Bays are 20.5 x 24 x 8 mm. Use four M2 x 6 mm screws and nuts. Optical ends and wiring remain exposed. Physical fit is unverified.

To edit geometry, install `pip install -r requirements-cad.txt`, then run a generator from the repository root, for example `python cad/build_direct_screw_case.py --pilot-diameter 1.7`. The included PLY files are intermediate geometry needed by the generators. Older snap geometry is retained only as a generator dependency; use the v5 direct-screw STLs for the current camera enclosure. No native Fusion timeline is included.
