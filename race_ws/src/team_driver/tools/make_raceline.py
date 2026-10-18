#!/usr/bin/env python3
"""Regenerate the shipped racing line from the map alone (rule 32).

    python3 race_ws/src/team_driver/tools/make_raceline.py                      # compare with the shipped line
    python3 race_ws/src/team_driver/tools/make_raceline.py --output line.csv    # also write the regenerated line

Pipeline (no hand-placed points anywhere):
  1. plan_raceline.generate(): rebuild the reference lap from maps/icra26.* and maps/tracks.yaml
     with scripts/track_tool.py, seal cone rows into barriers, then optimise normal offsets to
     minimise integrated curvature^2 (+ curvature variation, offset and length terms) inside the
     corridor that keeps the whole car body >= MARGIN from walls/cones.
  2. Resample the result with a periodic arc-length cubic spline at 1 cm (fast_tracking.spline_closed),
     which is what the driver loads.

Parameters were chosen by simulated lap time and robustness (tools/headless_sim, then ROS runs),
not by editing points: MARGIN 0.20 m, LENGTH_WEIGHT 0.8, MAX_OFFSET 2.5 m, COARSE_STRIDE 6.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))           # team_driver package (fast_tracking)
REFEREE = HERE.parents[1]/'roboracer_referee'  # track_tool needs roboracer_referee.tracks
if REFEREE.exists():
    sys.path.append(str(REFEREE))
import plan_raceline as pr                      # noqa: E402
from team_driver.fast_tracking import spline_closed  # noqa: E402

MARGIN, LENGTH_WEIGHT, MAX_OFFSET, COARSE_STRIDE = 0.20, 0.8, 2.5, 6
SHIPPED = HERE.parent/'config'/'raceline_sim_optimized_v3.csv'
TOLERANCE = 0.08   # m; see README: the optimiser's stopping point depends on the CPU's floating-point paths


def build():
    _, candidate, _, report = pr.generate(pr.Geometry(margin_m=MARGIN, max_offset_m=MAX_OFFSET),
                                          LENGTH_WEIGHT, COARSE_STRIDE)
    candidate = np.round(candidate, 6)          # same precision as the intermediate file used originally
    _, dense, _, _, length = spline_closed(candidate, ds=0.01)
    return dense, length, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, default=None, help='write the regenerated line here (never overwritten by default)')
    ap.add_argument('--check', action='store_true', help='compare only (the default; kept for compatibility)')
    args = ap.parse_args()
    dense, length, report = build()
    print(f'generated: {len(dense)} points, length {length:.3f} m, retained offset fraction '
          f'{report["retained_offset_fraction"]:.2f}, body margin >= {MARGIN} m')
    if SHIPPED.exists():
        shipped = np.loadtxt(SHIPPED, delimiter=',')
        if shipped.shape == dense.shape and np.array_equal(np.round(dense, 6), shipped):
            print(f'{SHIPPED.name}: identical (0.000 mm)')
        else:
            from scipy.spatial import cKDTree
            a, b = cKDTree(shipped).query(dense)[0], cKDTree(dense).query(shipped)[0]
            dev = max(a.max(), b.max())
            print(f'{SHIPPED.name}: {len(shipped)} vs {len(dense)} points, distance between the lines '
                  f'max {dev*1000:.1f} mm, mean {a.mean()*1000:.1f} mm '
                  f'({"within" if dev <= TOLERANCE else "OUTSIDE"} the {TOLERANCE*1000:.0f} mm tolerance)')
    if args.output:
        np.savetxt(args.output, dense, delimiter=',', fmt='%.6f')
        print('wrote', args.output)


if __name__ == '__main__':
    main()
