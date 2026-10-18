# Racing line generation (rule 32)

The line the driver loads, `config/raceline_sim_optimized_v3.csv`, is produced entirely by code in this folder
from the map files. No waypoint was placed or edited by hand.

Regenerate and compare with the shipped file (inside the container, from the repo root; nothing is overwritten):

```sh
python3 race_ws/src/team_driver/tools/make_raceline.py                     # regenerate and compare
python3 race_ws/src/team_driver/tools/make_raceline.py --output line.csv   # also save the regenerated line
```

It takes about 5 s. On an x86-64 machine with AVX-512 (where the shipped file was generated) the output is
`7409 points, length 74.093 m` and `identical (0.000 mm)`. On other CPUs see *Reproducibility* below.

## Pipeline

1. **Reference lap from the map** (`plan_raceline.reference_from_map`): `scripts/track_tool.py` reads
   `maps/icra26.yaml`/`.png` and `maps/tracks.yaml` and traces the lap in the official direction.
2. **Barriers**: cone rows are linked into continuous walls (rule 11), so the corridor never passes between cones.
3. **Corridor**: for each reference point, how far the line may move left/right along the normal before the car's
   half-width plus a 0.20 m margin would touch a wall or cone (capped at 2.5 m).
4. **Optimisation**: L-BFGS-B over the normal offsets, minimising integrated curvature², curvature variation and
   offset size plus a length term (weight 0.8), inside the corridor. It is solved on every 6th point first, and that
   solution seeds the full-resolution solve.
5. **Body check**: the offsets are scaled back towards the reference until the whole car body (three covering discs)
   keeps the 0.20 m margin everywhere; for this map 80 % of the optimised offset is kept.
6. **Resample**: periodic arc-length cubic spline at 1 cm (`team_driver.fast_tracking.spline_closed`).

The four parameters (margin 0.20 m, length weight 0.8, max offset 2.5 m, stride 6) were chosen by comparing
generated candidates in simulation for lap time and robustness to control delay; they are constants in
`make_raceline.py`.

Dependencies: numpy, scipy and pyyaml, already present in the judging image (installed with the simulator).

## Reproducibility

Same code, same map, same Python 3.12.3 / numpy 2.5.3 / scipy 1.18.1 (the versions the judging image installs):

| Machine | Points | Length | Max distance from shipped line |
| --- | --- | --- | --- |
| x86-64 with AVX-512 (where the shipped line was generated) | 7409 | 74.093 m | identical |
| x86-64, numpy AVX-512 paths disabled (like a desktop i9) | 7415 | 74.154 m | 54.9 mm |
| x86-64, OpenBLAS forced to Haswell kernels | 7414 | 74.135 m | 22.5 mm |
| Apple Silicon, inside the judging Docker image | 7404 | 74.036 m | 57.1 mm (mean 13.8 mm) |

The difference is not randomness: each machine gives the same result every time. The full-resolution L-BFGS-B
solve takes thousands of iterations and stops on a relative-decrease test while the objective is still creeping
down, so rounding differences between SIMD code paths change the iteration at which it stops. `make_raceline.py` therefore
reports the distance between the regenerated and shipped lines and accepts up to 80 mm. Every regenerated variant
passes the same 0.20 m body-margin check before it is written (`generate` raises otherwise).

We tried running the solver to full convergence (a truncated-Newton polish after L-BFGS-B). The result then agrees
to within 5 mm on every CPU path above, but at these parameters that converged line was slightly slower in
simulation and touched a wall in two of our jitter tests, so the shipped line is the one tested in ROS.
