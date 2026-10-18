# Team AftergIOw

## Members

| Name | Matric number | Email |
| --- | --- | --- |
| Shen Haoyang | U2620317D | shen0279@e.ntu.edu.sg |
| Peng Junxi | U2620547C | peng0218@e.ntu.edu.sg |
| Deng Beiang | U2620617F | deng0170@e.ntu.edu.sg |
| Wang Hanyu | U2620407C | wa0021yu@e.ntu.edu.sg |

## Approach

    Our final driver follows a precomputed racing line. It uses a pure pursuit controller to calculate steering commands. The lookahead distance increases with vehicle speed. This approach allows the driver to use the known circuit layout and its current position to anticipate upcoming corners.

    The driver builds a speed profile from the line curvature and selected lateral acceleration limits. Forward and backward passes account for acceleration and braking constraints. We apply lower limits in selected track segments and use a gentler profile during the initial untimed laps. A short forward position prediction is also used to compensate for control delay.

    The approach of follow the gap are also implemented, while follow the gap is less stable in fast-velocity situations. The inability to process the lidar input with a more accurate approach leads to this situation.

## How it uses the inputs

- LiDAR: The final approach does not use LiDAR scans or subscribe to it.
- Odometry:  We use the simulator's ground-truth position, orientation and velocity from `/ego_racecar/odom`. These provide the vehicle's location, heading and speed for path tracking and speed control. Each incoming odometry message triggers a control update.
- Map: The circuit is represented in the controller by the precomputed racing line in `race_ws/src/team_driver/config/raceline_sim_optimized_v3.csv`, which our code generates offline from the map files (`maps/icra26.yaml`/`.png` and `maps/tracks.yaml`); see *Anything precomputed*. The driver loads this file at startup and calculates its curvature and speed profile. It does not subscribe to `/map` or generate a new racing line during the run.

## Results on our own machine



| Best lap | 10-lap total | Collisions | Runs attempted |
| --- | --- | --- | --- |
| 12.0199 | 120.2047 | 0 | 2 |

### The machine that produced them

| CPU | GPU | RAM | OS | Typical real-time factor |
| --- | --- | --- | --- | --- |
| Intel Core Ultra 9 275HX | NVIDIA GeForce RTX 5060 Laptop GPU (8 GB) | 16 GB | Windows 11 Home; Ubuntu 24.04.5 LTS via WSL2; Docker | 2.38–2.77× |

Hardware strength, software versions and ROS message timing may contribute to differences between machines, but we have not isolated their effects. 

## Anything precomputed

**The racing line** (`race_ws/src/team_driver/config/raceline_sim_optimized_v3.csv`) is generated from the map alone by `race_ws/src/team_driver/tools/plan_raceline.py`; no waypoint was placed or edited by hand. The pipeline: rebuild the reference lap from the map with `scripts/track_tool.py`, link cone rows into barriers (rule 11), compute for each point how far the line may move sideways while the car keeps a 0.20 m margin from walls and cones, minimise integrated curvature squared (plus curvature variation, offset and length terms) with L-BFGS-B inside that corridor, check the full car footprint, and resample with a periodic cubic spline at 1 cm. Details are in `race_ws/src/team_driver/tools/README.md`.

To regenerate it and compare with the shipped file (inside the container, about 5 s, overwrites nothing):

    python3 race_ws/src/team_driver/tools/make_raceline.py

On the x86-64 machine where the shipped file was generated, the output is identical. On other CPUs the optimiser stops at a slightly different iteration because of floating-point differences between SIMD code paths, so the regenerated line is within about 6 cm of the shipped one (57 mm max, 14 mm mean in the judging image on Apple Silicon); the measurements are in the tools README.

The four generator parameters (margin 0.20 m, length weight 0.8, maximum offset 2.5 m, coarse stride 6) were chosen by generating candidates and comparing them in simulation for lap time and robustness to control delay.

We also tested combinations of controller parameters, including the lateral acceleration limit, per-segment speed scales, the warm-up speed scale, the lookahead distance and the pose prediction time, and selected the final configuration based on lap times and collision counts. The chosen values are stored in the driver defaults and configuration files. During evaluation, the driver uses this configuration at startup.

## Dependencies we added

none

## Third-party code and references

none

## AI assistance

Claude, Codex and ChatGPT were used for assistance in building the testing environment, giving feedback for codes and testing the running result in certain cases. AI tools also assisted with generating or modifying the follow the gap algorithm, and Claude assisted with the racing-line generator, the fast pure pursuit tracker and speed profile, and diagnosing control latency in the ROS runs. Team members reviewed these changes and validated their behaviour through simulation tests.

## Known issues

A more stable route tracking algorithm should be carried out as the car is shaky in high speed. Also, we haven't had the time for further algorithms like combining pure pursuit with follow the gap and else.

The tightest hairpin needs steering right at the car's 0.419 rad limit, so pure pursuit cuts it a little. In roughly 1 in 10-20 of our Mac runs the car clipped the last cone of that hairpin near (4.5, 14.85) and could not recover. We did not ship our recovery attempt because it reversed into the same cone; with more time we would adjust the line there or add a slow reverse-then-crawl recovery.
