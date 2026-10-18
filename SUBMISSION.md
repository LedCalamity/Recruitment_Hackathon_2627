# Team AftergIOw

## Members

| Name | Matric number | Email |
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
- Map: The circuit is represented in the controller by the precomputed racing line in `raceline_sim_optimized_v3.csv`. The driver loads this file at startup and calculates its curvature and speed profile. It does not subscribe to `/map` or generate a new racing line during the run.

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

We generated an improved racing line offline. The resulting coordinates are stored in raceline_sim_optimized_v3.csv and loaded by the driver at startup.

We also tested combinations of controller parameters including max_speed, min_speed and look_ahead and selected the final configuration based on lap times and collision counts. The chosen values are stored in the driver defaults and configuration files. During evaluation, the driver uses this configuration at startup.

## Dependencies we added

none

## Third-party code and references

none

## AI assistance

Codex and ChatGPT work are used for assistance in building the testing environment, giving feedback for codes and testing the running result in certain cases. AI tools also assisted with generating or modifying follow the gap algorithm. Team members reviewed these changes and validated their behaviour through simulation tests.

## Known issues

A more stable route tracking algorithm should be carried out as the car is shaky in high speed. Also, we haven't had the time for further algorithms like combining pure pursuit with follow the gap and else.
