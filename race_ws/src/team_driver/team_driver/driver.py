#!/usr/bin/env python3
"""YOUR DRIVER GOES HERE.

This is the node the judges run. Keep `driver` as the executable name and
`/drive` as the output topic and everything else is yours to change - rewrite
this file completely if you want to.

--------------------------------------------------------------------------
THIS TEMPLATE DOES NOT DRIVE
--------------------------------------------------------------------------
It is wiring, not a driver. It connects to the simulator, subscribes to the
sensors, and then asks for a slow constant speed with the wheels straight. It
will set off from the grid and into the first thing in front of it. That is
deliberate and it is the whole point: **there is no algorithm here and no
algorithm is shipped anywhere else in this repository.** Writing one is the
hackathon.

What the template is good for is proving your setup works. If the car moves
when you run it, then the image, the bridge, the workspace, the topics and
your commands are all correct, and every problem left is yours.

    ros2 run team_driver driver
    ros2 launch team_driver driver.launch.py

`docs/04-algorithms.md` lists the approaches worth starting from - reactive
ones that need nothing but the LiDAR, planners that follow a line, model-based
control, and learned policies - with what each needs and where each breaks.
Pick one and replace `plan()` below.

What you are allowed to read (see docs/06-rules.md):
    /scan               LiDAR, 819 beams over 270 degrees
    /ego_racecar/odom   ground-truth pose and velocity - ALLOWED and RECOMMENDED
    TF, /map            the static map
What you publish:
    /drive              AckermannDriveStamped - you ask for a SPEED, and the
                        simulator closes that loop for you
    /driver/...         anything of your own, for visualisation
"""

import math

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from team_driver.fast_tracking import Tracker as FastTracker


class Driver(Node):

    def __init__(self):
        super().__init__('driver')

        # Declared parameters can be retuned without editing code:
        #   ros2 run team_driver driver --ros-args -p crawl_speed:=2.0
        # Add your own as you go; config/driver_params.yaml loads them.
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('crawl_speed', 1.0)        # [m/s]
        self.declare_parameter('max_range', 8.0)          # [m] clip the scan here

        self.crawl_speed = self.get_parameter('crawl_speed').value
        self.max_range = self.get_parameter('max_range').value

        """ver3"""
        # Teammate's line for the 'pp' controller. Relative path, so only works when started from
        # the repo root; it must not crash the node when the fast controller is used.
        try:
            import os
            self.path = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'config', 'teammate',
                                                'icra26_f1_collision_adjusted_xy.csv'), delimiter=",", comments="#")
        except OSError:
            self.path = None

        """testing done here"""
        self.declare_parameter('look_ahead',0.9)
        self.declare_parameter('max_speed',7)
        self.declare_parameter('turn_slowdown', 3.5)
        self.declare_parameter('min_speed', 3.5)

        self.look_ahead = self.get_parameter('look_ahead').value
        self.max_speed = self.get_parameter('max_speed').value
        self.turn_slowdown = self.get_parameter('turn_slowdown').value
        self.min_speed = self.get_parameter('min_speed').value

        # controller: 'pp' = plan_pure_persuit below (default, unchanged behaviour);
        # 'fast' = raceline tracker validated in the headless sim (config/driver_params_fast.yaml).
        # Default = 'fast' with the v3 line (0.20 m body margin), fast_alat 16 with slower hairpins and
        # right-side approach: this is what `ros2 run team_driver driver` (the judges' command) runs.
        # ROS headless 10-lap checks: 12.013-12.019 s best, 120.20 s total, 8/9 runs clean.
        self.declare_parameter('controller', 'fast')
        self.declare_parameter('path_file', '')
        for name, default in [('fast_alat', 16.0), ('fast_vmax', 10.0), ('fast_acc', 9.0), ('fast_brk', 9.0),
                              ('fast_la0', .506), ('fast_la_k', .1), ('fast_la_min', .3), ('fast_tau', .05)]:
            self.declare_parameter(name, default)
        # Per-corner grip scale (comma-separated strings to keep ROS param typing simple).
        # bounds = segment start stations along the line [m]; scale = multiplier on fast_alat per segment.
        # Default: right-side approach (s 22-29 m) at alat 15, both hairpins at 0.85 x 16 = 13.6.
        self.declare_parameter('fast_seg_bounds', '0.0,22.0,29.0,37.5,44.3')
        self.declare_parameter('fast_seg_scale', '1.0,0.9375,0.85,0.85,1.0')
        # Out lap + warm-up lap are not timed: drive them at fast_warm_scale x grip (1.0 = off).
        self.declare_parameter('fast_warm_scale', 0.8)
        self.fast = None
        if self.get_parameter('controller').value == 'fast':
            g = lambda n: float(self.get_parameter(n).value)
            line = np.loadtxt(self._resolve_line(str(self.get_parameter('path_file').value)),
                              delimiter=",", comments="#")
            self.fast = FastTracker(line, alat=g('fast_alat'), vmax=g('fast_vmax'), acc=g('fast_acc'),
                                    brk=g('fast_brk'), la0=g('fast_la0'), la_k=g('fast_la_k'),
                                    la_min=g('fast_la_min'), tau=g('fast_tau'),
                                    seg_bounds=self._floats('fast_seg_bounds'), seg_scale=self._floats('fast_seg_scale'),
                                    warm_scale=g('fast_warm_scale'))
            self.get_logger().info(f'FAST raceline tracking: est lap {self.fast.est:.2f}s, line {self.fast.L:.2f} m')

        # Latest known pose and speed. Ground truth from the simulator, which
        # the rules allow you to use - so use it.
        self.position = None      # (x, y) in the map frame
        self.yaw = 0.0            # [rad]
        self.speed = 0.0
        self.last_steering = 0.0          # [m/s]

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.get_parameter('drive_topic').value, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/driver/markers', 1)

        # Fast mode: queue depth 1 so the driver always works from the newest pose instead of a
        # backlog (a depth-10 queue that falls behind the sim means up to ~100 ms stale poses),
        # and the control step runs on each odometry message. /clock is used only to log latency.
        depth = 1 if self.fast is not None else 10
        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value, self.scan_callback, depth)
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self.odom_callback, depth)
        self.clock_now = None
        self.odom_stamp = None
        self.ages = []
        self.last_age_report = None
        self.declare_parameter('latency_log_dir', '')
        self.latency_file = None
        if self.fast is not None:
            from rosgraph_msgs.msg import Clock
            self.create_subscription(Clock, '/clock', self.clock_callback, 1)
            log_dir = str(self.get_parameter('latency_log_dir').value)
            if log_dir:
                import os, time
                os.makedirs(log_dir, exist_ok=True)
                path = os.path.join(log_dir, time.strftime('pose_age_%Y%m%dT%H%M%S.csv'))
                self.latency_file = open(path, 'w', buffering=1)
                self.latency_file.write('sim_time_s,pose_age_ms\n')
                self.get_logger().info(f'logging pose age to {path}')

        self._marker_divisor = 0
        self.get_logger().warn(
            'team_driver is up, but this template has no driving logic: it will '
            'crawl straight ahead until it hits something. Implement plan().')

    # ------------------------------------------------------------------
    # Odometry: where the car is. Free, accurate, and worth building on.
    # ------------------------------------------------------------------
    def _floats(self, name):
        text = str(self.get_parameter(name).value).strip()
        return [float(x) for x in text.split(',')] if text else None

    @staticmethod
    def _resolve_line(path_file):
        """Explicit path_file wins; otherwise find the default line wherever the package lives."""
        import os
        name = 'raceline_sim_optimized_v3.csv'
        candidates = [path_file] if path_file else []
        try:
            from ament_index_python.packages import get_package_share_directory
            candidates.append(os.path.join(get_package_share_directory('team_driver'), 'config', name))
        except Exception:  # noqa: BLE001
            pass
        here = os.path.dirname(os.path.realpath(__file__))
        candidates += [os.path.join(here, '..', 'config', name),
                       os.path.join('/hackathon/race_ws/src/team_driver/config', name)]
        for c in candidates:
            if c and os.path.isfile(c):
                return c
        raise FileNotFoundError(f'raceline not found, tried: {candidates}')

    def odom_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        if self.fast is not None:
            self.odom_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
            steering, speed = self.fast(self.position[0], self.position[1], self.yaw, self.speed, self.odom_stamp)
            self.last_steering = steering
            self.publish(steering, speed)
            self.report_latency()

    def clock_callback(self, msg):
        self.clock_now = msg.clock.sec + msg.clock.nanosec*1e-9

    def report_latency(self):
        """Log how stale the pose is when we act on it (sim time), every ~2 s."""
        if self.clock_now is None or self.odom_stamp is None:
            return
        age = max(0.0, self.clock_now-self.odom_stamp)
        self.ages.append(age)
        if self.latency_file is not None:
            self.latency_file.write(f'{self.odom_stamp:.3f},{age*1000:.1f}\n')
        if self.last_age_report is None:
            self.last_age_report = self.odom_stamp
        if self.odom_stamp-self.last_age_report >= 2.0 and self.ages:
            a = np.array(self.ages)*1000
            self.get_logger().info(f'pose age ms: median {np.median(a):.0f}  p95 {np.percentile(a, 95):.0f}  '
                                   f'max {a.max():.0f}  n={len(a)}')
            self.ages = []
            self.last_age_report = self.odom_stamp

    # ------------------------------------------------------------------
    # The control loop, once per LiDAR scan (about 40 Hz).
    # ------------------------------------------------------------------
    def scan_callback(self, scan):
        if self.fast is not None:
            return  # fast mode drives from odom_callback
        ranges, angles = self.preprocess(scan)
        steering, speed = self.plan(ranges, angles)
        self.last_steering = steering
        self.publish(steering, speed)

        self._marker_divisor = (self._marker_divisor + 1) % 10
        if self._marker_divisor == 0:
            self.publish_marker(steering)

    def preprocess(self, scan):
        """Turn a raw scan into clean ranges plus the angle of each beam.

        Kept because every approach needs some version of it and the details
        are fiddly rather than interesting: the LiDAR reports NaN and inf, and
        arithmetic on those propagates silently through everything downstream.
        """
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=self.max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.max_range)

        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        return ranges, angles

    # ==================================================================
    # THIS IS THE PART YOU WRITE.
    # ==================================================================
    def plan_ttw(self, ranges, angles):
        """ver1: wall following below"""
        def beam(target_angle):
            idx=np.argmin(np.abs(angles-target_angle))
            return ranges[idx]
        b1 = beam(np.deg2rad(-90.0))
        b2 = beam(np.deg2rad(-45.0))  #obtain the distance from wall
        if(b1<0.05 or b2<0.05):
            return (0.0,0.0)
        theta = np.deg2rad(45.0) #angle bet b1&b2
        alpha = np.arctan2(b2*np.cos(theta)-b1,
                            b2*np.sin(theta)) #calc angle with wall,if - ,will closer
        perp_dis = b1*np.cos(alpha) #perpendicular dis with wall
        look_ahead = 0.8
        fut_dis = perp_dis + look_ahead*np.sin(alpha)
        want_dis = 0.8 #keep 0.8m to wall
        err_dis = want_dis-fut_dis # - then right, + then left
        kp = 0.7
        steering = kp*err_dis
        steering = np.clip(steering,-0.34,+0.34)
        speed = self.crawl_speed
        return (steering, speed)



    def plan_ftg(self, ranges, angles):
        """ver2: follow the gap  : 27s/lap can improve"""
        tmprange = ranges.copy()
        
        restriction = np.deg2rad(70.0)
        res_mask = np.abs(angles) <= restriction
        tmprange[~res_mask] = 0.0  #only consider front, here -70~70
        
        #find valid idx, which is in restrction area but not too close 
        valid_idx = np.flatnonzero(res_mask & (tmprange>=0.05)) 
        if(len(valid_idx)==0):
            return (0.0,0.0)
        
        nearest_idx = valid_idx[np.argmin(tmprange[valid_idx])]
        nearest_dis = tmprange[nearest_idx] #nearest obstacle
        
        # disparity extension
        tmprange2 = tmprange.copy()
        fov_idx = np.flatnonzero(res_mask)
        angle_step = abs(angles[1] - angles[0])
        car_half = 0.31/2.0
        protect_width = car_half + 0.16
        disparity_thresh = 0.5 # the threshold we consider it to be edge of obst
        for i in range(fov_idx[0],fov_idx[-1]):
            r0,r1 = tmprange2[i],tmprange2[i+1]
            if(r0<0.05 or r1<0.05):
                continue
            if(abs(r1-r0) <= disparity_thresh): #we find edge of obst
                continue
            nearer_dis = min(r0,r1)
            extend_angle = np.arctan2(protect_width,nearer_dis) 
            extend_ct = int(np.ceil(extend_angle/angle_step)) #obtain protect angle&its ct
            if(r0<r1): #left side obst, lidar from i to i+1+extend_ct
                stop = min(fov_idx[-1]+1,i+1+extend_ct)
                tmprange[i+1:stop] = np.minimum(tmprange[i+1:stop],nearer_dis)
            else: #right side obst, i+1-ct to i+1
                start = max(fov_idx[0],i+1-extend_ct)
                tmprange[start:i+1] = np.minimum(tmprange[start:i+1],nearer_dis)
        
        #to delete all possible range in bubble(not accessible) of closest obst
        bubble_radius = 0.35
        bubble_angle = np.arctan2(bubble_radius, max(nearest_dis,0.05))
        bubble_mask = np.abs(angles-angles[nearest_idx]) <= bubble_angle
        tmprange[bubble_mask] = 0.0 
        
        # get if accesible as a list to check for gaps
        min_clear = 0.8
        free = res_mask & (tmprange >= min_clear)
        
        padded = np.pad(free.astype(np.int8),(1,1))
        changes = np.diff(padded)  #calculate change to see if gap, 0->1 start gap
        gap_start = np.flatnonzero(changes == 1)
        gap_end = np.flatnonzero(changes == -1)-1 #get gaps
        if(len(gap_start) == 0):
            return (0.0,0.0)
        
        gap_length = gap_end - gap_start +1
        maxgap = np.argmax(gap_length)
        maxstart,maxend = gap_start[maxgap],gap_end[maxgap] #get data of max gap
        
        #get the range in the gap, find peak, and select available dir near peak for mid
        gap_range = tmprange[maxstart:maxend +1]
        peak = np.max(gap_range)
        peak_mask = gap_range >= 0.9 * peak
        peak_padded = np.pad(peak_mask.astype(np.int8),(1,1))
        peak_changes = np.diff(peak_padded)
        peak_start = np.flatnonzero(peak_changes == 1)
        peak_end = np.flatnonzero(peak_changes == -1)-1 #get high clearance area same as get gaps
        peak_length = peak_end - peak_start +1
        maxpeak = np.max(peak_length)
        #now we choose from longest segments of clearance
        candidates = np.flatnonzero(peak_length == maxpeak)
        candidate_center = (peak_start[candidates]+peak_end[candidates])//2
        gap_center = len(gap_range) / 2.0
        best_candidate = np.argmin(np.abs(candidate_center-gap_center))
        target_idx = maxstart + candidate_center[best_candidate]
        
        
        steering = angles[target_idx]
        steering = np.clip(steering,-0.34,0.34)
        
        speed = self.crawl_speed*(3.0 - 0.6 * (abs(steering)/0.34)) #stable 3.0
        speed = max(2.20,speed)
        
        return (steering,speed)

    def plan_pure_persuit(self, ranges, angles):
        """ver3: pure pure persuit  delta = arctan(2Ly_L/L_d^2) 15/lap,shaky but can improve"""
        if(self.position is None):
            return (0.0,0.0)
        look_ahead = np.clip(self.look_ahead + 0.12 * self.speed, 0.79, 1.55)
        wheel_base = 0.3302

        car_pos = np.array(self.position) #get pos of car
        pos_2_dis = np.linalg.norm(self.path-car_pos,axis = 1) #get the array of dis to points
        near_pt = np.argmin(pos_2_dis) #get nearest_point

        c = np.cos(self.yaw)
        s = np.sin(self.yaw)

        for i in range(1,len(self.path)):
            idx = (near_pt + i)%len(self.path)  #get idx of lookahead by iteration
            dx,dy = self.path[idx] - car_pos
            x = c*dx + s*dy
            y = -s*dx + c*dy #convert to car local
            if(x>0 and x*x+y*y>=look_ahead*look_ahead): #L_d>=lookahead
                steering = np.arctan2(2*wheel_base*y , (x*x+y*y))
                steering = np.clip(steering,-0.34,0.34)
                speed = self.crawl_speed*(self.max_speed - self.turn_slowdown * (abs(steering)/0.34))
                speed = max(self.min_speed,speed)
                return (steering,speed)
        return (0.0,0.0)
    
    def plan_pure_pursuit_v2(self, ranges, angles):

        if self.position is None:
            return (0.0, 0.0)

        car_pos = np.array(self.position)
        wheel_base = 0.3302

        # ==================================================
        # 1. Find nearest centerline point
        # ==================================================

        distances = np.linalg.norm(
            self.path - car_pos,
            axis=1
        )

        near_idx = np.argmin(distances)

        # ==================================================
        # 2. Adaptive lookahead
        #
        # Smaller than before:
        # better corner tracking, less corner cutting
        # ==================================================

        look_ahead = np.clip(
            0.70 + 0.12 * self.speed,
            0.79,
            1.55
        )

        # ==================================================
        # 3. Vehicle coordinate transform
        # ==================================================

        c = np.cos(self.yaw)
        s = np.sin(self.yaw)

        target_x = None
        target_y = None

        # ==================================================
        # 4. Find lookahead point
        # ==================================================

        for i in range(1, len(self.path)):

            idx = (near_idx + i) % len(self.path)

            dx, dy = self.path[idx] - car_pos

            x = c * dx + s * dy
            y = -s * dx + c * dy

            distance = np.hypot(x, y)

            if x > 0.0 and distance >= look_ahead:
                target_x = x
                target_y = y
                break

        if target_x is None:
            return (0.0, 0.0)

        # ==================================================
        # 5. Pure Pursuit
        # ==================================================

        ld2 = (
            target_x * target_x
            +
            target_y * target_y
        )

        raw_steering = np.arctan2(
            2.0 * wheel_base * target_y,
            ld2
        )

        raw_steering = np.clip(
            raw_steering,
            -0.34,
            0.34
        )

        # ==================================================
        # 6. Steering RATE limiter
        #
        # Unlike low-pass filter:
        # still reacts quickly in corners,
        # but prevents sudden steering jumps.
        # ==================================================

        max_change = 0.066

        steering_change = (
            raw_steering - self.last_steering
        )

        steering_change = np.clip(
            steering_change,
            -max_change,
            max_change
        )

        steering = (
            self.last_steering
            +
            steering_change
        )

        steering = np.clip(
            steering,
            -0.34,
            0.34
        )

        self.last_steering = steering

        # ==================================================
        # 7. Speed according to corner severity
        # ==================================================

        turn_ratio = min(
            abs(raw_steering) / 0.34,
            1.0
        )

        # straight ≈ 4.5
        # hard corner ≈ 2.2
        speed = (
            self.max_speed 
            - 
            self.min_speed * turn_ratio
        )
        speed = max(
            self.min_speed,
            speed
        )
        # ==================================================
        # 8. LiDAR emergency protection
        # ==================================================

        front_mask = np.abs(angles) < np.deg2rad(20.0)

        front_ranges = ranges[front_mask]

        valid_front = front_ranges[
            front_ranges > 0.05
        ]

        if len(valid_front) > 0:

            front_distance = np.min(valid_front)

            if front_distance < 0.55:
                speed = 0.0

            elif front_distance < 0.9:
                speed = min(speed, 1.2)

            elif front_distance < 1.3:
                speed = min(speed, 2.2)

        return (steering, speed)
    
    def plan(self, ranges, angles):
        """ver1: plan_ttw()
            ver2: plan_ftg() good
            ver3: plan_pure_persuit()
            ver4: plan_pure_pursuit_v2() for better stability"""
        if self.fast is not None:
            if self.position is None:
                return (0.0, 0.0)
            return self.fast(self.position[0], self.position[1], self.yaw, self.speed, 0.0)
        return self.plan_pure_persuit(ranges,angles)


    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def publish(self, steering, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

    def publish_marker(self, target_angle):
        """Draw where the car thinks it is going. Add /driver/markers in RViz."""
        marker = Marker()
        marker.header.frame_id = 'ego_racecar/base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'team_driver'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = 1.5, 0.15, 0.15
        marker.color.g, marker.color.b, marker.color.a = 0.8, 1.0, 0.9
        marker.pose.orientation.z = math.sin(target_angle / 2.0)
        marker.pose.orientation.w = math.cos(target_angle / 2.0)
        array = MarkerArray()
        array.markers.append(marker)
        self.marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = Driver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
