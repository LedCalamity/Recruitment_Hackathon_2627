#!/usr/bin/env python3
"""The official Track 1 referee.

Watches a run, times it, applies the collision penalties and writes a result
file. It only ever observes the car - it never sends a drive command - so the
same node is used for practice and for judging.

    ros2 run roboracer_referee referee --ros-args -p team:=my_team
    ros2 launch roboracer_referee evaluate.launch.py team:=my_team driver_pkg:=team_driver

Timing runs on the simulated clock the gym bridge publishes on /clock, not on
the wall clock. A run therefore scores the same on a fast desktop and a tired
laptop; see docs/05-evaluation.md.
"""

from __future__ import annotations

import json
import math
import os
import signal
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from typing import Optional

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker

from .session import RaceSession, Rules
from .tracks import TrackError, load_track

REFEREE_VERSION = "1.0.0"
RESULT_SCHEMA_VERSION = 2

# Node exit codes, so a shell harness can branch on the outcome.
EXIT_OK = 0
EXIT_NOT_SCORED = 2       # ran, but DNF or disqualified
EXIT_SETUP_FAILED = 3     # never got as far as racing


class Phase:
    WAIT_TOPICS = "waiting for the simulator"
    RESETTING = "placing the car on the grid"
    WAIT_DRIVER = "waiting for the driver node"
    RACING = "racing"
    DONE = "finished"


class Referee(Node):

    def __init__(self):
        super().__init__("referee")

        # -- parameters ----------------------------------------------------
        self.declare_parameter("team", "unnamed_team")
        self.declare_parameter("run_id", "")
        self.declare_parameter("track", "")
        self.declare_parameter("track_config", "")
        self.declare_parameter("output_dir", "/hackathon/results")

        self.declare_parameter("warmup_laps", 1)
        self.declare_parameter("timed_laps", 10)
        self.declare_parameter("collision_penalty_s", 10.0)
        self.declare_parameter("max_collisions", 10)
        self.declare_parameter("collision_interval_s", 1.0)
        self.declare_parameter("min_lap_time_s", 2.0)
        self.declare_parameter("session_timeout_s", 900.0)
        self.declare_parameter("stuck_speed_mps", 0.05)
        self.declare_parameter("stuck_timeout_s", 15.0)

        self.declare_parameter("startup_timeout_s", 60.0)
        self.declare_parameter("driver_timeout_s", 60.0)
        self.declare_parameter("wall_timeout_s", 1800.0)
        self.declare_parameter("reset_car", True)

        self.declare_parameter("odom_topic", "/ego_racecar/odom")
        self.declare_parameter("collision_topic", "/ego_racecar/collision")
        self.declare_parameter("sim_time_topic", "/clock")
        self.declare_parameter("drive_topic", "/drive")

        self.team = str(self.get_parameter("team").value).strip() or "unnamed_team"
        self.run_id = str(self.get_parameter("run_id").value).strip() \
            or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.output_dir = str(self.get_parameter("output_dir").value)
        self.startup_timeout_s = float(self.get_parameter("startup_timeout_s").value)
        self.driver_timeout_s = float(self.get_parameter("driver_timeout_s").value)
        self.wall_timeout_s = float(self.get_parameter("wall_timeout_s").value)
        self.reset_car = bool(self.get_parameter("reset_car").value)

        # -- track ---------------------------------------------------------
        track_name = str(self.get_parameter("track").value).strip() or None
        track_config = str(self.get_parameter("track_config").value).strip() or None
        try:
            self.track = load_track(track_name, track_config)
        except TrackError as exc:
            self.get_logger().fatal(str(exc))
            raise SystemExit(EXIT_SETUP_FAILED)

        rules = Rules(
            warmup_laps=int(self.get_parameter("warmup_laps").value),
            timed_laps=int(self.get_parameter("timed_laps").value),
            collision_penalty_s=float(self.get_parameter("collision_penalty_s").value),
            max_collisions=int(self.get_parameter("max_collisions").value),
            collision_interval_s=float(self.get_parameter("collision_interval_s").value),
            min_lap_time_s=float(self.get_parameter("min_lap_time_s").value),
            session_timeout_s=float(self.get_parameter("session_timeout_s").value),
            stuck_speed_mps=float(self.get_parameter("stuck_speed_mps").value),
            stuck_timeout_s=float(self.get_parameter("stuck_timeout_s").value),
        )
        try:
            self.session = RaceSession(
                rules, self.track.finish_line[0], self.track.finish_line[1],
                self.track.crossing_direction,
            )
        except ValueError as exc:
            self.get_logger().fatal(str(exc))
            raise SystemExit(EXIT_SETUP_FAILED)

        # -- state ---------------------------------------------------------
        self.phase = Phase.WAIT_TOPICS
        self.sim_time: Optional[float] = None
        self.first_sim_time: Optional[float] = None
        self.last_sim_time: Optional[float] = None
        self.last_odom_time: Optional[float] = None
        self.last_position: Optional[tuple] = None
        self.driver_seen = False
        self.reset_requested_at: Optional[float] = None
        self.phase_started_wall = time.monotonic()
        self.started_wall = time.monotonic()
        self.result_written = False
        self._callback_errors = 0

        # -- ROS interfaces ------------------------------------------------
        odom_topic = str(self.get_parameter("odom_topic").value)
        collision_topic = str(self.get_parameter("collision_topic").value)
        sim_time_topic = str(self.get_parameter("sim_time_topic").value)
        drive_topic = str(self.get_parameter("drive_topic").value)

        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.create_subscription(Bool, collision_topic, self._on_collision, 50)
        self.create_subscription(Clock, sim_time_topic, self._on_sim_time, 10)
        self.create_subscription(AckermannDriveStamped, drive_topic, self._on_drive, 10)

        self.reset_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        self.line_pub = self.create_publisher(Marker, "/referee/finish_line", 1)
        self.status_pub = self.create_publisher(Marker, "/referee/status", 1)

        self.create_timer(0.2, self._tick)
        self.create_timer(1.0, self._publish_markers)

        self._log_banner()

    # -- logging -----------------------------------------------------------

    def _log_banner(self) -> None:
        rules = self.session.rules
        (ax, ay), (bx, by) = self.track.finish_line
        self.get_logger().info(
            "\n"
            "=========================================================\n"
            f" RoboRacer Track 1 referee v{REFEREE_VERSION}\n"
            "=========================================================\n"
            f" team          : {self.team}\n"
            f" run id        : {self.run_id}\n"
            f" track         : {self.track.name}\n"
            f" map           : {self.track.map_path}\n"
            f" grid slot     : ({self.track.start_pose[0]:.2f}, {self.track.start_pose[1]:.2f}, "
            f"{math.degrees(self.track.start_pose[2]):.1f} deg)\n"
            f" finish line   : ({ax:.2f}, {ay:.2f}) -> ({bx:.2f}, {by:.2f})\n"
            f" format        : {rules.warmup_laps} warm-up lap(s) + {rules.timed_laps} timed lap(s)\n"
            f" penalty       : +{rules.collision_penalty_s:.0f}s per collision, "
            f"DQ above {rules.max_collisions}\n"
            f" results       : {self.output_dir}\n"
            "========================================================="
        )

    # -- subscriptions -----------------------------------------------------

    def _guard(self, where: str, exc: Exception) -> None:
        """Never let a bad message kill the referee mid-run."""
        self._callback_errors += 1
        if self._callback_errors <= 5:
            self.get_logger().error(f"{where}: {exc}\n{traceback.format_exc()}")
        elif self._callback_errors == 6:
            self.get_logger().error(f"{where}: further callback errors suppressed")

    def _on_sim_time(self, msg: Clock) -> None:
        try:
            value = msg.clock.sec + msg.clock.nanosec * 1e-9
            if not math.isfinite(value):
                return
            self.sim_time = value
            if self.first_sim_time is None:
                self.first_sim_time = value
            self.last_sim_time = value
        except Exception as exc:                            # noqa: BLE001
            self._guard("sim_time callback", exc)

    def _on_drive(self, _msg: AckermannDriveStamped) -> None:
        self.driver_seen = True

    def _on_collision(self, msg: Bool) -> None:
        try:
            if not msg.data or self.phase != Phase.RACING or self.sim_time is None:
                return
            if self.session.register_collision(self.sim_time):
                self.get_logger().warn(
                    f"COLLISION {self.session.collision_events}/"
                    f"{self.session.rules.max_collisions} "
                    f"(+{self.session.rules.collision_penalty_s:.0f}s on this lap)"
                )
        except Exception as exc:                            # noqa: BLE001
            self._guard("collision callback", exc)

    def _on_odom(self, msg: Odometry) -> None:
        try:
            self.last_odom_time = time.monotonic()
            self.last_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
            if self.phase != Phase.RACING or self.sim_time is None:
                return
            speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
            for event in self.session.update(
                    self.sim_time, self.last_position[0], self.last_position[1], speed):
                self.get_logger().info(event)
            if self.session.finished:
                self._conclude()
        except Exception as exc:                            # noqa: BLE001
            self._guard("odom callback", exc)

    # -- phases ------------------------------------------------------------

    def _enter(self, phase: str) -> None:
        self.phase = phase
        self.phase_started_wall = time.monotonic()
        self.get_logger().info(f"[{phase}]")

    def _phase_elapsed(self) -> float:
        return time.monotonic() - self.phase_started_wall

    def _tick(self) -> None:
        try:
            if self.phase == Phase.DONE:
                return

            if time.monotonic() - self.started_wall > self.wall_timeout_s:
                self.get_logger().error(
                    f"Wall-clock limit of {self.wall_timeout_s:.0f}s reached. Ending the run."
                )
                self.session.abort("wall-clock watchdog expired")
                self._conclude()
                return

            if self.phase == Phase.WAIT_TOPICS:
                self._tick_wait_topics()
            elif self.phase == Phase.RESETTING:
                self._tick_resetting()
            elif self.phase == Phase.WAIT_DRIVER:
                self._tick_wait_driver()
            elif self.phase == Phase.RACING and self.session.finished:
                self._conclude()
        except SystemExit:
            raise
        except Exception as exc:                            # noqa: BLE001
            self._guard("referee tick", exc)

    def _tick_wait_topics(self) -> None:
        have_sim_time = self.sim_time is not None
        have_odom = self.last_odom_time is not None
        if have_sim_time and have_odom:
            if self.reset_car:
                self._enter(Phase.RESETTING)
            else:
                self._enter(Phase.WAIT_DRIVER)
            return

        if self._phase_elapsed() < self.startup_timeout_s:
            if int(self._phase_elapsed()) % 5 == 0:
                missing = []
                if not have_sim_time:
                    missing.append(str(self.get_parameter("sim_time_topic").value))
                if not have_odom:
                    missing.append(str(self.get_parameter("odom_topic").value))
                self.get_logger().info(f"Waiting for: {', '.join(missing)}", throttle_duration_sec=5)
            return

        if not have_odom:
            self._fail_setup(
                f"No odometry on {self.get_parameter('odom_topic').value} after "
                f"{self.startup_timeout_s:.0f}s. Is the simulator running? Start it with:\n"
                "    ros2 launch roboracer_referee simulator.launch.py"
            )
        else:
            self._fail_setup(
                f"No simulated clock on {self.get_parameter('sim_time_topic').value} after "
                f"{self.startup_timeout_s:.0f}s.\n"
                "The simulator is running but is not publishing simulated time, so a fair run\n"
                "cannot be timed. The bridge needs use_sim_time_bridge set, which\n"
                "simulator.launch.py does for you:\n"
                "    ros2 launch roboracer_referee simulator.launch.py"
            )

    def _tick_resetting(self) -> None:
        # The car is teleported onto the grid so that every run starts from an
        # identical state. Repeat the request: /initialpose is best-effort and
        # the bridge may not have wired up its subscription on the first try.
        if self._phase_elapsed() < 3.0:
            self.reset_pub.publish(self._start_pose_msg())
            return

        near_start = (getattr(self, "last_position", None) is not None
                      and math.dist(self.last_position, self.track.start_pose[:2]) < 1.0)
        if near_start:
            self.get_logger().info("Car is on the grid.")
            self._enter(Phase.WAIT_DRIVER)
        elif self._phase_elapsed() > 15.0:
            self.get_logger().warn(
                "The car did not move to the grid slot. Continuing from wherever it is; "
                "lap counting still works, but the out lap will be a different length."
            )
            self._enter(Phase.WAIT_DRIVER)
        else:
            self.reset_pub.publish(self._start_pose_msg())

    def _tick_wait_driver(self) -> None:
        if self.driver_seen:
            self.get_logger().info("Driver is publishing. Green flag.")
            self._enter(Phase.RACING)
            return
        if self._phase_elapsed() > self.driver_timeout_s:
            self._fail_setup(
                f"Nothing published on {self.get_parameter('drive_topic').value} after "
                f"{self.driver_timeout_s:.0f}s.\n"
                "Start your driver node, e.g.:\n"
                "    ros2 run team_driver driver"
            )
        else:
            self.get_logger().info(
                f"Waiting for drive commands on {self.get_parameter('drive_topic').value}",
                throttle_duration_sec=5,
            )

    def _start_pose_msg(self) -> PoseWithCovarianceStamped:
        x, y, theta = self.track.start_pose
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(theta / 2.0)
        msg.pose.covariance = [0.0] * 36
        return msg

    # -- visualisation -----------------------------------------------------

    def _publish_markers(self) -> None:
        try:
            (ax, ay), (bx, by) = self.track.finish_line
            line = Marker()
            line.header.frame_id = "map"
            line.header.stamp = self.get_clock().now().to_msg()
            line.ns = "referee"
            line.id = 0
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.15
            line.color.r, line.color.g, line.color.b, line.color.a = 1.0, 0.85, 0.0, 1.0
            line.pose.orientation.w = 1.0
            line.points = [Point(x=ax, y=ay, z=0.0), Point(x=bx, y=by, z=0.0)]
            self.line_pub.publish(line)

            text = Marker()
            text.header.frame_id = "map"
            text.header.stamp = line.header.stamp
            text.ns = "referee"
            text.id = 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.scale.z = 0.6
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.pose.position.x = (ax + bx) / 2.0
            text.pose.position.y = (ay + by) / 2.0
            text.pose.position.z = 1.5
            text.pose.orientation.w = 1.0
            text.text = self._status_line()
            self.status_pub.publish(text)
        except Exception as exc:                            # noqa: BLE001
            self._guard("marker publisher", exc)

    def _status_line(self) -> str:
        rules = self.session.rules
        if self.phase != Phase.RACING and self.phase != Phase.DONE:
            return f"{self.team} | {self.phase}"
        best = self.session.best_lap()
        parts = [
            f"{self.team}",
            f"lap {self.session.timed_laps_done}/{rules.timed_laps}",
            f"collisions {self.session.collision_events}/{rules.max_collisions}",
        ]
        if best:
            parts.append(f"best {best.net_time:.2f}s")
        if self.session.finished:
            parts.append(self.session.status.value)
        return " | ".join(parts)

    # -- finishing ---------------------------------------------------------

    def _fail_setup(self, message: str) -> None:
        self.get_logger().fatal(message)
        self.session.abort("setup failed: " + message.splitlines()[0])
        self._write_result()
        self._shutdown(EXIT_SETUP_FAILED)

    def _conclude(self) -> None:
        if self.phase == Phase.DONE:
            return
        self._enter(Phase.DONE)
        self._publish_markers()
        result = self._write_result()
        self._print_summary(result)
        self._shutdown(EXIT_OK if self.session.scored else EXIT_NOT_SCORED)

    def real_time_factor(self) -> Optional[float]:
        if self.first_sim_time is None or self.last_sim_time is None:
            return None
        wall = time.monotonic() - self.started_wall
        if wall <= 0:
            return None
        return (self.last_sim_time - self.first_sim_time) / wall

    def _build_result(self) -> dict:
        rtf = self.real_time_factor()
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "referee_version": REFEREE_VERSION,
            "team": self.team,
            "run_id": self.run_id,
            "track": self.track.name,
            "map_path": self.track.map_path,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "environment": {
                "real_time_factor": round(rtf, 3) if rtf else None,
                "wall_duration_s": round(time.monotonic() - self.started_wall, 2),
                "sim_duration_s": (round(self.last_sim_time - self.first_sim_time, 2)
                                   if self.first_sim_time is not None else None),
                "callback_errors": self._callback_errors,
            },
        }
        result.update(self.session.result())
        return result

    def _write_result(self) -> dict:
        result = self._build_result()
        if self.result_written:
            return result
        self.result_written = True

        directory = self.output_dir
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            fallback = os.path.join(tempfile.gettempdir(), "roboracer_results")
            self.get_logger().error(f"Cannot create {directory} ({exc}); using {fallback}")
            directory = fallback
            os.makedirs(directory, exist_ok=True)

        safe_team = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.team)
        path = os.path.join(directory, f"{safe_team}__{self.run_id}.json")
        try:
            # Write to a sibling temp file and rename, so a result file is never
            # left half written if the process is killed at the wrong moment.
            fd, tmp = tempfile.mkstemp(dir=directory, prefix=".referee-", suffix=".json")
            with os.fdopen(fd, "w") as handle:
                json.dump(result, handle, indent=2, sort_keys=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            os.chmod(path, 0o644)
            self.get_logger().info(f"Result written to {path}")
            result["_path"] = path
        except OSError as exc:
            self.get_logger().error(f"Could not write the result file: {exc}")
            self.get_logger().error("Result JSON follows so the run is not lost:\n"
                                    + json.dumps(result, indent=2))
        return result

    def _print_summary(self, result: dict) -> None:
        rules = self.session.rules
        lines = [
            "",
            "=========================================================",
            f" RESULT - {self.team} on {self.track.name}",
            "=========================================================",
            f" status        : {result['status']}",
        ]
        if result.get("reason"):
            lines.append(f" reason        : {result['reason']}")
        lines.append(f" laps          : {result['laps_completed']}/{result['laps_required']}")
        lines.append(f" collisions    : {result['collisions']}/{rules.max_collisions}"
                     f"  (penalty {result['total_penalty_s']:.0f}s)")

        if result["laps"]:
            lines.append("")
            lines.append("  lap    raw       pen    net")
            for lap in result["laps"]:
                lines.append(f"  {lap['number']:>3}  {lap['raw_time']:>7.3f}  "
                             f"{lap['penalty_s']:>5.0f}  {lap['net_time']:>7.3f}")

        lines.append("")
        if result["scored"]:
            lines.append(f" BEST LAP      : {result['best_lap_time']:.3f} s  "
                         f"(lap {result['best_lap_number']})")
            lines.append(f" {rules.timed_laps}-LAP TOTAL  : {result['total_time']:.3f} s")
        else:
            lines.append(" NOT SCORED - this run does not produce leaderboard times.")

        rtf = result["environment"]["real_time_factor"]
        if rtf is not None:
            lines.append(f" real-time factor: {rtf:.2f}x"
                         + ("  (machine could not keep up, but times are unaffected)"
                            if rtf < 0.8 else ""))
        lines.append("=========================================================")
        self.get_logger().info("\n".join(lines))

    def _shutdown(self, code: int) -> None:
        self.exit_code = code
        raise SystemExit(code)


def main(args=None) -> int:
    rclpy.init(args=args)
    node = None
    code = EXIT_SETUP_FAILED
    try:
        node = Referee()

        # Ctrl-C and `ros2 launch` teardown must still leave a result file behind.
        def _on_signal(signum, _frame):
            if node and not node.result_written:
                node.get_logger().warn(f"Signal {signum} received; saving a partial result.")
                node.session.abort(f"interrupted by signal {signum}")
                node._write_result()
            raise KeyboardInterrupt

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, _on_signal)

        rclpy.spin(node)
        code = EXIT_OK
    except SystemExit as exc:
        code = int(exc.code) if exc.code is not None else EXIT_OK
    except KeyboardInterrupt:
        code = EXIT_NOT_SCORED
    except Exception:                                       # noqa: BLE001
        traceback.print_exc()
        code = EXIT_SETUP_FAILED
    finally:
        if node is not None:
            if not node.result_written and node.phase != Phase.WAIT_TOPICS:
                node.session.abort("referee stopped before the run finished")
                node._write_result()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
