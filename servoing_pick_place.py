#!/usr/bin/env python3

# --- MUST be first: patch collections for old protobuf (Python 3.10+) ---
import collections
import collections.abc
import os
import sys
import time

_COLLECTIONS_COMPAT_ATTRS = (
    "Mapping", "MutableMapping", "Sequence", "MutableSequence",
    "Set", "MutableSet", "Iterable",
)
for _name in _COLLECTIONS_COMPAT_ATTRS:
    if not hasattr(collections, _name) and hasattr(collections.abc, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

def _add_active_venv_site_packages():
    venv = os.environ.get("VIRTUAL_ENV")
    if not venv:
        return
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = os.path.join(venv, "lib", py_version, "site-packages")
    if os.path.isdir(site_packages) and site_packages not in sys.path:
        sys.path.insert(0, site_packages)

_add_active_venv_site_packages()
# -----------------------------------------------------------------------

import math
import queue
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.TCPTransport import TCPTransport
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
from kortex_api.autogen.messages import Base_pb2, ControlConfig_pb2, Session_pb2


# ── States ────────────────────────────────────────────────────────────────────
STATE_TRACKING = "TRACKING"
STATE_GRABBING = "GRABBING"
STATE_GRABBED  = "GRABBED"
# ─────────────────────────────────────────────────────────────────────────────

# ── Box drop location ─────────────────────────────────────────────────────────
# Position of the box where the ball is dropped.
# Tune these values in the lab by jogging the arm to the box position.

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def euler_to_R(tx, ty, tz):
    rx, ry, rz = math.radians(tx), math.radians(ty), math.radians(tz)
    Rx = np.array([[1, 0, 0],
                   [0,  math.cos(rx), -math.sin(rx)],
                   [0,  math.sin(rx),  math.cos(rx)]])
    Ry = np.array([[ math.cos(ry), 0, math.sin(ry)],
                   [0,             1, 0            ],
                   [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                   [math.sin(rz),  math.cos(rz), 0],
                   [0,             0,             1]])
    return Rz @ Ry @ Rx

def pixel_to_cam(px, py, fx, fy, cx, cy, depth):
    return np.array([
        -((px - cx) / fx) * depth,
        -((py - cy) / fy) * depth,
        depth,
    ], dtype=np.float64)

def cam_to_world(p, R, t):
    return R @ p + t


@dataclass
class KortexConnectionConfig:
    ip: str
    port: int
    username: str
    password: str
    session_inactivity_timeout_ms: int = 60000
    connection_inactivity_timeout_ms: int = 2000


@dataclass
class CartesianPose:
    x: float
    y: float
    z: float
    theta_x: float
    theta_y: float
    theta_z: float

BOX_POSE = CartesianPose(
    x=0.44,       # <-- REPLACE with measured value
    y= -0.46,      # <-- REPLACE with measured value
    z=0.195,       # hover height above box (arm approaches from above)
    theta_x=-180.0,  # keep same orientation as tracking
    theta_y=0.0,
    theta_z=90.0,
)
# How far to descend into the box before releasing (metres)
BOX_DESCENT_M = 0.10   # <-- tune so gripper is just inside box rim
# ─────────────────────────────────────────────────────────────────────────────


class KortexSession:
    def __init__(self, config: KortexConnectionConfig):
        self._config = config
        self._transport = self._router = self._session_manager = None

    def __enter__(self):
        self._transport = TCPTransport()
        self._transport.connect(self._config.ip, self._config.port)
        self._router = RouterClient(
            self._transport, lambda e: print(f"Kortex error: {e}")
        )
        self._session_manager = SessionManager(self._router)
        info = Session_pb2.CreateSessionInfo()
        info.username = self._config.username
        info.password = self._config.password
        info.session_inactivity_timeout = self._config.session_inactivity_timeout_ms
        info.connection_inactivity_timeout = self._config.connection_inactivity_timeout_ms
        self._session_manager.CreateSession(info)
        return self._router

    def __exit__(self, *_):
        if self._session_manager:
            opts = RouterClientSendOptions()
            opts.timeout_ms = 1000
            self._session_manager.CloseSession(opts)
        if self._transport:
            self._transport.disconnect()


class VideoCaptureThread:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise ValueError("Camera not opened")
        self.q = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put((ret, frame))

    def read(self):
        return self.q.get()

    def release(self):
        self.running = False
        self.cap.release()
        self.thread.join()


class BallRealtimeTracker(Node):
    def __init__(self):
        super().__init__("ball_realtime_tracker")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("robot_ip",              "192.168.1.10")
        self.declare_parameter("robot_port",             10000)
        self.declare_parameter("username",              "admin")
        self.declare_parameter("password",              "admin")
        self.declare_parameter("action_timeout_s",       20.0)

        # Start pose
        self.declare_parameter("track_x",               0.35)
        self.declare_parameter("track_y",              -0.2)
        self.declare_parameter("track_z",               0.2)
        self.declare_parameter("track_theta_x",        -180.0)
        self.declare_parameter("track_theta_y",          0.0)
        self.declare_parameter("track_theta_z",          90.0)

        # Workspace safety limits
        self.declare_parameter("workspace_min_x",        0.20)
        self.declare_parameter("workspace_max_x",        0.55)
        self.declare_parameter("workspace_min_y",       -0.70)
        self.declare_parameter("workspace_max_y",       0.0)
        self.declare_parameter("workspace_min_z",        0.10)
        self.declare_parameter("workspace_max_z",        0.45)

        # Camera intrinsics
        self.declare_parameter("fx",                     1024.0)
        self.declare_parameter("fy",                     1024.0)
        self.declare_parameter("cx",                     640.0)
        self.declare_parameter("cy",                     360.0)
        self.declare_parameter("camera_height",          0.41)

        # Ball detection
        self.declare_parameter("ball_hsv_lower",         [100, 80, 20])
        self.declare_parameter("ball_hsv_upper",         [130, 255, 255])
        self.declare_parameter("min_ball_area",          2000)
        self.declare_parameter("morph_kernel_size",      5)
        self.declare_parameter("min_circularity",        0.6)

        # PD tracking controller
        self.declare_parameter("control_rate_hz",        15.0)
        self.declare_parameter("stale_timeout_s",        0.3)
        self.declare_parameter("kp_xy",                  1.2)
        self.declare_parameter("kd_xy",                  0.15)
        self.declare_parameter("xy_deadband_m",          0.005)
        self.declare_parameter("max_speed_xy_mps",       0.05)
        self.declare_parameter("target_diameter_px",     300)
        self.declare_parameter("max_speed_z_mps",        0.04)
        self.declare_parameter("kp_z",                   0.002)
        self.declare_parameter("kd_z",                   0.0005)
        self.declare_parameter("z_deadband_px",          3.0)

        # Grab
        self.declare_parameter("grab_descent_m",         0.15)
        self.declare_parameter("gripper_x_offset",       0.034)
        self.declare_parameter("gripper_close_value",    0.7)
        self.declare_parameter("gripper_open_value",     0.0)
        # ─────────────────────────────────────────────────────────────────

        def gp(name):
            return self.get_parameter(name).value

        config = KortexConnectionConfig(
            ip=gp("robot_ip"), port=gp("robot_port"),
            username=gp("username"), password=gp("password"),
        )

        self._action_timeout_s  = float(gp("action_timeout_s"))
        self._fx                = float(gp("fx"))
        self._fy                = float(gp("fy"))
        self._cx                = float(gp("cx"))
        self._cy                = float(gp("cy"))
        self._camera_height     = float(gp("camera_height"))
        self._control_rate_hz   = float(gp("control_rate_hz"))
        self._stale_timeout_s   = float(gp("stale_timeout_s"))
        self._kp_xy             = float(gp("kp_xy"))
        self._kd_xy             = float(gp("kd_xy"))
        self._xy_deadband_m     = float(gp("xy_deadband_m"))
        self._max_speed_xy      = float(gp("max_speed_xy_mps"))
        self._target_diam_px    = float(gp("target_diameter_px"))
        self._kp_z              = float(gp("kp_z"))
        self._kd_z              = float(gp("kd_z"))
        self._z_deadband_px     = float(gp("z_deadband_px"))
        self._max_speed_z       = float(gp("max_speed_z_mps"))
        self._ws_min_x          = float(gp("workspace_min_x"))
        self._ws_max_x          = float(gp("workspace_max_x"))
        self._ws_min_y          = float(gp("workspace_min_y"))
        self._ws_max_y          = float(gp("workspace_max_y"))
        self._ws_min_z          = float(gp("workspace_min_z"))
        self._ws_max_z          = float(gp("workspace_max_z"))
        self._grab_descent_m    = float(gp("grab_descent_m"))
        self._gripper_x_offset  = float(gp("gripper_x_offset"))
        self._gripper_close     = float(gp("gripper_close_value"))
        self._gripper_open      = float(gp("gripper_open_value"))

        self._fixed_pose = CartesianPose(
            x=float(gp("track_x")), y=float(gp("track_y")), z=float(gp("track_z")),
            theta_x=float(gp("track_theta_x")),
            theta_y=float(gp("track_theta_y")),
            theta_z=float(gp("track_theta_z")),
        )

        self._hsv_lower = np.array(gp("ball_hsv_lower"), dtype=np.uint8)
        self._hsv_upper = np.array(gp("ball_hsv_upper"), dtype=np.uint8)
        self._min_area  = float(gp("min_ball_area"))
        self._min_circ  = float(gp("min_circularity"))
        ks = int(gp("morph_kernel_size"))
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))

        # ── Connect to robot ──────────────────────────────────────────────
        self._session = KortexSession(config)
        self.get_logger().info(f"Connecting to {config.ip}:{config.port}...")
        router = self._session.__enter__()
        self._base = BaseClient(router)
        self._control_config = ControlConfigClient(router)
        self._hard_limits = None

        self._set_soft_limits()
        self._log_limits()
        self._clear_faults()

        # ── Runtime state ─────────────────────────────────────────────────
        self._state         = STATE_TRACKING
        self._manual_grab   = False
        self._twist_is_zero = True

        self._pose_lock      = threading.Lock()
        self._detection_lock = threading.Lock()

        self._measured_pose: Optional[CartesianPose] = None
        self._ball_world:    Optional[np.ndarray]    = None
        self._ball_diam_px:  Optional[float]         = None
        self._last_det_time     = 0.0
        self._last_no_ball_warn = 0.0

        self._prev_err_xy = np.zeros(2)
        self._prev_err_z  = 0.0
        self._prev_ctrl_t = time.monotonic()
        self._first_step  = True

        # ── Startup sequence ──────────────────────────────────────────────
        # Gen3 uses SINGLE_LEVEL_SERVOING for BOTH reach_pose AND twist.
        # Set it once here at startup and never change it.
        self._set_servoing_mode(Base_pb2.SINGLE_LEVEL_SERVOING)

        self.get_logger().info("Opening gripper...")
        self._set_gripper(self._gripper_open)

        self.get_logger().info("Moving to start pose...")
        self._reach_pose(self._fixed_pose)


        self.get_logger().info("Opening camera...")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        pipeline = (
            "rtspsrc location=rtsp://192.168.1.10/color latency=30 "
            "! rtph264depay ! avdec_h264 ! videoconvert ! appsink"
        )
        self._cap = VideoCaptureThread(pipeline)
        time.sleep(1.0)
        ret, f = self._cap.read()
        if not ret or f is None:
            raise RuntimeError("Camera opened but no frame received")
        h, w = f.shape[:2]
        self.get_logger().info(f"Camera ready: {w}x{h}")

        self._frame_timer = self.create_timer(1.0 / 100.0, self._process_frame)
        self._ctrl_timer  = self.create_timer(
            1.0 / max(self._control_rate_hz, 1.0), self._control_step
        )

        self.get_logger().info(
            "Ready! Tracking ball in 3D.\n"
            "  'g' = grab (down + close gripper + up)\n"
            "  'o' = open gripper\n"
            "  'r' = open gripper + return to start + resume tracking\n"
            "  'q' = quit"
        )

    # ── Servoing mode ─────────────────────────────────────────────────────────

    def _set_servoing_mode(self, mode):
        """
        Switch robot control mode.
        SINGLE_LEVEL_SERVOING = both reach_pose AND twist on Gen3
        LOW_LEVEL_SERVOING    = joint-level torque (NOT used here)

        This is the key fix — the robot must be explicitly told to switch
        modes before sending a different type of command. Mixing twist and
        reach_pose without switching modes causes the fault.
        """
        servoing_mode = Base_pb2.ServoingModeInformation()
        servoing_mode.servoing_mode = mode
        try:
            self._base.SetServoingMode(servoing_mode)
            time.sleep(0.05)   # small settle time after mode switch
        except Exception as exc:
            self.get_logger().warn(f"SetServoingMode failed: {exc}")

    # ── Fault handling ────────────────────────────────────────────────────────

    def _clear_faults(self):
        """Clear robot faults. Call on startup, before reach_pose, and after errors."""
        try:
            self._base.ClearFaults()
            self.get_logger().info("Faults cleared")
            time.sleep(0.3)
        except Exception as exc:
            self.get_logger().warn(f"Could not clear faults: {exc}")

    # ── Gripper ───────────────────────────────────────────────────────────────

    def _set_gripper(self, value: float):
        cmd = Base_pb2.GripperCommand()
        finger = cmd.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = clamp(value, 0.0, 1.0)
        cmd.mode = Base_pb2.GRIPPER_POSITION
        try:
            self._base.SendGripperCommand(cmd)
            self.get_logger().info(f"Gripper → {value:.2f}")
        except Exception as exc:
            self.get_logger().error(f"Gripper failed: {exc}")

    # ── Grab sequence ─────────────────────────────────────────────────────────

    def _grab_sequence(self):
        """
        Runs in its own thread.

        The fault was caused by twist and reach_pose being sent simultaneously.
        Fix: explicitly switch servoing mode before changing command type,
        and wait enough time for the control loop to stop sending twist first.

        Sequence:
          1. Set STATE_GRABBING → control loop stops sending twist
          2. Wait 3 control cycles (~200ms at 15Hz) for loop to stop
          3. Send zero twist + switch to SINGLE_LEVEL_SERVOING (position mode)
          4. Clear any faults
          5. Move down + apply X offset
          6. Close gripper
          7. Move back up
          8. Switch back to LOW_LEVEL_SERVOING (velocity mode)
          9. Set STATE_GRABBED
        """
        # Step 1: Set state FIRST — control loop checks this and stops twist
        self._state = STATE_GRABBING
        self.get_logger().info("Grab triggered — stopping tracking...")

        # Step 2: Wait for control loop to definitely see the state change
        # At 15Hz, one cycle = 67ms. Wait 3 cycles = ~200ms to be safe.
        control_cycle_s = 1.0 / max(self._control_rate_hz, 1.0)
        time.sleep(control_cycle_s * 3)

        # Step 3: Send zero twist then switch to position mode
        self._send_zero_twist(force=True)
        time.sleep(0.05)

        # Step 4: Clear any faults accumulated during tracking
        self._clear_faults()

        # Step 5: Read current pose
        try:
            current = self._read_pose()
        except Exception as exc:
            self.get_logger().error(f"Could not read pose: {exc}")
            self._state = STATE_TRACKING
            return

        grab_x    = current.x + self._gripper_x_offset
        grab_y    = current.y
        grab_z    = max(self._ws_min_z, current.z - self._grab_descent_m)
        retreat_z = current.z

        self.get_logger().info(
            f"Descending: ({current.x:.3f},{current.y:.3f},{current.z:.3f}) "
            f"→ ({grab_x:.3f},{grab_y:.3f},{grab_z:.3f})"
        )

        # Step 6: Move down (X offset + Z descent simultaneously)
        ok = self._reach_pose(CartesianPose(
            x=grab_x, y=grab_y, z=grab_z,
            theta_x=current.theta_x,
            theta_y=current.theta_y,
            theta_z=current.theta_z,
        ))
        if not ok:
            self.get_logger().error("Descent failed — aborting grab")
            self._clear_faults()
            self._state = STATE_TRACKING
            return

        # Step 7: Close gripper
        self.get_logger().info("Closing gripper...")
        self._set_gripper(self._gripper_close)
        time.sleep(0.8)

        # Step 8: Move back up
        self.get_logger().info(f"Retreating: z={grab_z:.3f} → z={retreat_z:.3f}")
        self._reach_pose(CartesianPose(
            x=grab_x, y=grab_y, z=retreat_z,
            theta_x=current.theta_x,
            theta_y=current.theta_y,
            theta_z=current.theta_z,
        ))

        # # Step 9: Stay in position mode (arm holds still with ball)
        # # We do NOT switch back to LOW_LEVEL_SERVOING here because
        # # we don't want the arm moving while holding the ball.
        # self._state = STATE_GRABBED
        # self.get_logger().info(
        #     "Grabbed! Press 'o' to open gripper, 'r' to reset."
        # )

        # Step 9: Move to box hover position (above the box)
        self.get_logger().info(
            f"Moving to box: ({BOX_POSE.x:.3f},{BOX_POSE.y:.3f},{BOX_POSE.z:.3f})"
        )
        ok = self._reach_pose(BOX_POSE)
        if not ok:
            self.get_logger().error("Failed to reach box — dropping ball here")

        # Step 10: Descend into box
        box_drop_z = max(self._ws_min_z, BOX_POSE.z - BOX_DESCENT_M)
        self.get_logger().info(f"Descending into box: z={BOX_POSE.z:.3f} → z={box_drop_z:.3f}")
        self._reach_pose(CartesianPose(
            x=BOX_POSE.x, y=BOX_POSE.y, z=box_drop_z,
            theta_x=BOX_POSE.theta_x,
            theta_y=BOX_POSE.theta_y,
            theta_z=BOX_POSE.theta_z,
        ))

        # Step 11: Open gripper to drop ball
        self.get_logger().info("Dropping ball...")
        self._set_gripper(self._gripper_open)
        time.sleep(0.5)   # wait for ball to drop

        # Step 12: Rise back up out of box
        self.get_logger().info("Rising out of box...")
        self._reach_pose(BOX_POSE)

        # Step 13: Return to tracking start pose
        self.get_logger().info("Returning to start pose...")
        self._reach_pose(self._fixed_pose)

        # Step 14: Resume tracking
        self._prev_err_xy[:] = 0.0
        self._prev_err_z = 0.0
        self._state = STATE_TRACKING
        self.get_logger().info("Back at start — resuming tracking!")

    def _process_frame(self):
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        valid = [c for c in contours if self._valid_ball(c)]

        if valid:
            largest = max(valid, key=cv2.contourArea)
            (px_f, py_f), radius = cv2.minEnclosingCircle(largest)
            px, py   = int(px_f), int(py_f)
            diam_px  = 2.0 * float(radius)

            with self._pose_lock:
                pose = self._measured_pose or self._fixed_pose

            R       = euler_to_R(pose.theta_x, pose.theta_y, pose.theta_z)
            t       = np.array([pose.x, pose.y, pose.z])
            p_world = cam_to_world(
                pixel_to_cam(px, py, self._fx, self._fy, self._cx, self._cy, self._camera_height),
                R, t
            )

            with self._detection_lock:
                self._ball_world    = p_world
                self._ball_diam_px  = diam_px
                self._last_det_time = time.monotonic()

            # ── Visualize ─────────────────────────────────────────────────
            color = {
                STATE_TRACKING: (0,   255, 0),
                STATE_GRABBING: (0,   165, 255),
                STATE_GRABBED:  (255, 0,   0),
            }.get(self._state, (255, 255, 255))

            cx_i, cy_i = int(self._cx), int(self._cy)
            cv2.circle(frame, (px, py), int(radius), color, 2)
            cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
            cv2.drawMarker(frame, (cx_i, cy_i), (255, 255, 0),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.line(frame, (cx_i, cy_i), (px, py), (255, 255, 255), 1)
            cv2.putText(frame,
                        f"world=({p_world[0]:.3f},{p_world[1]:.3f},{p_world[2]:.3f})",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(frame,
                        f"diam={diam_px:.1f}px  target={self._target_diam_px:.1f}px",
                        (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(frame, f"State: {self._state}",
                        (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if self._state == STATE_TRACKING:
                cv2.putText(frame, "Press 'g' to grab",
                            (20, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        else:
            now = time.monotonic()
            if now - self._last_no_ball_warn > 1.0:
                self.get_logger().warn("No ball detected")
                self._last_no_ball_warn = now
            cv2.putText(frame, "No ball detected",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"State: {self._state}",
                        (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Ball Tracker", frame)
        cv2.imshow("Mask", mask)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("g") and self._state == STATE_TRACKING:
            self.get_logger().info("'g' pressed — starting grab")
            self._manual_grab = True

        elif key == ord("o"):
            self.get_logger().info("Opening gripper...")
            self._set_gripper(self._gripper_open)
            if self._state == STATE_GRABBED:
                # Switch back to velocity mode so tracking resumes
                self._state = STATE_TRACKING

        elif key == ord("r"):
            self.get_logger().info("Resetting...")
            self._state = STATE_TRACKING
            self._set_gripper(self._gripper_open)
            self._clear_faults()
            self._prev_err_xy[:] = 0.0
            self._prev_err_z = 0.0
            # Use position mode to return to start, then switch back to velocity
            try:
                self._reach_pose(self._fixed_pose)
            except Exception as exc:
                self.get_logger().error(f"Reset failed: {exc}")

        elif key == ord("q"):
            rclpy.shutdown()

    def _control_step(self):
        now = time.monotonic()
        dt  = max(now - self._prev_ctrl_t, 1e-3)
        self._prev_ctrl_t = now

        if not self._first_step and dt > 0.3:
            self.get_logger().warn(f"Control step late: dt={dt:.3f}s")
        self._first_step = False

        # ── Only send twist in TRACKING state ─────────────────────────────
        # GRABBING and GRABBED both halt twist completely.
        # This check is the primary guard against the race condition.
        if self._state != STATE_TRACKING:
            self._send_zero_twist()
            return

        # Grab key pressed — hand off to grab thread
        if self._manual_grab:
            self._manual_grab = False
            threading.Thread(target=self._grab_sequence, daemon=True).start()
            return

        # Read current pose
        try:
            pose = self._read_pose()
        except Exception as exc:
            self.get_logger().warn(f"Pose read failed: {exc}")
            return

        with self._pose_lock:
            self._measured_pose = pose

        with self._detection_lock:
            age      = now - self._last_det_time
            ball     = None if self._ball_world is None else self._ball_world.copy()
            diam_px  = self._ball_diam_px

        # No ball or stale detection — stop
        if ball is None or diam_px is None or age > self._stale_timeout_s:
            self._prev_err_xy[:] = 0.0
            self._prev_err_z = 0.0
            self._send_zero_twist()
            return

        # ── XY PD control (keep ball centered in frame) ───────────────────
        err_x  = float(ball[0] - pose.x)
        err_y  = float(ball[1] - pose.y)
        err_xy = np.array([err_x, err_y])

        if math.hypot(err_x, err_y) < self._xy_deadband_m:
            self._prev_err_xy = err_xy
            vx = vy = 0.0
        else:
            d_xy = (err_xy - self._prev_err_xy) / dt
            self._prev_err_xy = err_xy
            vx = clamp(
                self._kp_xy * err_x + self._kd_xy * float(d_xy[0]),
                -self._max_speed_xy, self._max_speed_xy
            )
            vy = clamp(
                self._kp_xy * err_y + self._kd_xy * float(d_xy[1]),
                -self._max_speed_xy, self._max_speed_xy
            )

        # ── Z PD control (keep constant distance via ball size in pixels) ─
        # Larger diameter = ball closer = move arm up
        # Smaller diameter = ball further = move arm down
        err_z = float(diam_px - self._target_diam_px)
        if abs(err_z) < self._z_deadband_px:
            self._prev_err_z = err_z
            vz = 0.0
        else:
            d_z = (err_z - self._prev_err_z) / dt
            self._prev_err_z = err_z
            vz = clamp(
                self._kp_z * err_z + self._kd_z * d_z,
                -self._max_speed_z, self._max_speed_z
            )

        # ── Workspace safety clamps ───────────────────────────────────────
        if pose.x <= self._ws_min_x and vx < 0: vx = 0.0
        if pose.x >= self._ws_max_x and vx > 0: vx = 0.0
        if pose.y <= self._ws_min_y and vy < 0: vy = 0.0
        if pose.y >= self._ws_max_y and vy > 0: vy = 0.0
        if pose.z <= self._ws_min_z and vz < 0: vz = 0.0
        if pose.z >= self._ws_max_z and vz > 0: vz = 0.0

        self._send_twist(vx, vy, vz)

    # ── Kortex helpers ────────────────────────────────────────────────────────

    def _log_limits(self):
        try:
            lim = self._control_config.GetKinematicHardLimits()
            self._hard_limits = lim
            self.get_logger().info(
                f"Hard limits: linear={lim.twist_linear:.3f}m/s "
                f"angular={lim.twist_angular:.1f}deg/s"
            )
        except Exception as exc:
            self.get_logger().error(f"Could not read limits: {exc}")

    def _set_soft_limits(self):
        try:
            lim = self._control_config.GetKinematicHardLimits()
            self._hard_limits = lim
        except Exception as exc:
            self.get_logger().error(f"Could not read hard limits: {exc}")
            return

        ll = ControlConfig_pb2.TwistLinearSoftLimit()
        ll.twist_linear_soft_limit = lim.twist_linear
        al = ControlConfig_pb2.TwistAngularSoftLimit()
        al.twist_angular_soft_limit = lim.twist_angular

        for mode in (
            ControlConfig_pb2.CARTESIAN_JOYSTICK,
            ControlConfig_pb2.CARTESIAN_TRAJECTORY,
        ):
            ll.control_mode = al.control_mode = mode
            try:
                self._control_config.SetTwistLinearSoftLimit(ll)
                self._control_config.SetTwistAngularSoftLimit(al)
            except Exception as exc:
                self.get_logger().warn(f"Soft limit failed for mode {mode}: {exc}")

    def _read_pose(self) -> CartesianPose:
        p = self._base.GetMeasuredCartesianPose()
        return CartesianPose(
            x=p.x, y=p.y, z=p.z,
            theta_x=p.theta_x, theta_y=p.theta_y, theta_z=p.theta_z,
        )

    def _send_twist(self, vx: float, vy: float, vz: float):
        cmd = Base_pb2.TwistCommand()
        cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        if hasattr(cmd, "duration"):
            cmd.duration = 0
        cmd.twist.linear_x  = vx
        cmd.twist.linear_y  = vy
        cmd.twist.linear_z  = vz
        cmd.twist.angular_x = 0.0
        cmd.twist.angular_y = 0.0
        cmd.twist.angular_z = 0.0
        try:
            self._base.SendTwistCommand(cmd)
            self._twist_is_zero = (vx == vy == vz == 0.0)
        except Exception as exc:
            self.get_logger().warn(f"Twist failed: {exc} — clearing faults")
            self._clear_faults()

    def _send_zero_twist(self, force: bool = False):
        if self._twist_is_zero and not force:
            return
        self._send_twist(0.0, 0.0, 0.0)

    def _reach_pose(self, pose: CartesianPose) -> bool:
        """Send a reach_pose action and wait for completion."""
        done   = threading.Event()
        abort  = {"val": False}

        def cb(n):
            if n.action_event == Base_pb2.ACTION_ABORT:
                abort["val"] = True
            if n.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
                done.set()

        handle = self._base.OnNotificationActionTopic(
            cb, Base_pb2.NotificationOptions()
        )
        action = Base_pb2.Action()
        action.name = "move"
        action.application_data = ""
        tp = action.reach_pose.target_pose
        tp.x = pose.x; tp.y = pose.y; tp.z = pose.z
        tp.theta_x = pose.theta_x
        tp.theta_y = pose.theta_y
        tp.theta_z = pose.theta_z
        action.reach_pose.constraint.speed.translation = (
            self._hard_limits.twist_linear if self._hard_limits else 0.5
        )
        action.reach_pose.constraint.speed.orientation = (
            self._hard_limits.twist_angular if self._hard_limits else 100.0
        )

        try:
            self._base.ExecuteAction(action)
        except Exception as exc:
            self.get_logger().error(f"ExecuteAction failed: {exc}")
            self._base.Unsubscribe(handle)
            return False

        ok = done.wait(self._action_timeout_s)
        self._base.Unsubscribe(handle)
        return ok and not abort["val"]

    def _valid_ball(self, c) -> bool:
        a = cv2.contourArea(c)
        if a < self._min_area:
            return False
        p = cv2.arcLength(c, True)
        if p == 0:
            return False
        return (4 * math.pi * a / p ** 2) >= self._min_circ

    def destroy_node(self):
        self.get_logger().info("Shutting down...")
        try:
            self._send_zero_twist(force=True)
        except Exception:
            pass
        try:
            self._cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        self._session.__exit__(None, None, None)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BallRealtimeTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
