#!/usr/bin/env python3
"""
Perception node — detects ball, estimates trajectory, publishes intercept ONCE.
Pressing 'r' resets state AND sends arm back to home position.

Workflow per demo run:
    1. Arm is at home (goalkeeper.py already running)
    2. Roll ball toward goal line
    3. Node collects REQUIRED_DETECTIONS frames
    4. Publishes intercept ONCE → arm moves to block
    5. Press 'r' → arm returns home, node resets for next run

Run alongside goalkeeper node:
    Terminal 1: python3 goalkeeper.py
    Terminal 2: python3 perception_node.py
"""

import os
import collections
import collections.abc
import sys

# --- protobuf patch ---
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
# ----------------------

import math
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Bool

from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.TCPTransport import TCPTransport
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Session_pb2


# ── Camera stream ─────────────────────────────────────────────────────────────
ROBOT_IP = "192.168.1.10"
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
RTSP_URL = f"rtsp://admin:admin@{ROBOT_IP}:554/color"

FX = 1024.0
FY = 1024.0
CX = 640.0
CY = 360.0
CAMERA_HEIGHT = 0.41
# ─────────────────────────────────────────────────────────────────────────────


# ── Ball detection tuning ─────────────────────────────────────────────────────
BALL_HSV_LOWER = np.array([100, 80,  20])   # <-- REPLACE with your values
BALL_HSV_UPPER = np.array([130, 255, 255])  # <-- REPLACE with your values

# Minimum blob area in pixels to count as a valid ball detection.
# Increase this if random noise is being detected.
# A tennis ball at ~40cm camera height should be ~2000-5000px area.
# Start at 2000 and increase if still noisy.
MIN_BALL_AREA = 2000

# Morphological kernel size — larger = more aggressive noise removal.
# 5 removes small noise blobs. Increase to 7 or 9 if still noisy.
MORPH_KERNEL_SIZE = 5

# Minimum circularity (0.0 to 1.0) — a perfect circle = 1.0.
# Filters out non-circular blobs like shadows or reflections.
# 0.6 means "must be at least 60% circular". Increase to 0.7 or 0.8
# if non-ball shapes are still being detected.
MIN_CIRCULARITY = 0.6
# ─────────────────────────────────────────────────────────────────────────────


# ── Goal line (must match goalkeeper.py exactly) ──────────────────────────────
GOAL_LINE_Y_WORLD = 0.036
GOAL_LINE_X_MIN   = 0.231
GOAL_LINE_X_MAX   = 0.522
# ─────────────────────────────────────────────────────────────────────────────


# ── Detection settings ────────────────────────────────────────────────────────
# How many frames to collect before predicting intercept.
# At 30fps:
#    5 frames = ~0.17s  (fast but noisier prediction)
#   10 frames = ~0.33s  (good balance)
#   20 frames = ~0.67s  (slow but accurate)
# Start with 8 — tune based on accuracy vs speed tradeoff.
REQUIRED_DETECTIONS = 10
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class KortexConnectionConfig:
    ip: str
    port: int
    username: str
    password: str
    session_inactivity_timeout_ms: int = 60000
    connection_inactivity_timeout_ms: int = 2000


class KortexSession:
    def __init__(self, config: KortexConnectionConfig):
        self._config = config
        self._transport: Optional[TCPTransport] = None
        self._router = None
        self._session_manager = None

    def __enter__(self):
        self._transport = TCPTransport()
        self._transport.connect(self._config.ip, self._config.port)
        self._router = RouterClient(
            self._transport, lambda e: print(f"Kortex error: {e}")
        )
        self._session_manager = SessionManager(self._router)
        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = self._config.username
        session_info.password = self._config.password
        session_info.session_inactivity_timeout = (
            self._config.session_inactivity_timeout_ms
        )
        session_info.connection_inactivity_timeout = (
            self._config.connection_inactivity_timeout_ms
        )
        self._session_manager.CreateSession(session_info)
        return self._router

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._session_manager:
            opts = RouterClientSendOptions()
            opts.timeout_ms = 1000
            self._session_manager.CloseSession(opts)
        if self._transport:
            self._transport.disconnect()


# ── Coordinate helpers ────────────────────────────────────────────────────────

def euler_deg_to_rotation_matrix(tx, ty, tz):
    rx, ry, rz = math.radians(tx), math.radians(ty), math.radians(tz)
    Rx = np.array([[1, 0, 0],
                   [0,  math.cos(rx), -math.sin(rx)],
                   [0,  math.sin(rx),  math.cos(rx)]])
    Ry = np.array([[ math.cos(ry), 0, math.sin(ry)],
                   [0, 1, 0],
                   [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                   [math.sin(rz),  math.cos(rz), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def pixel_to_camera_frame(px, py):
    Z = CAMERA_HEIGHT
    return np.array([-(px - CX) / FX * Z,
                     -(py - CY) / FY * Z,
                     Z])

def camera_to_world(p_cam, R, t):
    return R @ p_cam + t

# ─────────────────────────────────────────────────────────────────────────────


STATE_COLLECTING = "COLLECTING"
STATE_DONE       = "DONE"


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        # ── Read camera starting pose ─────────────────────────────────────
        config = KortexConnectionConfig(
            ip=ROBOT_IP, port=10000,
            username="admin", password="admin"
        )
        self.get_logger().info("Reading camera starting pose from Kinova...")
        self._kortex_session = KortexSession(config)
        router = self._kortex_session.__enter__()
        base   = BaseClient(router)

        # Check current tool configuration
        # tool_config = base.ReadToolConfiguration()
        # self.get_logger().info(
        #     f"Current tool config: x={tool_config.tool_transform.x:.3f}, "
        #     f"y={tool_config.tool_transform.y:.3f}, "
        #     f"z={tool_config.tool_transform.z:.3f}, "
        #     f"theta_x={tool_config.tool_transform.theta_x:.1f}, "
        #     f"theta_y={tool_config.tool_transform.theta_y:.1f}, "
        #     f"theta_z={tool_config.tool_transform.theta_z:.1f}"
        # )

        cam_pose = base.GetMeasuredCartesianPose()
        self.get_logger().info(
            f"Camera pose: x={cam_pose.x:.3f}, y={cam_pose.y:.3f}, "
            f"z={cam_pose.z:.3f}, theta_x={cam_pose.theta_x:.1f}, "
            f"theta_y={cam_pose.theta_y:.1f}, theta_z={cam_pose.theta_z:.1f}"
        )

        self._R = euler_deg_to_rotation_matrix(
            cam_pose.theta_x, cam_pose.theta_y, cam_pose.theta_z
        )
        self._t = np.array([cam_pose.x, cam_pose.y, cam_pose.z])

        self._kortex_session.__exit__(None, None, None)
        self.get_logger().info("Kinova connection closed (pose captured)")

        # ── ROS2 publishers ───────────────────────────────────────────────
        self._intercept_pub = self.create_publisher(Point, "intercept_point", 10)
        self._home_pub      = self.create_publisher(Bool,  "go_home", 10)

        # ── State ─────────────────────────────────────────────────────────
        self._state   = STATE_COLLECTING
        self._history = []

        # ── Morphological kernel (pre-built for efficiency) ───────────────
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
        )

        # ── Open camera ───────────────────────────────────────────────────
        self.get_logger().info(f"Opening camera at {RTSP_URL} ...")
        self._cap = cv2.VideoCapture(RTSP_URL)

        if not self._cap.isOpened():
            self.get_logger().error("Could not open camera stream!")
            return

        ret, test_frame = self._cap.read()
        if not ret or test_frame is None:
            self.get_logger().error("Camera opened but could not read frame!")
            return

        h, w = test_frame.shape[:2]
        self.get_logger().info(
            f"Camera ready. Resolution: {w}x{h}\n"
            f"Detection settings: MIN_BALL_AREA={MIN_BALL_AREA}, "
            f"MORPH_KERNEL={MORPH_KERNEL_SIZE}, "
            f"MIN_CIRCULARITY={MIN_CIRCULARITY}, "
            f"REQUIRED_DETECTIONS={REQUIRED_DETECTIONS}\n"
            f"Keys: 'r' = reset + send arm home | 'q' = quit"
        )

        self._timer = self.create_timer(0.033, self._process_frame)

    def _reset(self):
        """Reset for next demo and send arm home."""
        self._state   = STATE_COLLECTING
        self._history = []

        msg = Bool()
        msg.data = True
        self._home_pub.publish(msg)

        self.get_logger().info(
            "RESET — arm home command sent. "
            "Roll ball when arm reaches home."
        )

    def _is_valid_ball(self, contour):
        """
        Returns True only if contour passes ALL of:
          1. Area > MIN_BALL_AREA         (filters tiny noise)
          2. Circularity > MIN_CIRCULARITY (filters non-circular shapes)
        """
        area = cv2.contourArea(contour)
        if area < MIN_BALL_AREA:
            return False

        # Circularity = 4π * area / perimeter²
        # Perfect circle = 1.0, irregular shapes < 1.0
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
        circularity = 4 * math.pi * area / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY:
            return False

        return True

    def _process_frame(self):
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return

        # ── Detect ball ───────────────────────────────────────────────────
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)

        # More aggressive morphological cleanup:
        # Opening (erode then dilate) removes small noise blobs
        # Closing (dilate then erode) fills gaps in ball mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        ball_detected = False

        # Filter contours by area AND circularity
        valid_contours = [c for c in contours if self._is_valid_ball(c)]

        if valid_contours:
            # Take largest valid contour as the ball
            largest = max(valid_contours, key=cv2.contourArea)
            (px, py), radius = cv2.minEnclosingCircle(largest)
            px, py = int(px), int(py)
            ball_detected = True

            # Pixel → camera frame → world frame
            p_cam   = pixel_to_camera_frame(px, py)
            p_world = camera_to_world(p_cam, self._R, self._t)
            X_world, Y_world = p_world[0], p_world[1]

            # self.get_logger().info(f"pixel=({px},{py}), camera=({p_cam}) world=({p_world})")

            if self._state == STATE_COLLECTING:
                self._history.append((X_world, Y_world))
                n = len(self._history)

                self.get_logger().info(
                    f"[{n}/{REQUIRED_DETECTIONS}] "
                    f"pixel=({px},{py}) "
                    f"world=({X_world:.3f},{Y_world:.3f})"
                )

                # Progress bar
                progress = int((n / REQUIRED_DETECTIONS) * frame.shape[1])
                cv2.rectangle(
                    frame,
                    (0, frame.shape[0] - 20),
                    (progress, frame.shape[0]),
                    (0, 255, 0), -1
                )
                cv2.putText(
                    frame,
                    f"Collecting {n}/{REQUIRED_DETECTIONS}",
                    (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

                if n >= REQUIRED_DETECTIONS:
                    self._predict_and_publish()

            elif self._state == STATE_DONE:
                cv2.putText(
                    frame, "DONE — press 'r' to reset + send arm home",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                )

            cv2.circle(frame, (px, py), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"({X_world:.2f},{Y_world:.2f})",
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        if not ball_detected:
            if self._state == STATE_COLLECTING and len(self._history) > 0:
                self.get_logger().warn(
                    f"Ball lost — resetting history "
                    f"({len(self._history)}/{REQUIRED_DETECTIONS})"
                )
                self._history = []

            label = (
                "No ball detected"
                if self._state == STATE_COLLECTING
                else "DONE — press 'r' to reset + send arm home"
            )
            color = (
                (0, 0, 255)
                if self._state == STATE_COLLECTING
                else (0, 165, 255)
            )
            cv2.putText(frame, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Perception", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            self._reset()
        elif key == ord('q'):
            rclpy.shutdown()

    def _predict_and_publish(self):
        """Fit line, predict intercept X, publish ONCE, enter DONE state."""
        self.get_logger().info(
            f"Collected {len(self._history)} detections — predicting..."
        )

        pts      = np.array(self._history, dtype=np.float32)
        y_values = pts[:, 1]

        if y_values[-1] <= y_values[0]:
            self.get_logger().warn(
                "Ball not moving toward goal line — using center as fallback"
            )
            predicted_x = (GOAL_LINE_X_MIN + GOAL_LINE_X_MAX) / 2.0
        else:
            res = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            vx = float(res[0].item())
            vy = float(res[1].item())
            x0 = float(res[2].item())
            y0 = float(res[3].item())

            if vy == 0 or not np.isfinite(vx / vy):
                self.get_logger().warn("Degenerate trajectory — cannot predict")
                return

            predicted_x = x0 + vx * (GOAL_LINE_Y_WORLD - y0) / vy

            if not np.isfinite(predicted_x):
                self.get_logger().warn("Predicted x is not finite — aborting")
                return

            self.get_logger().info(
                f"Line fit: vx={vx:.4f}, vy={vy:.4f}, "
                f"x0={x0:.4f}, y0={y0:.4f}"
            )

        predicted_x_clamped = max(
            GOAL_LINE_X_MIN, min(GOAL_LINE_X_MAX, predicted_x)
        )

        self.get_logger().info(
            f"Predicted intercept: x={predicted_x:.4f} → "
            f"clamped={predicted_x_clamped:.4f}"
        )

        msg = Point()
        msg.x = float(predicted_x_clamped)
        msg.y = 0.0
        msg.z = 0.0
        self._intercept_pub.publish(msg)

        self.get_logger().info(
            f"Intercept PUBLISHED: x={predicted_x_clamped:.4f} — "
            "goalkeeper moving."
        )

        self._state = STATE_DONE
        self.get_logger().info(
            "State → DONE. Press 'r' to send arm home and reset."
        )

    def destroy_node(self):
        if self._cap.isOpened():
            self._cap.release()
        cv2.destroyAllWindows()
        try:
            self._kortex_session.__exit__(None, None, None)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
