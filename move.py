#!/usr/bin/env python3

# --- MUST be first: patch collections for old protobuf (Python 3.10+) ---
import collections
import collections.abc
import os
import sys
import time

_COLLECTIONS_COMPAT_ATTRS = (
    "Mapping",
    "MutableMapping",
    "Sequence",
    "MutableSequence",
    "Set",
    "MutableSet",
    "Iterable",
)
for _name in _COLLECTIONS_COMPAT_ATTRS:
    if not hasattr(collections, _name) and hasattr(collections.abc, _name):
        setattr(collections, _name, getattr(collections.abc, _name))
# -----------------------------------------------------------------------


def _add_active_venv_site_packages():
    venv = os.environ.get("VIRTUAL_ENV")
    if not venv:
        return

    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = os.path.join(venv, "lib", py_version, "site-packages")
    if os.path.isdir(site_packages) and site_packages not in sys.path:
        sys.path.insert(0, site_packages)


_add_active_venv_site_packages()

import math
import threading
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from rclpy.node import Node

from kortex_api.RouterClisent import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.TCPTransport import TCPTransport
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
from kortex_api.autogen.messages import Base_pb2, ControlConfig_pb2, Session_pb2


# ── Physical setup constants ──────────────────────────────────────────────────
GOAL_LINE_X_MIN = 0.231
GOAL_LINE_X_MAX = 0.522
GOAL_LINE_Y     = 0.036
GOAL_LINE_Z     = 0.007

EE_THETA_X = 0
EE_THETA_Y = 180
EE_THETA_Z = 180

HOME = {
    "x":       0.351,
    "y":      -0.488,
    "z":       0.331,
    "theta_x": -25,
    "theta_y": 180,
    "theta_z": 180,
}
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
        self._router: Optional[RouterClient] = None
        self._session_manager: Optional[SessionManager] = None

    def __enter__(self):
        self._transport = TCPTransport()
        self._transport.connect(self._config.ip, self._config.port)
        self._router = RouterClient(self._transport, self._error_callback)

        self._session_manager = SessionManager(self._router)
        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = self._config.username
        session_info.password = self._config.password
        session_info.session_inactivity_timeout = self._config.session_inactivity_timeout_ms
        session_info.connection_inactivity_timeout = self._config.connection_inactivity_timeout_ms

        self._session_manager.CreateSession(session_info)
        return self._router

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._session_manager is not None:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000
            self._session_manager.CloseSession(router_options)

        if self._transport is not None:
            self._transport.disconnect()

    @staticmethod
    def _error_callback(exception):
        print(f"Kortex router error: {exception}")


def control_mode_name(mode: int) -> str:
    try:
        return ControlConfig_pb2.ControlMode.Name(mode)
    except ValueError:
        return f"UNKNOWN_CONTROL_MODE_{mode}"


class KinovaPoseControllerNode(Node):
    def __init__(self):
        super().__init__("kinova_pose_controller")

        self.declare_parameter("robot_ip", "192.168.1.10")
        self.declare_parameter("robot_port", 10000)
        self.declare_parameter("username", "admin")
        self.declare_parameter("password", "admin")
        self.declare_parameter("action_timeout_s", 20.0)

        config = KortexConnectionConfig(
            ip=self.get_parameter("robot_ip").get_parameter_value().string_value,
            port=self.get_parameter("robot_port").get_parameter_value().integer_value,
            username=self.get_parameter("username").get_parameter_value().string_value,
            password=self.get_parameter("password").get_parameter_value().string_value,
        )
        self._action_timeout_s = self.get_parameter("action_timeout_s").get_parameter_value().double_value

        self._session = KortexSession(config)
        self.get_logger().info(f"Connecting to Kinova Gen3 at {config.ip}:{config.port}...")
        router = self._session.__enter__()
        self.get_logger().info("Kortex session established")
        self._base = BaseClient(router)
        self._control_config = ControlConfigClient(router)
        self._hard_limits = None

        self._move_lock = threading.Lock()

        # ── Subscribe to intercept point ──────────────────────────────────
        # msg.x = predicted X where ball crosses goal line
        self._intercept_sub = self.create_subscription(
            Point,
            "intercept_point",
            self._on_intercept_point,
            10
        )

        # ── Subscribe to home command ─────────────────────────────────────
        # Any message on this topic sends arm back to home position
        # Published by perception_node when 'r' is pressed
        self._home_sub = self.create_subscription(
            Bool,
            "go_home",
            self._on_go_home,
            10
        )

        self._ee_pose_timer = self.create_timer(1.0, self._log_measured_cartesian_pose)

        self._set_cartesian_soft_limits_to_hard_limits()
        self._log_kinematic_constraints()
        self.get_logger().info(
            f"Connected to Kinova Gen3 at {config.ip}:{config.port}.\n"
            f"Listening on 'intercept_point' (block) and 'go_home' (reset) topics."
        )

    def run(self):
        """Move to home position on startup and wait."""
        self.get_logger().info("Moving to home position...")
        ok = self._execute_cartesian_action(
            x=HOME["x"],
            y=HOME["y"],
            z=HOME["z"],
            theta_x=HOME["theta_x"],
            theta_y=HOME["theta_y"],
            theta_z=HOME["theta_z"],
        )
        if ok:
            self.get_logger().info("Home position reached. Ready for demo.")
        else:
            self.get_logger().error("Failed to reach home position!")

    def _on_intercept_point(self, msg: Point):
        """Move arm to predicted intercept X on goal line."""
        with self._move_lock:
            x_raw     = msg.x
            x_clamped = max(GOAL_LINE_X_MIN, min(GOAL_LINE_X_MAX, x_raw))

            if abs(x_raw - x_clamped) > 0.001:
                self.get_logger().warn(
                    f"Intercept x={x_raw:.3f} out of bounds, "
                    f"clamped to x={x_clamped:.3f}"
                )

            self.get_logger().info(
                f"Intercept received: x={x_clamped:.3f} — moving to block..."
            )

            ok = self._execute_cartesian_action(
                x=x_clamped,
                y=GOAL_LINE_Y,
                z=GOAL_LINE_Z,
                theta_x=EE_THETA_X,
                theta_y=EE_THETA_Y,
                theta_z=EE_THETA_Z,
            )

            if ok:
                self.get_logger().info(f"Block position x={x_clamped:.3f} reached ✓")
            else:
                self.get_logger().error("Failed to reach block position!")

    def _on_go_home(self, msg: Bool):
        """Move arm back to home position for next demo run."""
        with self._move_lock:
            self.get_logger().info("Go home command received — returning to home...")
            ok = self._execute_cartesian_action(
                x=HOME["x"],
                y=HOME["y"],
                z=HOME["z"],
                theta_x=HOME["theta_x"],
                theta_y=HOME["theta_y"],
                theta_z=HOME["theta_z"],
            )
            if ok:
                self.get_logger().info("Home position reached. Ready for next demo run ✓")
            else:
                self.get_logger().error("Failed to reach home position!")

    def destroy_node(self):
        self.get_logger().info("Closing Kortex session...")
        self._session.__exit__(None, None, None)
        return super().destroy_node()

    def _log_measured_cartesian_pose(self):
        try:
            pose = self._base.GetMeasuredCartesianPose()
        except Exception as exc:
            self.get_logger().error(f"Failed to read measured cartesian pose: {exc}")
            return

        self.get_logger().info(
            "EE pose "
            f"x={pose.x:.3f}, y={pose.y:.3f}, z={pose.z:.3f}, "
            f"theta_x={pose.theta_x:.1f}, theta_y={pose.theta_y:.1f}, "
            f"theta_z={pose.theta_z:.1f}"
        )

    def _log_kinematic_constraints(self):
        try:
            hard_limits = self._control_config.GetKinematicHardLimits()
        except Exception as exc:
            self.get_logger().error(f"Failed to query hard kinematic limits: {exc}")
        else:
            self._hard_limits = hard_limits
            self.get_logger().info(
                "Hard limits "
                f"mode={control_mode_name(hard_limits.control_mode)}, "
                f"twist_linear={hard_limits.twist_linear:.3f} m/s, "
                f"twist_angular={hard_limits.twist_angular:.1f} deg/s, "
                f"joint_speed_limits="
                f"{[round(v, 3) for v in hard_limits.joint_speed_limits]}, "
                f"joint_acceleration_limits="
                f"{[round(v, 3) for v in hard_limits.joint_acceleration_limits]}"
            )

        try:
            soft_limits = self._control_config.GetAllKinematicSoftLimits()
        except Exception as exc:
            self.get_logger().error(f"Failed to query soft kinematic limits: {exc}")
            return

        for limits in soft_limits.kinematic_limits_list:
            self.get_logger().info(
                "Soft limits "
                f"mode={control_mode_name(limits.control_mode)}, "
                f"twist_linear={limits.twist_linear:.3f} m/s, "
                f"twist_angular={limits.twist_angular:.1f} deg/s, "
                f"joint_speed_limits="
                f"{[round(v, 3) for v in limits.joint_speed_limits]}, "
                f"joint_acceleration_limits="
                f"{[round(v, 3) for v in limits.joint_acceleration_limits]}"
            )

    def _set_cartesian_soft_limits_to_hard_limits(self):
        try:
            hard_limits = self._control_config.GetKinematicHardLimits()
        except Exception as exc:
            self.get_logger().error(
                f"Failed to query hard limits before setting soft limits: {exc}"
            )
            return

        self._hard_limits = hard_limits

        linear_limit = ControlConfig_pb2.TwistLinearSoftLimit()
        linear_limit.twist_linear_soft_limit = hard_limits.twist_linear

        angular_limit = ControlConfig_pb2.TwistAngularSoftLimit()
        angular_limit.twist_angular_soft_limit = hard_limits.twist_angular

        cartesian_modes = (
            ControlConfig_pb2.CARTESIAN_JOYSTICK,
            ControlConfig_pb2.CARTESIAN_TRAJECTORY,
            ControlConfig_pb2.CARTESIAN_WAYPOINT_TRAJECTORY,
        )

        for mode in cartesian_modes:
            linear_limit.control_mode = mode
            angular_limit.control_mode = mode
            try:
                self._control_config.SetTwistLinearSoftLimit(linear_limit)
                self._control_config.SetTwistAngularSoftLimit(angular_limit)
            except Exception as exc:
                self.get_logger().error(
                    f"Failed to set soft Cartesian limits for "
                    f"mode={control_mode_name(mode)}: {exc}"
                )
                continue

            self.get_logger().info(
                "Set soft Cartesian limits "
                f"mode={control_mode_name(mode)}, "
                f"twist_linear={hard_limits.twist_linear:.3f} m/s, "
                f"twist_angular={hard_limits.twist_angular:.1f} deg/s"
            )

    def _execute_cartesian_action(
        self,
        x: float,
        y: float,
        z: float,
        theta_x: float,
        theta_y: float,
        theta_z: float,
    ) -> bool:
        done_event = threading.Event()

        def notification_callback(notification):
            action_event = notification.action_event
            if action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
                done_event.set()

        notification_handle = self._base.OnNotificationActionTopic(
            notification_callback,
            Base_pb2.NotificationOptions(),
        )

        action = Base_pb2.Action()
        action.name = "ros2_reach_pose"
        action.application_data = ""

        target_pose = action.reach_pose.target_pose
        target_pose.x = x
        target_pose.y = y
        target_pose.z = z
        target_pose.theta_x = theta_x
        target_pose.theta_y = theta_y
        target_pose.theta_z = theta_z

        translation_speed = 0.5
        orientation_speed = 100.0
        if self._hard_limits is not None:
            translation_speed = self._hard_limits.twist_linear
            orientation_speed = self._hard_limits.twist_angular

        action.reach_pose.constraint.speed.translation = translation_speed
        action.reach_pose.constraint.speed.orientation = orientation_speed

        try:
            self._base.ExecuteAction(action)
        except Exception as exc:
            self.get_logger().error(f"ExecuteAction failed: {exc}")
            self._base.Unsubscribe(notification_handle)
            return False

        finished = done_event.wait(self._action_timeout_s)
        self._base.Unsubscribe(notification_handle)
        return finished


def main(args=None):
    rclpy.init(args=args)
    node = KinovaPoseControllerNode()

    try:
        node.run()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
