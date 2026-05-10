"""
Quadrotor flight controller – works for any single drone instance.

Topic names are RELATIVE so ROS2 namespace routing handles multi-drone:
  Node in namespace /drone1  →  subscribes odom  = /drone1/odom
                               subscribes cmd_vel = /drone1/cmd_vel
                               publishes cmd_force = /drone1/cmd_force

Parameters
----------
hover_altitude : float (default 0.0)
    Initial altitude setpoint in metres.
    0.0  → drone sits on ground until teleop commands it up   (single-drone mode)
    1.0  → drone auto-hovers at 1 m after spawning           (swarm formation mode)
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench, Twist
from nav_msgs.msg import Odometry


# Physical constants
MASS        = 1.04           # kg  (body 1.0 + 4 rotors × 0.01)
G           = 9.81           # m/s²
HOVER_FORCE = MASS * G       # ≈ 10.2 N

# Controller gains
KP_Z   = 9.0    # altitude P
KD_Z   = 5.0    # altitude D
KP_V   = 2.8    # XY velocity P
KP_YAW = 0.18   # yaw-rate P  (Nm per rad/s)

# Safety limits
F_Z_MAX   = HOVER_FORCE * 3.0
F_XY_MAX  = HOVER_FORCE * 0.6
T_Z_MAX   = 2.0
ALT_SPEED = 1.0   # m/s per unit of cmd_vel.linear.z


class DroneController(Node):
    def __init__(self):
        super().__init__('drone_controller')

        # --- parameter: initial altitude setpoint ---
        self.declare_parameter('hover_altitude', 0.0)
        self.target_z = (
            self.get_parameter('hover_altitude')
                .get_parameter_value().double_value
        )

        # --- publishers / subscribers (relative names → namespaced by launch) ---
        self.force_pub = self.create_publisher(Wrench, 'cmd_force', 10)
        self.create_subscription(Odometry, 'odom',    self._odom_cb,    10)
        self.create_subscription(Twist,    'cmd_vel', self._cmd_vel_cb, 10)

        # State (world frame, filled by _odom_cb)
        self.pos_z    = 0.0
        self.vel_x    = 0.0
        self.vel_y    = 0.0
        self.vel_z    = 0.0
        self.yaw      = 0.0
        self.yaw_rate = 0.0
        self.state_received = False

        # Teleop commands
        self.cmd_vx       = 0.0
        self.cmd_vy       = 0.0
        self.cmd_vz       = 0.0
        self.cmd_yaw_rate = 0.0

        self.create_timer(0.01, self._control_loop)   # 100 Hz

        self.get_logger().info(
            f'Controller ready  hover_altitude={self.target_z:.1f} m  '
            f'– waiting for odom …'
        )

    # ------------------------------------------------------------------
    def _odom_cb(self, msg: Odometry):
        self.pos_z = msg.pose.pose.position.z

        # p3d twist is in BODY frame – rotate XY to world frame
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)

        cy  = math.cos(self.yaw)
        sy  = math.sin(self.yaw)
        bvx = msg.twist.twist.linear.x
        bvy = msg.twist.twist.linear.y
        self.vel_x    = bvx * cy - bvy * sy
        self.vel_y    = bvx * sy + bvy * cy
        self.vel_z    = msg.twist.twist.linear.z
        self.yaw_rate = msg.twist.twist.angular.z

        if not self.state_received:
            self.get_logger().info(f'Odom received  z={self.pos_z:.2f} m – active.')
            self.state_received = True

    def _cmd_vel_cb(self, msg: Twist):
        self.cmd_vx       = msg.linear.x
        self.cmd_vy       = msg.linear.y
        self.cmd_vz       = msg.linear.z
        self.cmd_yaw_rate = msg.angular.z

    # ------------------------------------------------------------------
    def _control_loop(self):
        w = Wrench()

        if not self.state_received:
            self.force_pub.publish(w)   # zero force – let drone settle
            return

        dt = 0.01

        # Altitude setpoint integration (W / S from teleop)
        self.target_z += self.cmd_vz * ALT_SPEED * dt
        self.target_z  = max(0.0, min(15.0, self.target_z))

        # --- Z: PD + gravity feedforward ---
        z_err  = self.target_z - self.pos_z
        vz_err = -self.vel_z
        fz = HOVER_FORCE + KP_Z * z_err + KD_Z * vz_err
        fz = max(0.0, min(F_Z_MAX, fz))

        # --- XY: velocity P (world frame) ---
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        cmd_vx_w = self.cmd_vx * cy - self.cmd_vy * sy
        cmd_vy_w = self.cmd_vx * sy + self.cmd_vy * cy
        fx = MASS * KP_V * (cmd_vx_w - self.vel_x)
        fy = MASS * KP_V * (cmd_vy_w - self.vel_y)
        fx = max(-F_XY_MAX, min(F_XY_MAX, fx))
        fy = max(-F_XY_MAX, min(F_XY_MAX, fy))

        # --- Yaw: rate P ---
        tz = KP_YAW * (self.cmd_yaw_rate - self.yaw_rate)
        tz = max(-T_Z_MAX, min(T_Z_MAX, tz))

        w.force.x  = fx
        w.force.y  = fy
        w.force.z  = fz
        w.torque.z = tz
        self.force_pub.publish(w)


def main(args=None):
    rclpy.init(args=args)
    node = DroneController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
