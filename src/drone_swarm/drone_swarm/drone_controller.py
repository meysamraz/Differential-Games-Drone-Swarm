"""
Quadrotor flight controller – works for any single drone instance.

Topic names are RELATIVE so ROS2 namespace routing handles multi-drone:
  Node in namespace /drone1  →  subscribes odom       = /drone1/odom
                               subscribes cmd_vel     = /drone1/cmd_vel
                               subscribes wind_force  = /drone1/wind_force
                               publishes  cmd_force   = /drone1/cmd_force

Parameters
----------
hover_altitude : float (default 0.0)
    Altitude setpoint in metres.
noise_sigma : float (default 0.0)
    Itô noise intensity σ.  Force std = MASS * σ / sqrt(DT) per axis.
    Matches SIGMA = [0.08, 0.15, 0.22] from the differential-game simulation.
delay_steps : int (default 0)
    Actuation delay in control cycles (DT = 0.01 s each).
    delay_steps=3 → 30 ms delay.  Smith predictor in formation_controller
    compensates for this.
"""

import math
import random
import collections

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench, Twist
from nav_msgs.msg import Odometry


# Physical constants
MASS        = 1.04
G           = 9.81
HOVER_FORCE = MASS * G   # ≈ 10.2 N

# Controller gains
KP_Z   = 9.0
KD_Z   = 5.0
KP_V   = 2.8
KP_YAW = 0.18

# Safety limits
F_Z_MAX   = HOVER_FORCE * 3.0
F_XY_MAX  = HOVER_FORCE * 0.6
T_Z_MAX   = 2.0
ALT_SPEED = 1.0

DT = 0.01   # 100 Hz


class DroneController(Node):

    def __init__(self):
        super().__init__('drone_controller')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('hover_altitude', 0.0)
        self.declare_parameter('noise_sigma',    0.0)
        self.declare_parameter('delay_steps',    0)

        self.target_z    = self.get_parameter('hover_altitude').get_parameter_value().double_value
        self.noise_sigma = self.get_parameter('noise_sigma').get_parameter_value().double_value
        self.delay_steps = self.get_parameter('delay_steps').get_parameter_value().integer_value

        # ── Actuation delay buffer ─────────────────────────────────────────
        # FIFO of length delay_steps+1.  We append each new Wrench to the
        # right; reading deque[0] gives the command computed delay_steps
        # cycles ago.  delay_steps=0 → immediate (buf_size=1, reads same
        # frame's command).
        buf_size = self.delay_steps + 1
        self.delay_buf = collections.deque(
            [Wrench()] * buf_size, maxlen=buf_size
        )

        # ── Publishers / subscribers ───────────────────────────────────────
        self.force_pub = self.create_publisher(Wrench, 'cmd_force', 10)
        self.create_subscription(Odometry, 'odom',       self._odom_cb,    10)
        self.create_subscription(Twist,    'cmd_vel',    self._cmd_vel_cb, 10)
        self.create_subscription(Wrench,   'wind_force', self._wind_cb,    10)

        # ── State ──────────────────────────────────────────────────────────
        self.pos_z = self.vel_x = self.vel_y = self.vel_z = 0.0
        self.yaw = self.yaw_rate = 0.0
        self.state_received = False

        self.cmd_vx = self.cmd_vy = self.cmd_vz = self.cmd_yaw_rate = 0.0

        # Latest wind force (world frame, N) from wind_node
        self.wind_fx = self.wind_fy = 0.0

        self.create_timer(DT, self._control_loop)

        self.get_logger().info(
            f'Controller ready  '
            f'hover={self.target_z:.1f} m  '
            f'σ={self.noise_sigma}  '
            f'delay={self.delay_steps}×{DT*1000:.0f} ms'
        )

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self.pos_z = msg.pose.pose.position.z

        q    = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)

        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
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

    def _wind_cb(self, msg: Wrench):
        self.wind_fx = msg.force.x
        self.wind_fy = msg.force.y

    # ── Control loop (100 Hz) ──────────────────────────────────────────────

    def _control_loop(self):
        if not self.state_received:
            self.force_pub.publish(Wrench())
            return

        # ── Altitude setpoint integration ──────────────────────────────
        self.target_z += self.cmd_vz * ALT_SPEED * DT
        self.target_z  = max(0.0, min(15.0, self.target_z))

        # ── Z: PD + gravity feedforward ────────────────────────────────
        fz = HOVER_FORCE + KP_Z * (self.target_z - self.pos_z) + KD_Z * (-self.vel_z)
        fz = max(0.0, min(F_Z_MAX, fz))

        # ── XY: velocity P (world frame) ───────────────────────────────
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        cmd_vx_w = self.cmd_vx * cy - self.cmd_vy * sy
        cmd_vy_w = self.cmd_vx * sy + self.cmd_vy * cy
        fx = max(-F_XY_MAX, min(F_XY_MAX, MASS * KP_V * (cmd_vx_w - self.vel_x)))
        fy = max(-F_XY_MAX, min(F_XY_MAX, MASS * KP_V * (cmd_vy_w - self.vel_y)))

        # ── Yaw: rate P ────────────────────────────────────────────────
        tz = max(-T_Z_MAX, min(T_Z_MAX, KP_YAW * (self.cmd_yaw_rate - self.yaw_rate)))

        # ── Itô noise ──────────────────────────────────────────────────
        # Models actuator vibration and aerodynamic turbulence.
        # Derivation: simulation adds velocity noise dv ~ N(0, σ²·dt).
        # In force: F = MASS·dv/dt = MASS·σ·randn()/√dt.
        # Applied only to XY — altitude PD suppresses vertical noise quickly.
        if self.noise_sigma > 0.0:
            scale = MASS * self.noise_sigma / math.sqrt(DT)
            fx += random.gauss(0.0, scale)
            fy += random.gauss(0.0, scale)

        # ── Actuation delay ────────────────────────────────────────────
        # Buffer the freshly computed control Wrench.
        # Reading deque[0] returns the command from delay_steps cycles ago.
        ctrl = Wrench()
        ctrl.force.x  = fx
        ctrl.force.y  = fy
        ctrl.force.z  = fz
        ctrl.torque.z = tz
        self.delay_buf.append(ctrl)
        delayed = self.delay_buf[0]

        # ── Apply delayed control + undelayed wind ─────────────────────
        # Wind is environmental — it acts on the drone regardless of
        # what command was sent, so it is NOT delayed.
        out = Wrench()
        out.force.x  = delayed.force.x + self.wind_fx
        out.force.y  = delayed.force.y + self.wind_fy
        out.force.z  = delayed.force.z
        out.torque.z = delayed.torque.z
        self.force_pub.publish(out)


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
