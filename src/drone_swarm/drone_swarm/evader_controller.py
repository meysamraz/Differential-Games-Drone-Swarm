"""
Evader drone — adaptive evasion for pursuit and surround modes.

Pursuit mode (single pursuer):
  Isaacs optimal evasion: flee directly away from drone1 + lateral jink.

Surround mode (3 cooperative pursuers):
  Gap-finding evasion: compute angular coverage of all 3 drones as seen
  from the evader, identify the largest angular gap, sprint through it.
  Jinking is added along the escape heading for unpredictability.

The gap-finding creates emergent "mouse finds the hole in the fence"
behaviour: if any drone lags behind its assigned slot, its gap widens
and the evader exploits it — exactly the incentive that drives drones
to minimise their own Nash cost.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String


V_EVADE         = 1.1    # m/s  pursuit mode (vs drone1 at 1.3 m/s)
V_EVADE_SURROUND = 0.75  # m/s  surround mode (drones at 1.6 m/s → ring closes)
JINK_AMP  = 0.50   # m/s  lateral oscillation amplitude
JINK_FREQ = 0.65   # Hz
DT        = 0.05   # 20 Hz

SURROUND_CMDS = {'triangle_surround', 'line_blockade', 'v_intercept', 'shrinking'}


class EvaderController(Node):

    def __init__(self):
        super().__init__('evader_controller')

        self.evader_pos  = np.zeros(2)
        self.evader_yaw  = 0.0
        self.drone_pos   = np.full((3, 2), np.nan)   # all 3 pursuers
        self.ready_ev    = False
        self.mode        = 'idle'   # 'idle' | 'pursuit' | 'surround'
        self.t           = 0.0

        self.create_subscription(Odometry, '/evader/odom', self._evader_cb, 10)

        for i, ns in enumerate(['drone1', 'drone2', 'drone3']):
            self.create_subscription(
                Odometry, f'/{ns}/odom',
                lambda msg, idx=i: self._drone_cb(msg, idx), 10,
            )

        self.create_subscription(String, '/formation_cmd', self._mode_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, '/evader/cmd_vel', 10)
        self.create_timer(DT, self._loop)

        self.get_logger().info(
            'Evader ready.  Activates on "pursuit" or any surround command.'
        )

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _evader_cb(self, msg: Odometry):
        self.evader_pos[0] = msg.pose.pose.position.x
        self.evader_pos[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.evader_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.ready_ev = True

    def _drone_cb(self, msg: Odometry, i: int):
        self.drone_pos[i, 0] = msg.pose.pose.position.x
        self.drone_pos[i, 1] = msg.pose.pose.position.y

    def _mode_cb(self, msg: String):
        cmd = msg.data.strip().lower()
        prev = self.mode
        if cmd == 'pursuit':
            self.mode = 'pursuit'
        elif cmd in SURROUND_CMDS:
            self.mode = 'surround'
        elif cmd in ('return', 'triangle', 'line', 'v_shape', 'diamond'):
            self.mode = 'idle'
        if self.mode != prev:
            self.get_logger().info(f'Evader mode: {prev} → {self.mode}')

    # ── Evasion helpers ────────────────────────────────────────────────────

    def _gap_escape(self) -> np.ndarray:
        """
        Find the largest angular gap between pursuers as seen from the evader
        and return a unit vector pointing through the centre of that gap.
        """
        angles = []
        for i in range(3):
            if np.any(np.isnan(self.drone_pos[i])):
                continue
            diff = self.drone_pos[i] - self.evader_pos
            if np.linalg.norm(diff) > 0.3:
                angles.append(math.atan2(diff[1], diff[0]))

        if len(angles) < 2:
            # Not enough info — flee opposite to current yaw
            return np.array([math.cos(self.evader_yaw + math.pi),
                             math.sin(self.evader_yaw + math.pi)])

        angles.sort()
        n = len(angles)
        best_size  = -1.0
        best_angle = 0.0
        for i in range(n):
            gap = (angles[(i + 1) % n] - angles[i]) % (2 * math.pi)
            if gap > best_size:
                best_size  = gap
                best_angle = angles[i] + gap / 2.0

        return np.array([math.cos(best_angle), math.sin(best_angle)])

    # ── Control loop ───────────────────────────────────────────────────────

    def _loop(self):
        if not self.ready_ev:
            return

        cmd     = Twist()
        v_world = np.zeros(2)

        if self.mode == 'pursuit':
            # Isaacs: flee from drone1, jink laterally
            diff = self.evader_pos - self.drone_pos[0]
            dist = np.linalg.norm(diff)
            if dist > 0.2:
                away = diff / dist
                perp = np.array([-away[1], away[0]])
                jink = JINK_AMP * math.sin(2.0 * math.pi * JINK_FREQ * self.t)
                v_world = V_EVADE * away + jink * perp
            else:
                v_world = V_EVADE * np.array([
                    math.cos(self.evader_yaw + math.pi),
                    math.sin(self.evader_yaw + math.pi),
                ])

        elif self.mode == 'surround':
            # Gap-finding: escape through the widest hole in the ring
            escape = self._gap_escape()
            perp   = np.array([-escape[1], escape[0]])
            jink   = JINK_AMP * math.sin(2.0 * math.pi * JINK_FREQ * self.t)
            v_world = V_EVADE_SURROUND * escape + jink * perp

        if np.linalg.norm(v_world) > 0:
            cy, sy = math.cos(self.evader_yaw), math.sin(self.evader_yaw)
            cmd.linear.x = float( v_world[0] * cy + v_world[1] * sy)
            cmd.linear.y = float(-v_world[0] * sy + v_world[1] * cy)

        self.cmd_pub.publish(cmd)
        self.t += DT


def main(args=None):
    rclpy.init(args=args)
    node = EvaderController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
