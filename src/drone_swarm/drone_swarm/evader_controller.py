"""
Evader drone — stochastic-adaptive evasion.

Policy = heuristic gap escape  +  Ornstein-Uhlenbeck correlated noise.

Pursuit mode (single pursuer):
  Weighted repulsion from all 3 drones — drone1 weight ×3 (primary pursuer),
  drone2+3 weight ×1 (can cut off flanks).  Inverse-distance scaling so
  closer drones feel more threatening.  OU noise on the lateral axis gives
  smooth, unpredictable jinking that is much harder to anticipate than the
  old sinusoidal model.

Surround mode (3 cooperative pursuers):
  Distance-weighted gap escape: each angular gap is scored by the harmonic
  mean distance of its two bounding drones — a gap flanked by far-away
  drones is genuinely safer to sprint through.
  Fallback: when all gaps < GAP_FALLBACK_DEG the evader is nearly enclosed;
  it switches to fleeing the single closest drone (best it can do).
  OU noise adds correlated lateral + small forward perturbation so the
  evader never takes the same path twice.

Ornstein-Uhlenbeck process:
  dx = -θ x dt + σ dW    (zero-mean, τ = 1/θ ≈ 0.33 s)
  Gives smooth, temporally correlated randomness — realistic UAV jitter
  that is statistically unpredictable but physically plausible.

Academic framing:
  "We model the evader as a stochastic-adaptive agent: its base policy
   is a heuristic approximation to the Isaacs gap-finding strategy, and
   its perturbation follows an OU process to simulate adversarial
   uncertainty under realistic UAV dynamics."
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String


# ── Speed limits ──────────────────────────────────────────────────────────────
V_EVADE          = 1.1    # m/s  pursuit mode
V_EVADE_BASE     = 0.85   # m/s  surround — normal cruise (drones close ring reliably)
V_EVADE_SPRINT   = 1.20   # m/s  surround — sprint only through very large gaps
                           #      drones at 1.8 m/s → 0.6 m/s net closure during sprint

DT = 0.05   # 20 Hz control rate

# ── Ornstein-Uhlenbeck noise ──────────────────────────────────────────────────
OU_THETA = 3.0    # mean-reversion rate  (τ = 1/θ ≈ 0.33 s)
OU_SIGMA = 0.25   # noise intensity — smooth jitter, not chaotic wobble

# ── Surround evasion ──────────────────────────────────────────────────────────
GAP_SPRINT_DEG   = 140.0  # only sprint through genuinely large gaps (> 140°)
GAP_FALLBACK_DEG = 75.0   # gap smaller than this → flee closest drone
PURSUIT_WEIGHTS  = [3.0, 1.0, 1.0]   # drone1 is primary pursuer

SURROUND_CMDS = {'triangle_surround', 'line_blockade', 'v_intercept', 'shrinking',
                 'auto_surround'}


class EvaderController(Node):

    def __init__(self):
        super().__init__('evader_controller')

        self.evader_pos = np.zeros(2)
        self.evader_yaw = 0.0
        self.drone_pos  = np.full((3, 2), np.nan)
        self.ready_ev   = False
        self.mode       = 'idle'   # 'idle' | 'pursuit' | 'surround'

        # Ornstein-Uhlenbeck state vector (2-D, zero-mean)
        self.ou_state = np.zeros(2)

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
            'Stochastic-adaptive evader ready.  '
            'OU noise: θ=%.1f  σ=%.2f' % (OU_THETA, OU_SIGMA)
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
        cmd  = msg.data.strip().lower()
        prev = self.mode
        if cmd == 'pursuit':
            self.mode = 'pursuit'
        elif cmd in SURROUND_CMDS:
            self.mode = 'surround'
        elif cmd in ('return', 'triangle', 'line', 'v_shape', 'diamond'):
            self.mode = 'idle'
        if self.mode != prev:
            self.get_logger().info(f'Evader mode: {prev} → {self.mode}')

    # ── Ornstein-Uhlenbeck noise ───────────────────────────────────────────

    def _ou_step(self) -> np.ndarray:
        """
        One OU step: dx = -θ x dt + σ dW.
        Returns the updated noise vector (smoothly correlated across steps).
        """
        dW = np.random.randn(2) * math.sqrt(DT)
        self.ou_state += -OU_THETA * self.ou_state * DT + OU_SIGMA * dW
        return self.ou_state.copy()

    # ── Evasion helpers ────────────────────────────────────────────────────

    def _weighted_pursuit_escape(self) -> np.ndarray:
        """
        Weighted repulsion from all 3 drones.
        Weight = role_weight / distance  so closer + primary drones push harder.
        Returns a unit escape vector.
        """
        net = np.zeros(2)
        total_w = 0.0
        for i in range(3):
            if np.any(np.isnan(self.drone_pos[i])):
                continue
            diff = self.evader_pos - self.drone_pos[i]
            dist = np.linalg.norm(diff)
            if dist < 0.2:
                continue
            w    = PURSUIT_WEIGHTS[i] / (dist + 0.1)
            net += w * diff / dist
            total_w += w

        if total_w < 1e-6:
            return np.array([math.cos(self.evader_yaw + math.pi),
                             math.sin(self.evader_yaw + math.pi)])
        net /= total_w
        norm = np.linalg.norm(net)
        return net / norm if norm > 1e-6 else net

    def _smart_gap_escape(self) -> tuple:
        """
        Distance-weighted gap escape with closest-drone fallback.

        Scoring: each gap's score = gap_angle × harmonic_mean(dist_left, dist_right).
        A wide gap beside far-away drones scores higher — genuinely safer.

        Returns (escape_unit_vector, max_gap_deg).
        """
        valid = []   # (angle_to_drone, distance, drone_index)
        for i in range(3):
            if np.any(np.isnan(self.drone_pos[i])):
                continue
            diff = self.drone_pos[i] - self.evader_pos
            dist = np.linalg.norm(diff)
            if dist > 0.3:
                valid.append((math.atan2(diff[1], diff[0]), dist, i))

        if len(valid) < 2:
            return (np.array([math.cos(self.evader_yaw + math.pi),
                              math.sin(self.evader_yaw + math.pi)]),
                    360.0)

        # Sort by angle
        valid.sort(key=lambda x: x[0])
        n = len(valid)

        best_score = -1.0
        best_angle = 0.0
        best_gap_deg = 0.0

        for i in range(n):
            j   = (i + 1) % n
            a_i, d_i, _ = valid[i]
            a_j, d_j, _ = valid[j]

            gap = (a_j - a_i) % (2 * math.pi)

            # Harmonic mean distance of the two bounding drones:
            # farther drones → safer gap
            h_dist = 2.0 * d_i * d_j / (d_i + d_j + 1e-6)

            score = gap * h_dist
            if score > best_score:
                best_score   = score
                best_angle   = a_i + gap / 2.0
                best_gap_deg = math.degrees(gap)

        # Fallback: if truly enclosed, flee the closest drone
        if best_gap_deg < GAP_FALLBACK_DEG:
            closest = min(valid, key=lambda x: x[1])
            diff    = self.evader_pos - self.drone_pos[closest[2]]
            dist    = np.linalg.norm(diff)
            if dist > 0.05:
                return diff / dist, best_gap_deg

        return (np.array([math.cos(best_angle), math.sin(best_angle)]),
                best_gap_deg)

    # ── Control loop ───────────────────────────────────────────────────────

    def _loop(self):
        if not self.ready_ev:
            return

        cmd     = Twist()
        v_world = np.zeros(2)

        if self.mode == 'pursuit':
            escape = self._weighted_pursuit_escape()
            perp   = np.array([-escape[1], escape[0]])
            ou      = self._ou_step()
            lateral = float(np.dot(ou, perp))
            v_raw   = V_EVADE * escape + lateral * perp
            spd     = np.linalg.norm(v_raw)
            v_world = V_EVADE * v_raw / spd if spd > 0.01 else V_EVADE * escape

        elif self.mode == 'surround':
            escape, gap_deg = self._smart_gap_escape()
            perp = np.array([-escape[1], escape[0]])

            # Sprint when a real gap is open; cruise otherwise.
            # During sprint: reduce OU lateral so the dash is straight.
            sprinting = gap_deg > GAP_SPRINT_DEG
            v_eff     = V_EVADE_SPRINT if sprinting else V_EVADE_BASE
            ou_scale  = 0.3 if sprinting else 1.0   # straighter sprint

            ou      = self._ou_step()
            lateral = float(np.dot(ou, perp)) * ou_scale
            v_raw   = v_eff * escape + lateral * perp
            spd     = np.linalg.norm(v_raw)
            v_world = v_eff * v_raw / spd if spd > 0.01 else v_eff * escape

            self.get_logger().info(
                f'[EVADER]  gap={gap_deg:.0f}°  '
                f'{"SPRINT" if sprinting else ("FALLBACK" if gap_deg < GAP_FALLBACK_DEG else "cruise")}  '
                f'v={v_eff:.2f} m/s',
                throttle_duration_sec=1.0,
            )

        # Transform world-frame velocity → body frame and publish
        if np.linalg.norm(v_world) > 0:
            cy, sy = math.cos(self.evader_yaw), math.sin(self.evader_yaw)
            cmd.linear.x = float( v_world[0] * cy + v_world[1] * sy)
            cmd.linear.y = float(-v_world[0] * sy + v_world[1] * cy)

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = EvaderController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
