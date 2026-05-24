"""
Unified Nash formation + surround-and-capture controller.

Three operating modes:

  FORMATION (Nash leader-follower):
    drone1 holds a world anchor; drone2+3 maintain Nash-optimal offsets.
    Formation: triangle | line | v_shape | diamond

  PURSUIT (zero-sum, Isaacs):
    drone1 minimax-pursues the evader; drone2+3 trail in Nash formation.

  SURROUND (cooperative multi-pursuer, new):
    All 3 drones take assigned slots in a geometry centred on the evader.
    Geometry shrinks over time until capture.
    Strategies: triangle_surround | line_blockade | v_intercept | shrinking

    Slot assignment: Hungarian-style brute-force (3! = 6 permutations) at
    strategy start → minimises total travel distance, stable thereafter.
    Capture: any drone within CAPTURE_RADIUS m OR ring collapses to minimum.

/formation_cmd accepts:
  triangle | line | v_shape | diamond | pursuit | return
  triangle_surround | line_blockade | v_intercept | shrinking
"""

import math
import time
import numpy as np
from itertools import permutations
from scipy.linalg import solve_continuous_are

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float64MultiArray


# ── Formation offsets: [drone2, drone3] relative to drone1 ────────────────────
FORMATION_OFFSETS = {
    'triangle': np.array([[-1.0, -1.732], [ 1.0, -1.732]]),
    'line':     np.array([[-2.0,  0.0  ], [ 2.0,  0.0  ]]),
    'v_shape':  np.array([[-2.0, -2.0  ], [ 2.0, -2.0  ]]),
    'diamond':  np.array([[-2.0, -3.0  ], [ 2.0, -3.0  ]]),
}

SURROUND_STRATEGIES = {'triangle_surround', 'line_blockade', 'v_intercept', 'shrinking'}

DRONE_NS    = ['drone1', 'drone2', 'drone3']
ROLES       = ['Leader', 'Follower', 'Scout']
DT          = 0.05   # 20 Hz
SMITH_STEPS = 3
V_MAX       = 1.2    # m/s  formation mode cap
V_SURROUND  = 1.8    # m/s  surround mode cap (faster than evader's 0.75 m/s)
V_PURSUIT   = 1.3    # m/s

Q_POS  = [3.0, 2.0, 1.0]
R_CTRL = [0.8, 0.4, 0.15]

GAMMA = np.array([
    [0.0, 0.3, 0.3],
    [0.8, 0.0, 0.5],
    [0.6, 0.5, 0.0],
])

# ── Surround-and-capture parameters ───────────────────────────────────────────
SURROUND_RADIUS_INIT = 2.2    # m   initial ring radius
SURROUND_RADIUS_MIN  = 1.5    # m   safe standoff — ring stops here
CAPTURE_RADIUS       = 1.2    # m   legacy reference (actual check is geometric)
SHRINK_RATE          = 0.07   # m/s ring closure (~10 s from init to min)
ORBIT_OMEGA          = 0.12   # rad/s orbit rate
LINE_AHEAD_BASE      = 1.5    # m   how far ahead line/V forms
LINE_SPREAD_BASE     = 2.0    # m   initial lateral spread of line/V
SAFETY_RADIUS        = 1.3    # m   hard evader exclusion zone — drones bounce off
DRONE_SEP_MIN        = 1.1    # m   soft inter-drone separation radius
K_REPULSE_EV         = 8.0    # evader repulsion gain
K_REPULSE_DRONE      = 4.0    # inter-drone repulsion gain


# ── Nash solver ───────────────────────────────────────────────────────────────

def _solve_nash():
    A_s = np.array([[0.]])
    B_s = np.array([[1.]])

    P0 = solve_continuous_are(A_s, B_s, [[Q_POS[0]]], [[R_CTRL[0]]])
    K0 = float(P0[0, 0]) / R_CTRL[0]

    K_f = [1.0, 1.0]
    for _ in range(300):
        K_old = K_f.copy()
        for k in range(2):
            i       = k + 1
            j       = 2 - k
            Q_tilde = Q_POS[i] + GAMMA[i, j]
            P = solve_continuous_are(A_s, B_s,
                                     np.array([[Q_tilde]]),
                                     np.array([[R_CTRL[i]]]))
            K_f[k] = float(P[0, 0]) / R_CTRL[i]
        if max(abs(K_f[k] - K_old[k]) for k in range(2)) < 1e-8:
            break

    return [K0, K_f[0], K_f[1]]


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class FormationController(Node):

    def __init__(self):
        super().__init__('formation_controller')

        self.get_logger().info('Solving Nash equilibrium…')
        self.K = _solve_nash()
        self.get_logger().info(
            f'  Leader   K={self.K[0]:.4f}  Q={Q_POS[0]}  R={R_CTRL[0]}\n'
            f'  Follower K={self.K[1]:.4f}  Q={Q_POS[1]}  γ={GAMMA[1,2]}\n'
            f'  Scout    K={self.K[2]:.4f}  Q={Q_POS[2]}  γ={GAMMA[2,1]}'
        )

        # ── Core state ─────────────────────────────────────────────────────
        self.positions  = np.full((3, 2), np.nan)
        self.yaws       = np.zeros(3)
        self.ready      = [False] * 3
        self.evader_pos = None
        self.evader_vel = np.zeros(2)

        # Formation state
        self.mode              = 'formation'
        self.current_formation = 'triangle'
        self.offsets           = FORMATION_OFFSETS['triangle'].copy()
        self.leader_target     = None

        # Surround state
        self.surround_strategy    = None
        self.surround_t0          = None
        self.slot_assignment      = [0, 1, 2]  # slot_assignment[drone_i] = slot_index
        self.captured             = False
        self.surround_R_captured  = None       # frozen R at capture moment
        self.surround_cap_elapsed = None       # frozen elapsed for orbit freeze
        # Stable heading for line/V (slow EMA — unaffected by jinking)
        self.surround_heading     = np.array([1.0, 0.0])
        # Per-drone initial orbit angles for shrinking (no slot-crossing possible)
        self.orbit_init_angles    = [0.0, 2*math.pi/3, 4*math.pi/3]

        # ── Publishers / Subscribers ───────────────────────────────────────
        for i, ns in enumerate(DRONE_NS):
            self.create_subscription(
                Odometry, f'/{ns}/odom',
                lambda msg, idx=i: self._odom_cb(msg, idx), 10,
            )
        self.create_subscription(Odometry, '/evader/odom', self._evader_cb, 10)
        self.create_subscription(String,   '/formation_cmd', self._cmd_cb, 10)

        self.cmd_pubs = [
            self.create_publisher(Twist, f'/{ns}/cmd_vel', 10)
            for ns in DRONE_NS
        ]
        # Publishes [R, min_dist, captured(0/1), elapsed_s] during surround mode
        self.status_pub = self.create_publisher(
            Float64MultiArray, '/surround_status', 10
        )

        self.create_timer(DT, self._loop)
        self.get_logger().info(
            'Ready.  /formation_cmd accepts:\n'
            '  triangle | line | v_shape | diamond | pursuit | return\n'
            '  triangle_surround | line_blockade | v_intercept | shrinking'
        )

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry, i: int):
        self.positions[i, 0] = msg.pose.pose.position.x
        self.positions[i, 1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaws[i] = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.ready[i] = True

    def _evader_cb(self, msg: Odometry):
        new_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        ])
        if self.evader_pos is not None:
            raw_vel = (new_pos - self.evader_pos) / DT
            # EMA low-pass — α=0.10 smooths jinking, α=0.25 was too noisy
            self.evader_vel = 0.10 * raw_vel + 0.90 * self.evader_vel
        self.evader_pos = new_pos

    def _cmd_cb(self, msg: String):
        cmd = msg.data.strip().lower()

        if cmd in FORMATION_OFFSETS:
            self.mode              = 'formation'
            self.current_formation = cmd
            self.offsets           = FORMATION_OFFSETS[cmd].copy()
            self.captured          = False
            if all(self.ready):
                self.leader_target = self.positions[0].copy()
            self.get_logger().info(f'Formation → {cmd}')

        elif cmd == 'pursuit':
            if self.evader_pos is None:
                self.get_logger().warn('Pursuit: no /evader/odom yet.')
                return
            self.mode     = 'pursuit'
            self.captured = False
            self.get_logger().info('Mode → zero-sum pursuit-evasion')

        elif cmd == 'return':
            self.mode     = 'formation'
            self.captured = False
            if all(self.ready):
                self.leader_target = self.positions[0].copy()
            self.get_logger().info(
                f'Mode → formation/{self.current_formation}'
            )

        elif cmd in SURROUND_STRATEGIES:
            if self.evader_pos is None:
                self.get_logger().warn(f'Surround/{cmd}: no /evader/odom yet.')
                return
            if np.any(np.isnan(self.positions)):
                self.get_logger().warn('Surround: drones not ready yet.')
                return
            self.mode               = 'surround'
            self.surround_strategy  = cmd
            self.surround_t0        = time.monotonic()
            self.captured           = False
            self.surround_R_captured  = None
            self.surround_cap_elapsed = None

            # Stable heading for line/V: use evader velocity direction if available,
            # otherwise point from formation centroid toward evader.
            speed = np.linalg.norm(self.evader_vel)
            if speed > 0.1:
                self.surround_heading = (self.evader_vel / speed).copy()
            else:
                centroid = np.mean(self.positions, axis=0)
                to_ev    = self.evader_pos - centroid
                d        = np.linalg.norm(to_ev)
                self.surround_heading = (to_ev / d) if d > 0.1 else np.array([1.0, 0.0])

            # Per-drone orbit angles for shrinking.
            # Drones often start clustered on one side of the evader (formation
            # positions), so naively using actual angles gives targets only 20-30°
            # apart → inter-drone collisions.  Fix: pick 3 evenly-spaced (120°)
            # orbit lanes and assign each drone to its nearest lane.
            if cmd == 'shrinking':
                actual = []
                for i in range(3):
                    diff = self.positions[i] - self.evader_pos
                    d = np.linalg.norm(diff)
                    actual.append(
                        math.atan2(diff[1], diff[0]) if d > 0.15
                        else 2 * math.pi * i / 3
                    )
                # Try all 3 phase offsets for evenly-spaced lanes,
                # pick the one that minimises total angular travel.
                best_cost = float('inf')
                for base_idx in range(3):
                    base = actual[base_idx]
                    lanes = [base + 2 * math.pi * k / 3 for k in range(3)]
                    for perm in permutations(range(3)):
                        cost = sum(
                            abs((lanes[perm[d]] - actual[d] + math.pi)
                                % (2 * math.pi) - math.pi)
                            for d in range(3)
                        )
                        if cost < best_cost:
                            best_cost = cost
                            best_lanes = [lanes[perm[d]] for d in range(3)]
                self.orbit_init_angles = best_lanes
            else:
                # Slot-based assignment for all other strategies
                initial_targets = self._surround_slot_positions(0.0)
                self.slot_assignment = self._assign_slots(initial_targets)

            self.get_logger().info(
                f'Mode → surround/{cmd}  '
                f'heading={self.surround_heading.tolist()}  '
                f'R_init={SURROUND_RADIUS_INIT}'
            )

    # ── Surround helpers ───────────────────────────────────────────────────

    def _surround_radius(self, elapsed: float) -> float:
        return max(SURROUND_RADIUS_MIN,
                   SURROUND_RADIUS_INIT - SHRINK_RATE * elapsed)

    def _surround_slot_positions(self, elapsed: float, eff_elapsed: float = None):
        """
        Returns 3 target positions.

        For triangle/line/v_intercept: positions are indexed by SLOT (0,1,2).
          drone i goes to targets[slot_assignment[i]].

        For shrinking: positions are indexed by DRONE (0,1,2) directly.
          drone i goes to targets[i].  No slot_assignment used — avoids crossing.
        """
        if eff_elapsed is None:
            eff_elapsed = elapsed

        ev = self.evader_pos.copy()
        R  = self._surround_radius(elapsed)
        s  = self.surround_strategy

        if s == 'triangle_surround':
            return [
                ev + R * np.array([math.cos(2 * math.pi * k / 3),
                                   math.sin(2 * math.pi * k / 3)])
                for k in range(3)
            ]

        elif s == 'line_blockade':
            # Use the STABLE heading — not raw evader_vel which jinks every 1.5 s.
            heading = self.surround_heading
            perp    = np.array([-heading[1], heading[0]])
            ahead   = ev + (LINE_AHEAD_BASE + R) * heading
            spread  = max(0.8, LINE_SPREAD_BASE * (R / SURROUND_RADIUS_INIT))
            return [
                ahead + spread * perp,    # slot 0 — left blocker
                ahead,                    # slot 1 — centre blocker
                ahead - spread * perp,    # slot 2 — right blocker
            ]

        elif s == 'v_intercept':
            # Use stable heading — tip in evader's travel direction, wings trailing.
            # Tip is 2.5 m ahead (was 1.5 m) — well outside SAFETY_RADIUS so no
            # tug-of-war between P-control and evader exclusion repulsion.
            heading = self.surround_heading
            perp    = np.array([-heading[1], heading[0]])
            spread  = max(0.8, LINE_SPREAD_BASE * (R / SURROUND_RADIUS_INIT))
            return [
                ev + 2.5 * heading,                           # slot 0 — tip (ahead)
                ev - 1.5 * heading + spread * perp,           # slot 1 — left wing
                ev - 1.5 * heading - spread * perp,           # slot 2 — right wing
            ]

        elif s == 'shrinking':
            # Per-drone orbiting targets — indexed by DRONE i, not slot.
            # Each drone orbits from its own initial angle → no crossing ever.
            return [
                ev + R * np.array([
                    math.cos(self.orbit_init_angles[i] + ORBIT_OMEGA * eff_elapsed),
                    math.sin(self.orbit_init_angles[i] + ORBIT_OMEGA * eff_elapsed),
                ])
                for i in range(3)
            ]

        return [ev.copy() for _ in range(3)]

    def _assign_slots(self, slot_positions):
        """
        Optimal assignment of drones → slots via brute-force (3! = 6).
        Returns assignment[i] = slot_index for drone i.
        """
        pos = self.positions.copy()
        best_cost = float('inf')
        best_perm = [0, 1, 2]
        for perm in permutations(range(3)):
            cost = sum(
                np.linalg.norm(pos[d] - slot_positions[perm[d]])
                for d in range(3)
            )
            if cost < best_cost:
                best_cost = cost
                best_perm = list(perm)
        return best_perm  # best_perm[drone_i] = slot_index

    # ── Capture geometry ──────────────────────────────────────────────────

    def _slot_drone_idx(self, slot: int) -> int:
        """Which drone is assigned to the given slot index?"""
        for i in range(3):
            if self.slot_assignment[i] == slot:
                return i
        return 0

    def _max_angular_gap_deg(self) -> float:
        """Largest angular gap (°) in drone coverage as seen from the evader."""
        angles = []
        for i in range(3):
            diff = self.positions[i] - self.evader_pos
            if np.linalg.norm(diff) > 0.15:
                angles.append(math.atan2(diff[1], diff[0]))
        if len(angles) < 2:
            return 360.0
        angles.sort()
        n = len(angles)
        max_gap = max(
            (angles[(i + 1) % n] - angles[i]) % (2 * math.pi)
            for i in range(n)
        )
        return math.degrees(max_gap)

    def _triangle_area(self) -> float:
        """Area of the triangle formed by the 3 drone positions (m²)."""
        a, b, c = self.positions[0], self.positions[1], self.positions[2]
        return 0.5 * abs(float(np.cross(b - a, c - a)))

    def _point_in_triangle(self) -> bool:
        """True if the evader is inside the drone triangle."""
        p = self.evader_pos
        a, b, c = self.positions[0], self.positions[1], self.positions[2]
        def _cross(o, u, v):
            return float((u[0]-o[0])*(v[1]-o[1]) - (u[1]-o[1])*(v[0]-o[0]))
        d1 = _cross(p, a, b)
        d2 = _cross(p, b, c)
        d3 = _cross(p, c, a)
        neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (neg and pos)

    def _surround_capture_check(self):
        """
        Strategy-specific capture condition.

        Returns (captured: bool, metric: float, threshold: float, label: str)
        Captured when metric < threshold.

        Triangle Surround → Encirclement: evader inside drone triangle
          metric = triangle area (m²), threshold = 2.0 m²
          (capture requires BOTH inside AND area small enough)

        Shrinking Ring → No-Escape: largest angular gap < 150°
          metric = max gap (°), threshold = 150°

        V Intercept → Funnel: wing-drone separation < threshold
          metric = wing separation (m), threshold = 1.8 m

        Line Blockade → Barrier: max gap along the drone line < threshold
          metric = max inter-drone gap (m), threshold = 1.5 m
        """
        s = self.surround_strategy

        if s == 'triangle_surround':
            area = self._triangle_area()
            inside = self._point_in_triangle()
            metric = area if inside else 999.0   # 999 = "not enclosed"
            threshold = 3.5   # ~equilateral triangle at R=1.5m has area≈5.8m²
            return metric < threshold, metric, threshold, 'Triangle area (m²)'

        elif s == 'shrinking':
            gap = self._max_angular_gap_deg()
            threshold = 135.0   # tighter than 150° — ring at 1.5m is well-closed
            return gap < threshold, gap, threshold, 'Max escape gap (°)'

        elif s == 'v_intercept':
            w_a = self._slot_drone_idx(1)   # left wing
            w_b = self._slot_drone_idx(2)   # right wing
            sep = float(np.linalg.norm(
                self.positions[w_a] - self.positions[w_b]
            ))
            threshold = 3.0   # wings 3 m apart at min radius still blocks exit
            return sep < threshold, sep, threshold, 'Wing separation (m)'

        elif s == 'line_blockade':
            # Sort drones by their slot order (slot 0=left, 1=centre, 2=right)
            ordered = [self._slot_drone_idx(slot) for slot in range(3)]
            max_gap = max(
                float(np.linalg.norm(
                    self.positions[ordered[k + 1]] - self.positions[ordered[k]]
                ))
                for k in range(2)
            )
            threshold = 1.5
            return max_gap < threshold, max_gap, threshold, 'Line gap (m)'

        return False, 0.0, 1.0, ''

    # ── Formation control helpers ──────────────────────────────────────────

    def _leader_v(self, p1: np.ndarray) -> np.ndarray:
        if self.leader_target is None:
            return np.zeros(2)
        e = p1 - self.leader_target
        v = -self.K[0] * e
        s = np.linalg.norm(v)
        if s > V_MAX:
            v *= V_MAX / s
        return v

    def _follower_v(self, f_pos: np.ndarray, p1: np.ndarray) -> np.ndarray:
        v = np.zeros((2, 2))
        for k in range(2):
            i   = k + 1
            j_k = 1 - k
            e_p = f_pos[k] - (p1 + self.offsets[k])
            vi  = -self.K[i] * e_p
            rel_err = ((f_pos[k] - f_pos[j_k])
                       - (self.offsets[k] - self.offsets[j_k]))
            vi -= GAMMA[i, 2 - k] * 0.4 * rel_err
            s = np.linalg.norm(vi)
            if s > V_MAX:
                vi *= V_MAX / s
            v[k] = vi
        return v

    # ── Main loop ──────────────────────────────────────────────────────────

    def _loop(self):
        if not all(self.ready):
            return

        if self.leader_target is None:
            self.leader_target = self.positions[0].copy()

        v = np.zeros((3, 2))

        # ── PURSUIT MODE ───────────────────────────────────────────────────
        if self.mode == 'pursuit':
            if self.evader_pos is not None:
                to_ev = self.evader_pos - self.positions[0]
                dist  = np.linalg.norm(to_ev)
                if dist > 0.3:
                    v[0] = V_PURSUIT * to_ev / dist
                self.get_logger().info(
                    f'[PURSUIT]  pursuer→evader = {dist:.2f} m',
                    throttle_duration_sec=1.0,
                )
            fv   = self._follower_v(self.positions[1:].copy(), self.positions[0])
            v[1] = fv[0]
            v[2] = fv[1]
            self._publish(v)
            return

        # ── SURROUND MODE ──────────────────────────────────────────────────
        if self.mode == 'surround':
            if self.evader_pos is None:
                return

            elapsed = time.monotonic() - self.surround_t0

            # After capture: freeze R and orbit angle so the cage holds tight
            # while still tracking the evader's position.
            if self.captured and self.surround_R_captured is not None:
                R           = self.surround_R_captured
                eff_elapsed = self.surround_cap_elapsed
            else:
                R           = self._surround_radius(elapsed)
                eff_elapsed = elapsed

            slots = self._surround_slot_positions(elapsed, eff_elapsed)

            # Slowly update stable heading for line/V strategies (α=0.04).
            # This makes it immune to the 0.65 Hz lateral jink while still
            # tracking genuine heading changes over several seconds.
            if self.surround_strategy in ('line_blockade', 'v_intercept'):
                speed = np.linalg.norm(self.evader_vel)
                if speed > 0.1:
                    new_h = self.evader_vel / speed
                    self.surround_heading = (
                        0.04 * new_h + 0.96 * self.surround_heading
                    )
                    h_norm = np.linalg.norm(self.surround_heading)
                    if h_norm > 0.01:
                        self.surround_heading /= h_norm

            # Nash P-control + partial feedforward (40 % of evader vel).
            is_shrinking = (self.surround_strategy == 'shrinking')
            for i in range(3):
                target = slots[i] if is_shrinking else slots[self.slot_assignment[i]]
                e      = self.positions[i] - target
                vi     = -self.K[i] * e + 0.4 * self.evader_vel

                # Clip formation velocity FIRST so repulsion is never cancelled.
                spd = np.linalg.norm(vi)
                if spd > V_SURROUND:
                    vi *= V_SURROUND / spd

                # Hard evader exclusion zone — applied after clip so it always wins.
                diff_ev = self.positions[i] - self.evader_pos
                d_ev    = np.linalg.norm(diff_ev)
                if 0.05 < d_ev < SAFETY_RADIUS:
                    vi += K_REPULSE_EV * (SAFETY_RADIUS - d_ev) / d_ev * diff_ev

                # Soft inter-drone separation — also after clip.
                for j in range(3):
                    if j == i:
                        continue
                    diff_ij = self.positions[i] - self.positions[j]
                    d_ij    = np.linalg.norm(diff_ij)
                    if 0.05 < d_ij < DRONE_SEP_MIN:
                        vi += K_REPULSE_DRONE * (DRONE_SEP_MIN - d_ij) / d_ij * diff_ij

                # Final hard cap — allows repulsion to push beyond V_SURROUND if needed.
                spd = np.linalg.norm(vi)
                if spd > V_SURROUND * 1.5:
                    vi *= V_SURROUND * 1.5 / spd
                v[i] = vi

            # Strategy-specific capture check
            captured, metric, threshold, label = self._surround_capture_check()
            if captured and not self.captured:
                self.captured             = True
                self.surround_R_captured  = R
                self.surround_cap_elapsed = elapsed
                self.get_logger().info(
                    f'★ CAPTURED!  strategy={self.surround_strategy}  '
                    f'{label}={metric:.2f} < {threshold:.2f}  R={R:.2f} m'
                )

            # Publish [R, metric, captured, elapsed, threshold] for plotter/UI
            status = Float64MultiArray()
            status.data = [
                R, metric, 1.0 if self.captured else 0.0, elapsed, threshold
            ]
            self.status_pub.publish(status)

            self.get_logger().info(
                f'[SURROUND/{self.surround_strategy}]  '
                f'R={R:.2f} m  {label}={metric:.2f}/{threshold:.2f}'
                + ('  ★ CAPTURED — cage tracking evader' if self.captured else ''),
                throttle_duration_sec=1.0,
            )
            self._publish(v)
            return

        # ── FORMATION MODE with Smith predictor ────────────────────────────
        pred_p1 = self.positions[0].copy()
        for _ in range(SMITH_STEPS):
            pred_p1 = pred_p1 + self._leader_v(pred_p1) * DT

        pred_f = self.positions[1:].copy()
        for _ in range(SMITH_STEPS):
            pred_f = pred_f + self._follower_v(pred_f, pred_p1) * DT

        v[0]   = self._leader_v(pred_p1)
        fv     = self._follower_v(pred_f, pred_p1)
        v[1]   = fv[0]
        v[2]   = fv[1]

        e_leader = np.linalg.norm(self.positions[0] - self.leader_target)
        e_f = [
            np.linalg.norm(self.positions[k+1]
                           - (self.positions[0] + self.offsets[k]))
            for k in range(2)
        ]
        self.get_logger().info(
            f'[{self.current_formation:8s}]  '
            f'leader={e_leader:.2f} m  '
            f'follower={e_f[0]:.2f} m  scout={e_f[1]:.2f} m',
            throttle_duration_sec=1.0,
        )
        self._publish(v)

    def _publish(self, v_world: np.ndarray):
        for i in range(3):
            cy, sy = math.cos(self.yaws[i]), math.sin(self.yaws[i])
            cmd = Twist()
            cmd.linear.x = float( v_world[i, 0] * cy + v_world[i, 1] * sy)
            cmd.linear.y = float(-v_world[i, 0] * sy + v_world[i, 1] * cy)
            self.cmd_pubs[i].publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = FormationController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
