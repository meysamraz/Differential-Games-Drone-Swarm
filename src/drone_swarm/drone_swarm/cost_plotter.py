"""
Live cost plotter — unified leader-follower Nash game.

Cost definitions (match formation_controller.py exactly):

  Leader (drone1, formation mode):
    J_0 = Q_0 · ‖p_1 − p_1*‖²
    (p_1* = world anchor; set to leader's position at last formation switch)

  Followers (always, both modes):
    J_i = Q_i · ‖p_i − (p_1 + dᵢ)‖²
        + γᵢⱼ · ‖(pᵢ − pⱼ) − (dᵢ − dⱼ)‖²
    where dᵢ = current formation offset for drone i

  Pursuit progress (drone1, pursuit mode):
    Capture distance = ‖p_1 − p_evader‖  (shown separately)

Three panels:
  Top    : per-agent costs Jᵢ(t)
  Middle : joint cost J(t) = Σᵢ Jᵢ
  Bottom : position errors — leader-to-anchor + followers-to-formation
"""

import math
import time
import threading
import collections

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float64MultiArray


# ── Must match formation_controller.py ────────────────────────────────────────
FORMATION_OFFSETS = {
    'triangle': np.array([[-1.0, -1.732], [ 1.0, -1.732]]),
    'line':     np.array([[-2.0,  0.0  ], [ 2.0,  0.0  ]]),
    'v_shape':  np.array([[-2.0, -2.0  ], [ 2.0, -2.0  ]]),
    'diamond':  np.array([[-2.0, -3.0  ], [ 2.0, -3.0  ]]),
}

DRONE_NS = ['drone1', 'drone2', 'drone3']
ROLES    = ['Leader', 'Follower', 'Scout']
Q_POS    = [3.0, 2.0, 1.0]
GAMMA    = np.array([[0.0,0.3,0.3],[0.8,0.0,0.5],[0.6,0.5,0.0]])

WINDOW_S  = 60.0
UPDATE_HZ = 10
COLORS    = ['#ef4444', '#22c55e', '#3b82f6']
JOINT_COL = '#a78bfa'
BG        = '#0d1117'
GRID_COL  = '#1e2840'
TEXT_COL  = '#c9d1e0'
TICK_COL  = '#8b9dc3'
ORANGE    = '#f97316'


class CostNode(Node):

    def __init__(self):
        super().__init__('cost_plotter')

        self.positions     = np.full((3, 2), np.nan)
        self.evader_pos    = None
        self.leader_target = None          # updated on /formation_cmd
        self.offsets       = FORMATION_OFFSETS['triangle'].copy()
        self.mode          = 'formation'
        self.current_form  = 'triangle'
        # Surround-and-capture status (from /surround_status)
        self.surround_R         = float('nan')
        self.surround_metric    = float('nan')   # strategy-specific (area/gap/sep)
        self.surround_threshold = float('nan')   # capture threshold for metric
        self.surround_captured  = False
        self.surround_strategy  = None
        self._lock         = threading.Lock()

        for i, ns in enumerate(DRONE_NS):
            self.create_subscription(
                Odometry, f'/{ns}/odom',
                lambda msg, idx=i: self._odom_cb(msg, idx), 10,
            )
        self.create_subscription(Odometry, '/evader/odom', self._evader_cb, 10)
        self.create_subscription(String,   '/formation_cmd', self._cmd_cb,  10)
        self.create_subscription(
            Float64MultiArray, '/surround_status', self._surround_cb, 10
        )

        self.get_logger().info('Cost plotter listening…')

    def _odom_cb(self, msg, i):
        with self._lock:
            self.positions[i, 0] = msg.pose.pose.position.x
            self.positions[i, 1] = msg.pose.pose.position.y
            # Initialise leader target on first leader odom
            if i == 0 and self.leader_target is None:
                self.leader_target = self.positions[0].copy()

    def _evader_cb(self, msg):
        with self._lock:
            self.evader_pos = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
            ])

    def _surround_cb(self, msg):
        if len(msg.data) >= 3:
            with self._lock:
                self.surround_R         = msg.data[0]
                self.surround_metric    = msg.data[1]
                self.surround_captured  = (msg.data[2] > 0.5)
                if len(msg.data) >= 5:
                    self.surround_threshold = msg.data[4]

    def _cmd_cb(self, msg):
        cmd = msg.data.strip().lower()
        with self._lock:
            if cmd in FORMATION_OFFSETS:
                self.offsets      = FORMATION_OFFSETS[cmd].copy()
                self.current_form = cmd
                self.mode         = 'formation'
                if not np.any(np.isnan(self.positions[0])):
                    self.leader_target = self.positions[0].copy()
            elif cmd == 'pursuit':
                self.mode = 'pursuit'
            elif cmd == 'return':
                self.mode = 'formation'
                if not np.any(np.isnan(self.positions[0])):
                    self.leader_target = self.positions[0].copy()
            elif cmd in {'triangle_surround', 'line_blockade',
                         'v_intercept', 'shrinking'}:
                self.mode               = 'surround'
                self.surround_strategy  = cmd
                self.surround_R         = float('nan')
                self.surround_metric    = float('nan')
                self.surround_threshold = float('nan')
                self.surround_captured  = False

    def snapshot(self):
        with self._lock:
            if np.any(np.isnan(self.positions)):
                return None
            return (
                self.positions.copy(),
                self.evader_pos.copy() if self.evader_pos is not None else None,
                self.leader_target.copy() if self.leader_target is not None else None,
                self.offsets.copy(),
                self.mode,
                self.current_form,
                self.surround_R,
                self.surround_metric,
                self.surround_threshold,
                self.surround_captured,
                self.surround_strategy,
            )


def _compute_costs(pos, leader_target, offsets):
    """
    Returns [J_0, J_1, J_2].

    J_0 (leader): Q_0 · ‖p1 − p1*‖²
    J_i (follower k, i=k+1):
      Q_i · ‖pᵢ − (p1 + dₖ)‖²  +  γᵢⱼ · ‖(pᵢ−pⱼ) − (dₖ−d_{1−k})‖²
    """
    costs = []

    # Leader
    e0 = pos[0] - leader_target if leader_target is not None else np.zeros(2)
    costs.append(Q_POS[0] * float(e0 @ e0))

    # Followers
    for k in range(2):
        i = k + 1
        j = 2 - k          # the other follower's drone index
        j_k = 1 - k        # the other follower's offset index

        e_i   = pos[i] - (pos[0] + offsets[k])
        rel   = (pos[i] - pos[j]) - (offsets[k] - offsets[j_k])
        Ji    = Q_POS[i] * float(e_i @ e_i) + GAMMA[i, j] * float(rel @ rel)
        costs.append(Ji)

    return costs


def _style_ax(ax, ylabel, title):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TICK_COL, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)
    ax.grid(True, alpha=0.15, color=TICK_COL)
    ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=9)
    ax.set_title(title,   color=TEXT_COL, fontsize=9)
    ax.axhline(0, color='#ffffff', alpha=0.2, lw=0.8, ls='--')


def main(args=None):
    rclpy.init(args=args)
    node = CostNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # ── Figure layout ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 9))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        'Nash Formation — Live Cost Monitor\n'
        r'Leader: $J_0 = Q_0\|e_0\|^2$  '
        r'Followers: $J_i = Q_i\|e_i\|^2 + \gamma_{ij}\|(p_i-p_j)-(d_i-d_j)\|^2$',
        color=TEXT_COL, fontsize=10,
    )

    gs = gridspec.GridSpec(3, 1, hspace=0.45,
                           left=0.09, right=0.97, top=0.88, bottom=0.07)
    ax_ji    = fig.add_subplot(gs[0])
    ax_joint = fig.add_subplot(gs[1])
    ax_err   = fig.add_subplot(gs[2])

    _style_ax(ax_ji,    'Jᵢ(t)',    'Per-Agent Cost')
    _style_ax(ax_joint, 'J(t)',     'Joint Cost  J = Σᵢ Jᵢ')
    _style_ax(ax_err,   '‖eᵢ‖ (m)', 'Position Errors  (leader-to-anchor + followers-to-slot)')
    ax_err.set_xlabel('Time (s)', color=TEXT_COL, fontsize=9)

    # Gain annotations
    for i in range(3):
        if i == 0:
            ann = f'Leader  K={math.sqrt(Q_POS[0]/[0.8,0.4,0.15][0]):.2f}  Q={Q_POS[0]}  R={[0.8,0.4,0.15][0]}  (no coupling)'
        else:
            j   = 3 - i
            gamma = GAMMA[i, j]
            K_i = math.sqrt((Q_POS[i] + gamma) / [0.8,0.4,0.15][i])
            ann = (f'{ROLES[i]}  K={K_i:.2f}  '
                   f'Q={Q_POS[i]}  γ={gamma}  R={[0.8,0.4,0.15][i]}')
        ax_ji.text(0.01, 0.97 - i * 0.12, ann,
                   transform=ax_ji.transAxes, color=COLORS[i],
                   fontsize=7.5, va='top', fontfamily='monospace')

    TEAL = '#2dd4bf'

    # Lines
    lines_ji   = [ax_ji.plot([], [], color=COLORS[i], lw=1.8,
                              label=f'J_{i} {ROLES[i]}')[0] for i in range(3)]
    line_joint = ax_joint.plot([], [], color=JOINT_COL, lw=2.0)[0]
    lines_err  = [ax_err.plot([], [], color=COLORS[i], lw=1.5,
                               label=ROLES[i], ls='--')[0] for i in range(3)]
    line_capture = ax_err.plot([], [], color=ORANGE, lw=1.5,
                                label='Leader→Evader dist', ls=':')[0]
    # Surround-mode lines (shown only in surround mode)
    line_surround_R   = ax_err.plot([], [], color=TEAL, lw=2.0,
                                     label='Surround R (m)', ls='-')[0]
    line_surround_d   = ax_err.plot([], [], color='#f43f5e', lw=1.5,
                                     label='Min drone↔evader', ls='-.')[0]
    ax_err.axhline(0.9, color='#f43f5e', alpha=0.3, lw=0.8, ls=':')  # capture radius

    # Mode label in cost plot
    mode_txt = ax_ji.text(0.99, 0.97, 'FORMATION / triangle',
                          transform=ax_ji.transAxes, color=TEXT_COL,
                          fontsize=8, va='top', ha='right', fontfamily='monospace')

    ax_ji.legend(facecolor='#0f1420', edgecolor=GRID_COL,
                 labelcolor=TEXT_COL, fontsize=8, loc='upper center',
                 bbox_to_anchor=(0.5, -0.05), ncol=3)
    ax_err.legend(facecolor='#0f1420', edgecolor=GRID_COL,
                  labelcolor=TEXT_COL, fontsize=8, loc='upper right')

    # ── Buffers ──────────────────────────────────────────────────────────
    maxpts       = int(WINDOW_S * UPDATE_HZ) + 20
    t_buf        = collections.deque(maxlen=maxpts)
    j_bufs       = [collections.deque(maxlen=maxpts) for _ in range(3)]
    jj_buf       = collections.deque(maxlen=maxpts)
    err_buf        = [collections.deque(maxlen=maxpts) for _ in range(3)]
    cap_buf        = collections.deque(maxlen=maxpts)
    surr_R_buf     = collections.deque(maxlen=maxpts)
    surr_met_buf   = collections.deque(maxlen=maxpts)   # strategy metric
    t0             = [None]
    last_threshold = [float('nan')]   # updated per strategy for threshold line

    def update(_frame):
        snap = node.snapshot()
        if snap is None:
            return lines_ji + [line_joint] + lines_err + [line_capture,
                    line_surround_R, line_surround_d]

        (pos, ev_pos, ldr_target, offsets, mode, current_form,
         surr_R, surr_metric, surr_threshold, surr_captured, surr_strategy) = snap

        now = time.monotonic()
        if t0[0] is None:
            t0[0] = now
        t = now - t0[0]

        costs = _compute_costs(pos, ldr_target, offsets)
        t_buf.append(t)
        for i in range(3):
            j_bufs[i].append(costs[i])
            if i == 0:
                e = np.linalg.norm(pos[0] - ldr_target) if ldr_target is not None else 0.0
            else:
                k = i - 1
                e = np.linalg.norm(pos[i] - (pos[0] + offsets[k]))
            err_buf[i].append(e)

        cap_dist = float(np.linalg.norm(pos[0] - ev_pos)) if ev_pos is not None else 0.0
        cap_buf.append(cap_dist if mode == 'pursuit' else float('nan'))
        jj_buf.append(sum(costs))

        # Surround ring metrics (NaN when not in surround mode)
        surr_R_buf.append(surr_R    if mode == 'surround' else float('nan'))
        surr_met_buf.append(surr_metric if mode == 'surround' else float('nan'))

        # Update capture threshold horizontal line when strategy changes
        if mode == 'surround' and not np.isnan(surr_threshold):
            if surr_threshold != last_threshold[0]:
                last_threshold[0] = surr_threshold
                line_surround_d.set_linestyle('-.')
                # threshold is drawn as a constant horizontal via ax_err.axhline
                # we update it by setting ydata on the threshold line
        # Dynamic threshold label update on the capture-threshold axhline
        ax_err.collections.clear()          # remove old fill if any

        t_arr = np.array(t_buf)
        t_lo  = max(0.0, t - WINDOW_S)
        t_hi  = t_lo + WINDOW_S

        for i in range(3):
            lines_ji[i].set_data(t_arr, np.array(j_bufs[i]))
            lines_err[i].set_data(t_arr, np.array(err_buf[i]))
        line_joint.set_data(t_arr, np.array(jj_buf))
        line_capture.set_data(t_arr, np.array(cap_buf))
        line_surround_R.set_data(t_arr, np.array(surr_R_buf))
        line_surround_d.set_data(t_arr, np.array(surr_met_buf))

        for ax in (ax_ji, ax_joint, ax_err):
            ax.set_xlim(t_lo, t_hi)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        # Draw dynamic capture threshold line
        if mode == 'surround' and not np.isnan(surr_threshold):
            ax_err.axhline(surr_threshold, color='#f43f5e',
                           alpha=0.5, lw=1.2, ls='--')

        if mode == 'pursuit':
            mode_str = 'PURSUIT-EVASION'
            mode_col = '#ef4444'
        elif mode == 'surround':
            cap_tag  = '  ★ CAPTURED' if surr_captured else ''
            mode_str = f'SURROUND / {surr_strategy or "?"}{cap_tag}'
            mode_col = TEAL
        else:
            mode_str = f'FORMATION / {current_form}'
            mode_col = TEXT_COL
        mode_txt.set_text(mode_str)
        mode_txt.set_color(mode_col)

        return lines_ji + [line_joint] + lines_err + [line_capture,
                line_surround_R, line_surround_d]

    ani = animation.FuncAnimation(       # noqa: F841
        fig, update,
        interval=1000 // UPDATE_HZ,
        blit=False,
        cache_frame_data=False,
    )

    plt.show()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
