"""
Live cost / game-metric plotter — mode-adaptive.

FORMATION mode
  Panel 1 : Per-agent Nash costs  Jᵢ(t)
  Panel 2 : Joint cost  J(t) = ΣJᵢ
  Panel 3 : Position errors  ‖eᵢ‖  (leader-to-anchor + followers-to-slot)

PURSUIT mode
  Panel 1 : Distance of each drone to evader  dᵢ(t)
  Panel 2 : Closing speed of drone1  −ḋ₁(t)  (positive = closing in)
  Panel 3 : Evader speed estimate  ‖v_ev‖(t)

SURROUND / CAPTURE mode
  Panel 1 : Escape fraction per strategy  f_s ∈ [0,1]  (auto mode)
             or surround ring radius breakdown (manual mode)
  Panel 2 : Surround ring radius  R(t)
  Panel 3 : Min drone↔evader distance  +  capture-threshold line
"""

import math
import time
import threading
import collections

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float64MultiArray


# ── Must match formation_controller.py ────────────────────────────────────────
FORMATION_OFFSETS = {
    'triangle': np.array([[-1.0, -1.732], [ 1.0, -1.732]]),
    'line':     np.array([[-2.0,  0.0  ], [ 2.0,  0.0  ]]),
    'v_shape':  np.array([[-2.0, -2.0  ], [ 2.0, -2.0  ]]),
    'diamond':  np.array([[-2.0, -3.0  ], [ 2.0, -3.0  ]]),
}

STRATEGIES_ORDERED = ['triangle_surround', 'line_blockade', 'v_intercept', 'shrinking']
STRATEGY_SHORT     = ['Triangle', 'Line', 'V-Shape', 'Shrinking']

DRONE_NS = ['drone1', 'drone2', 'drone3']
ROLES    = ['Leader', 'Follower', 'Scout']
Q_POS    = [3.0, 2.0, 1.0]
GAMMA    = np.array([[0.0, 0.3, 0.3],
                     [0.8, 0.0, 0.5],
                     [0.6, 0.5, 0.0]])

WINDOW_S  = 60.0
UPDATE_HZ = 10

# Colours
COLORS     = ['#ef4444', '#22c55e', '#3b82f6']   # drone1/2/3
STRAT_COLS = ['#2dd4bf', '#fbbf24', '#f97316', '#ef4444']  # 4 strategies
JOINT_COL  = '#a78bfa'
BG         = '#0d1117'
GRID_COL   = '#1e2840'
TEXT_COL   = '#c9d1e0'
TICK_COL   = '#8b9dc3'
ORANGE     = '#f97316'
TEAL       = '#2dd4bf'
GREEN      = '#22c55e'


class CostNode(Node):

    def __init__(self):
        super().__init__('cost_plotter')

        self.positions     = np.full((3, 2), np.nan)
        self.evader_pos    = None
        self.leader_target = None
        self.offsets       = FORMATION_OFFSETS['triangle'].copy()
        self.mode          = 'formation'
        self.current_form  = 'triangle'

        self.surround_R         = float('nan')
        self.surround_metric    = float('nan')
        self.surround_threshold = float('nan')
        self.surround_captured  = False
        self.surround_strategy  = None
        self.auto_mode          = False
        self.escape_fracs       = [float('nan')] * 4   # per strategy

        self._lock = threading.Lock()

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
        self.create_subscription(
            Float64MultiArray, '/auto_strategy_scores', self._auto_cb, 10
        )

    def _odom_cb(self, msg, i):
        with self._lock:
            self.positions[i, 0] = msg.pose.pose.position.x
            self.positions[i, 1] = msg.pose.pose.position.y
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
                self.surround_R        = msg.data[0]
                self.surround_metric   = msg.data[1]
                self.surround_captured = (msg.data[2] > 0.5)
                if len(msg.data) >= 5:
                    self.surround_threshold = msg.data[4]

    def _auto_cb(self, msg):
        if len(msg.data) >= 5:
            with self._lock:
                self.escape_fracs = list(msg.data[1:5])

    def _cmd_cb(self, msg):
        cmd = msg.data.strip().lower()
        with self._lock:
            if cmd in FORMATION_OFFSETS:
                self.offsets      = FORMATION_OFFSETS[cmd].copy()
                self.current_form = cmd
                self.mode         = 'formation'
                self.auto_mode    = False
                if not np.any(np.isnan(self.positions[0])):
                    self.leader_target = self.positions[0].copy()
            elif cmd == 'pursuit':
                self.mode      = 'pursuit'
                self.auto_mode = False
            elif cmd == 'return':
                self.mode      = 'formation'
                self.auto_mode = False
                if not np.any(np.isnan(self.positions[0])):
                    self.leader_target = self.positions[0].copy()
            elif cmd == 'auto_surround':
                self.mode               = 'surround'
                self.auto_mode          = True
                self.surround_R         = float('nan')
                self.surround_metric    = float('nan')
                self.surround_threshold = float('nan')
                self.surround_captured  = False
                self.escape_fracs       = [float('nan')] * 4
            elif cmd in {'triangle_surround', 'line_blockade', 'v_intercept', 'shrinking'}:
                self.mode               = 'surround'
                self.auto_mode          = False
                self.surround_strategy  = cmd
                self.surround_R         = float('nan')
                self.surround_metric    = float('nan')
                self.surround_threshold = float('nan')
                self.surround_captured  = False

    def snapshot(self):
        with self._lock:
            if np.any(np.isnan(self.positions)):
                return None
            return dict(
                pos            = self.positions.copy(),
                ev_pos         = self.evader_pos.copy() if self.evader_pos is not None else None,
                ldr_target     = self.leader_target.copy() if self.leader_target is not None else None,
                offsets        = self.offsets.copy(),
                mode           = self.mode,
                current_form   = self.current_form,
                surround_R     = self.surround_R,
                surround_metric    = self.surround_metric,
                surround_threshold = self.surround_threshold,
                surround_captured  = self.surround_captured,
                surround_strategy  = self.surround_strategy,
                auto_mode      = self.auto_mode,
                escape_fracs   = list(self.escape_fracs),
            )


def _compute_costs(pos, leader_target, offsets):
    costs = []
    e0 = pos[0] - leader_target if leader_target is not None else np.zeros(2)
    costs.append(Q_POS[0] * float(e0 @ e0))
    for k in range(2):
        i   = k + 1
        j   = 2 - k
        j_k = 1 - k
        e_i = pos[i] - (pos[0] + offsets[k])
        rel = (pos[i] - pos[j]) - (offsets[k] - offsets[j_k])
        costs.append(Q_POS[i] * float(e_i @ e_i) + GAMMA[i, j] * float(rel @ rel))
    return costs


def _style_ax(ax, ylabel='', title=''):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TICK_COL, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)
    ax.grid(True, alpha=0.15, color=TICK_COL)
    ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=9)
    ax.set_title(title,   color=TEXT_COL, fontsize=9)
    ax.axhline(0, color='#ffffff', alpha=0.15, lw=0.8, ls='--')


def main(args=None):
    rclpy.init(args=args)
    node = CostNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 9))
    fig.patch.set_facecolor(BG)
    title_txt = fig.suptitle(
        'FORMATION — Nash Live Cost Monitor',
        color=TEXT_COL, fontsize=11, fontweight='bold',
    )

    gs = gridspec.GridSpec(3, 1, hspace=0.50,
                           left=0.09, right=0.97, top=0.92, bottom=0.07)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    for ax in (ax1, ax2, ax3):
        _style_ax(ax)
    ax3.set_xlabel('Time (s)', color=TEXT_COL, fontsize=9)

    # ── All lines (shown/hidden per mode) ────────────────────────────────────

    # Formation lines
    l_ji     = [ax1.plot([], [], color=COLORS[i], lw=1.8,
                          label=f'J_{i} {ROLES[i]}')[0] for i in range(3)]
    l_joint  =  ax2.plot([], [], color=JOINT_COL, lw=2.0, label='J = ΣJᵢ')[0]
    l_err    = [ax3.plot([], [], color=COLORS[i], lw=1.5, ls='--',
                          label=ROLES[i])[0] for i in range(3)]

    # Pursuit lines
    l_dist   = [ax1.plot([], [], color=COLORS[i], lw=1.8, ls='-',
                          label=f'{ROLES[i]} → evader')[0] for i in range(3)]
    l_close  =  ax2.plot([], [], color=ORANGE, lw=2.0, label='Closing speed (m/s)')[0]
    l_evspd  =  ax3.plot([], [], color=GREEN,  lw=1.8, label='Evader speed (m/s)')[0]

    # Surround lines
    l_frac   = [ax1.plot([], [], color=STRAT_COLS[s], lw=1.8, ls='-',
                          label=f'{STRATEGY_SHORT[s]}')[0] for s in range(4)]
    l_ring   =  ax2.plot([], [], color=TEAL,   lw=2.0, label='Ring radius R (m)')[0]
    l_mindst =  ax3.plot([], [], color='#f43f5e', lw=1.8, label='Min drone↔evader (m)')[0]
    thr_line =  ax3.axhline(float('nan'), color='#f43f5e', alpha=0.45, lw=1.2, ls='--')

    # Gain annotation (formation only)
    gain_anns = []
    R_vals = [0.8, 0.4, 0.15]
    for i in range(3):
        if i == 0:
            txt = (f'Leader  Q={Q_POS[0]}  R={R_vals[0]}'
                   f'  K={math.sqrt(Q_POS[0]/R_vals[0]):.2f}  (no coupling)')
        else:
            j     = 3 - i
            gamma = GAMMA[i, j]
            K     = math.sqrt((Q_POS[i] + gamma) / R_vals[i])
            txt   = (f'{ROLES[i]}  Q={Q_POS[i]}  γ={gamma}  R={R_vals[i]}'
                     f'  K={K:.2f}')
        ann = ax1.text(0.01, 0.97 - i * 0.12, txt,
                       transform=ax1.transAxes, color=COLORS[i],
                       fontsize=7.5, va='top', fontfamily='monospace')
        gain_anns.append(ann)

    mode_badge = ax1.text(
        0.99, 0.97, 'FORMATION / triangle',
        transform=ax1.transAxes, color=TEXT_COL,
        fontsize=8, va='top', ha='right', fontfamily='monospace',
    )

    # Legends — drawn once, toggled with set_visible on the legend object
    leg1_form = ax1.legend(handles=l_ji,   facecolor='#0f1420', edgecolor=GRID_COL,
                            labelcolor=TEXT_COL, fontsize=8, loc='upper center',
                            bbox_to_anchor=(0.5, -0.05), ncol=3)
    leg1_purs = ax1.legend(handles=l_dist, facecolor='#0f1420', edgecolor=GRID_COL,
                            labelcolor=TEXT_COL, fontsize=8, loc='upper center',
                            bbox_to_anchor=(0.5, -0.05), ncol=3)
    leg1_surr = ax1.legend(handles=l_frac, facecolor='#0f1420', edgecolor=GRID_COL,
                            labelcolor=TEXT_COL, fontsize=8, loc='upper center',
                            bbox_to_anchor=(0.5, -0.05), ncol=4)
    ax1.add_artist(leg1_form)
    ax1.add_artist(leg1_purs)
    ax1.add_artist(leg1_surr)

    ax2.legend(handles=[l_joint, l_close, l_ring],
               facecolor='#0f1420', edgecolor=GRID_COL,
               labelcolor=TEXT_COL, fontsize=8, loc='upper right')
    ax3.legend(handles=[*l_err, l_evspd, l_mindst],
               facecolor='#0f1420', edgecolor=GRID_COL,
               labelcolor=TEXT_COL, fontsize=8, loc='upper right')

    # ── Buffers ───────────────────────────────────────────────────────────────
    maxpts = int(WINDOW_S * UPDATE_HZ) + 20

    t_buf       = collections.deque(maxlen=maxpts)

    # formation
    j_bufs      = [collections.deque(maxlen=maxpts) for _ in range(3)]
    jj_buf      = collections.deque(maxlen=maxpts)
    err_bufs    = [collections.deque(maxlen=maxpts) for _ in range(3)]

    # pursuit
    dist_bufs   = [collections.deque(maxlen=maxpts) for _ in range(3)]
    close_buf   = collections.deque(maxlen=maxpts)
    evspd_buf   = collections.deque(maxlen=maxpts)

    # surround
    frac_bufs   = [collections.deque(maxlen=maxpts) for _ in range(4)]
    ring_buf    = collections.deque(maxlen=maxpts)
    mindst_buf  = collections.deque(maxlen=maxpts)

    t0            = [None]
    prev_d1_ev    = [None]   # for closing speed
    prev_ev_pos   = [None]   # for evader speed
    prev_t        = [None]

    def _set_visible_mode(mode):
        is_form   = (mode == 'formation')
        is_purs   = (mode == 'pursuit')
        is_surr   = (mode == 'surround')

        for l in l_ji:    l.set_visible(is_form)
        l_joint.set_visible(is_form)
        for l in l_err:   l.set_visible(is_form)
        for ann in gain_anns: ann.set_visible(is_form)
        leg1_form.set_visible(is_form)

        for l in l_dist:  l.set_visible(is_purs)
        l_close.set_visible(is_purs)
        l_evspd.set_visible(is_purs)
        leg1_purs.set_visible(is_purs)

        for l in l_frac:  l.set_visible(is_surr)
        l_ring.set_visible(is_surr)
        l_mindst.set_visible(is_surr)
        leg1_surr.set_visible(is_surr)

    def _set_labels(mode, snap):
        if mode == 'formation':
            ax1.set_title('Per-Agent Nash Cost  Jᵢ(t)', color=TEXT_COL, fontsize=9)
            ax1.set_ylabel('Jᵢ(t)', color=TEXT_COL, fontsize=9)
            ax2.set_title('Joint Cost  J(t) = ΣJᵢ', color=TEXT_COL, fontsize=9)
            ax2.set_ylabel('J(t)', color=TEXT_COL, fontsize=9)
            ax3.set_title('Position Errors  ‖eᵢ‖', color=TEXT_COL, fontsize=9)
            ax3.set_ylabel('‖eᵢ‖ (m)', color=TEXT_COL, fontsize=9)
            title_txt.set_text(
                f'FORMATION — Nash Cost Monitor  [{snap["current_form"]}]'
            )
            title_txt.set_color(TEXT_COL)

        elif mode == 'pursuit':
            ax1.set_title('Drone → Evader Distance  dᵢ(t)', color=TEXT_COL, fontsize=9)
            ax1.set_ylabel('dist (m)', color=TEXT_COL, fontsize=9)
            ax2.set_title('Closing Speed  −ḋ₁(t)  (+ = closing in)', color=TEXT_COL, fontsize=9)
            ax2.set_ylabel('ṁ/s', color=TEXT_COL, fontsize=9)
            ax3.set_title('Evader Speed Estimate  ‖v_ev‖', color=TEXT_COL, fontsize=9)
            ax3.set_ylabel('m/s', color=TEXT_COL, fontsize=9)
            title_txt.set_text('PURSUIT-EVASION — Isaacs Differential Game')
            title_txt.set_color('#ef4444')

        elif mode == 'surround':
            auto = snap['auto_mode']
            p1_title = ('Escape Fraction per Strategy  fₛ(t)  [minimax]'
                        if auto else 'Surround Ring Radius  R(t)')
            ax1.set_title(p1_title, color=TEXT_COL, fontsize=9)
            ax1.set_ylabel('fₛ (0–1)' if auto else 'R (m)', color=TEXT_COL, fontsize=9)
            ax2.set_title('Surround Ring Radius  R(t)', color=TEXT_COL, fontsize=9)
            ax2.set_ylabel('R (m)', color=TEXT_COL, fontsize=9)
            captured = snap['surround_captured']
            ax3.set_title(
                ('★ CAPTURED' if captured else 'Min Drone↔Evader Distance'),
                color=('#f97316' if captured else TEXT_COL), fontsize=9,
            )
            ax3.set_ylabel('dist (m)', color=TEXT_COL, fontsize=9)
            strat = snap['surround_strategy'] or 'auto'
            cap_tag = '  ★ CAPTURED' if captured else ''
            title_txt.set_text(
                f'SURROUND — {"AUTO / " if auto else ""}{strat}{cap_tag}'
            )
            title_txt.set_color(TEAL)

    _prev_mode = [None]

    def update(_frame):
        snap = node.snapshot()
        if snap is None:
            return []

        pos        = snap['pos']
        ev_pos     = snap['ev_pos']
        ldr_target = snap['ldr_target']
        offsets    = snap['offsets']
        mode       = snap['mode']

        now = time.monotonic()
        if t0[0] is None:
            t0[0] = now
        t   = now - t0[0]
        dt  = (now - prev_t[0]) if prev_t[0] is not None else 0.05
        prev_t[0] = now

        t_buf.append(t)

        # ── Formation metrics ──────────────────────────────────────────────
        costs = _compute_costs(pos, ldr_target, offsets)
        for i in range(3):
            j_bufs[i].append(costs[i])
            if i == 0:
                e = np.linalg.norm(pos[0] - ldr_target) if ldr_target is not None else 0.0
            else:
                e = np.linalg.norm(pos[i] - (pos[0] + offsets[i - 1]))
            err_bufs[i].append(e)
        jj_buf.append(sum(costs))

        # ── Pursuit metrics ────────────────────────────────────────────────
        for i in range(3):
            d = float(np.linalg.norm(pos[i] - ev_pos)) if ev_pos is not None else float('nan')
            dist_bufs[i].append(d)

        d1_ev = float(np.linalg.norm(pos[0] - ev_pos)) if ev_pos is not None else None
        if d1_ev is not None and prev_d1_ev[0] is not None and dt > 0:
            closing = -(d1_ev - prev_d1_ev[0]) / dt   # positive = closing in
            close_buf.append(float(np.clip(closing, -5.0, 5.0)))
        else:
            close_buf.append(float('nan'))
        prev_d1_ev[0] = d1_ev

        if ev_pos is not None and prev_ev_pos[0] is not None and dt > 0:
            evspd_buf.append(float(np.linalg.norm(ev_pos - prev_ev_pos[0]) / dt))
        else:
            evspd_buf.append(float('nan'))
        prev_ev_pos[0] = ev_pos.copy() if ev_pos is not None else None

        # ── Surround metrics ───────────────────────────────────────────────
        fracs = snap['escape_fracs']   # list of 4 floats (NaN if manual)
        for s in range(4):
            frac_bufs[s].append(fracs[s])

        ring_buf.append(snap['surround_R'])
        mindst_buf.append(snap['surround_metric'])

        # Update capture threshold dashed line
        thr = snap['surround_threshold']
        if not math.isnan(thr):
            thr_line.set_ydata([thr, thr])

        # ── Toggle visibility on mode change ──────────────────────────────
        if mode != _prev_mode[0]:
            _set_visible_mode(mode)
            _prev_mode[0] = mode

        _set_labels(mode, snap)

        # Mode badge
        mode_labels = {
            'formation': f'FORMATION / {snap["current_form"]}',
            'pursuit':   'PURSUIT-EVASION',
            'surround':  f'SURROUND / {snap["surround_strategy"] or "auto"}',
        }
        mode_colors = {'formation': TEXT_COL, 'pursuit': '#ef4444', 'surround': TEAL}
        mode_badge.set_text(mode_labels.get(mode, mode))
        mode_badge.set_color(mode_colors.get(mode, TEXT_COL))

        t_arr = np.array(t_buf)
        t_lo  = max(0.0, t - WINDOW_S)
        t_hi  = t_lo + WINDOW_S

        # Set data for all lines (only visible ones matter for render cost)
        for i in range(3):
            l_ji[i].set_data(t_arr, np.array(j_bufs[i]))
            l_err[i].set_data(t_arr, np.array(err_bufs[i]))
            l_dist[i].set_data(t_arr, np.array(dist_bufs[i]))
        l_joint.set_data(t_arr, np.array(jj_buf))
        l_close.set_data(t_arr, np.array(close_buf))
        l_evspd.set_data(t_arr, np.array(evspd_buf))
        for s in range(4):
            l_frac[s].set_data(t_arr, np.array(frac_bufs[s]))
        l_ring.set_data(t_arr, np.array(ring_buf))
        l_mindst.set_data(t_arr, np.array(mindst_buf))

        for ax in (ax1, ax2, ax3):
            ax.set_xlim(t_lo, t_hi)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        return []

    _set_visible_mode('formation')

    ani = animation.FuncAnimation(      # noqa: F841
        fig, update,
        interval=1000 // UPDATE_HZ,
        blit=False,
        cache_frame_data=False,
    )

    plt.show()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
