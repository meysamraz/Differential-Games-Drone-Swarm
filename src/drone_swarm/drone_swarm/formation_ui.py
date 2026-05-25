"""
Formation, Pursuit, and Capture Strategy command UI.

Publishes std_msgs/String to /formation_cmd.
Subscribes to std_msgs/Float64MultiArray from /surround_status for live
ring-radius and capture feedback.

Run in a separate terminal:
  ros2 run drone_swarm formation_ui
"""

import threading
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.widgets as W
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray


BG     = '#0d1117'
PANEL  = '#0f1420'
ACCENT = '#1e2840'
TEXT   = '#c9d1e0'
TICK   = '#8b9dc3'
RED    = '#ef4444'
GREEN  = '#22c55e'
BLUE   = '#3b82f6'
YELLOW = '#fbbf24'
PURPLE = '#a78bfa'
ORANGE = '#f97316'
TEAL   = '#2dd4bf'

# Mini formation preview shapes
_SHAPES = {
    'triangle': np.array([[ 0.0, 1.0], [-0.85,-0.5], [ 0.85,-0.5], [ 0.0, 1.0]]),
    'line':     np.array([[-1.0, 0.0], [ 0.0,  0.0], [ 1.0,  0.0]]),
    'v_shape':  np.array([[ 0.0, 0.8], [-0.9, -0.6], [ 0.0,  0.8], [ 0.9,-0.6]]),
    'diamond':  np.array([[ 0.0, 1.0], [-0.9, -0.5], [ 0.0,  1.0], [ 0.9,-0.5]]),
}
_FORM_COLORS = {
    'triangle': BLUE, 'line': GREEN, 'v_shape': YELLOW, 'diamond': PURPLE,
}
_FORM_LABELS = {
    'triangle': '▲  TRIANGLE',
    'line':     '━  LINE',
    'v_shape':  'V  V-SHAPE',
    'diamond':  '◆  DIAMOND',
}

SURROUND_STRATEGIES = [
    ('triangle_surround', '△  TRIANGLE\nSURROUND',  TEAL),
    ('line_blockade',     '━  LINE\nBLOCKADE',      YELLOW),
    ('v_intercept',       'V  V-SHAPE\nINTERCEPT',  ORANGE),
    ('shrinking',         '◎  SHRINKING\nRING',      RED),
]


STRATEGIES_ORDERED = ['triangle_surround', 'line_blockade', 'v_intercept', 'shrinking']
STRATEGY_SHORT     = ['Triangle', 'Line', 'V-Shape', 'Shrinking']
STRATEGY_COLORS    = [TEAL, YELLOW, ORANGE, RED]


class UINode(Node):
    def __init__(self):
        super().__init__('formation_ui')
        self.pub = self.create_publisher(String, '/formation_cmd', 10)
        # Live surround status
        self.surround_R        = None
        self.surround_min_dist = None
        self.captured          = False
        self.create_subscription(
            Float64MultiArray, '/surround_status', self._status_cb, 10
        )
        # Auto strategy scores: [selected_idx, score0..3]
        self.auto_scores      = [1.0, 1.0, 1.0, 1.0]
        self.auto_selected    = -1   # -1 = auto not active
        self.create_subscription(
            Float64MultiArray, '/auto_strategy_scores', self._auto_cb, 10
        )

    def send(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.pub.publish(msg)
        self.get_logger().info(f'→ /formation_cmd: {cmd}')

    def _status_cb(self, msg):
        if len(msg.data) >= 3:
            self.surround_R        = msg.data[0]
            self.surround_min_dist = msg.data[1]
            self.captured          = (msg.data[2] > 0.5)

    def _auto_cb(self, msg):
        if len(msg.data) >= 5:
            self.auto_selected = int(msg.data[0])
            self.auto_scores   = list(msg.data[1:5])


def main(args=None):
    rclpy.init(args=args)
    node = UINode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Formation  ·  Pursuit-Evasion  ·  Capture Strategy',
                 color=TEXT, fontsize=12, fontweight='bold', y=0.988)

    formation_order = ['triangle', 'line', 'v_shape', 'diamond']

    # ── Formation shape previews (top strip) ──────────────────────────────
    gs = GridSpec(1, 4, figure=fig,
                  left=0.05, right=0.95, top=0.935, bottom=0.840,
                  wspace=0.30)
    preview_axes = {}
    for col, name in enumerate(formation_order):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(PANEL)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.0, 1.3)
        ax.axis('off')
        ax.set_title(_FORM_LABELS[name], color=_FORM_COLORS[name],
                     fontsize=8.5, fontweight='bold', pad=3)
        pts = _SHAPES[name]
        ax.plot(pts[:, 0], pts[:, 1], 'o-',
                color=_FORM_COLORS[name], lw=2, ms=6)
        preview_axes[name] = ax

    # ── Formation buttons ─────────────────────────────────────────────────
    btn_y, btn_h = 0.775, 0.055
    form_btns = {}
    for col, name in enumerate(formation_order):
        ax_b = fig.add_axes([0.05 + col * 0.232, btn_y, 0.20, btn_h])
        ax_b.set_facecolor(PANEL)
        btn = W.Button(ax_b, _FORM_LABELS[name], color=PANEL, hovercolor=ACCENT)
        btn.label.set_color(_FORM_COLORS[name])
        btn.label.set_fontsize(9)
        btn.label.set_fontweight('bold')
        form_btns[name] = btn

    # ── Divider: pursuit ──────────────────────────────────────────────────
    fig.text(0.5, 0.758, '─' * 58 + '  PURSUIT-EVASION  ' + '─' * 10,
             color=ACCENT, ha='center', fontsize=8)

    # ── Pursuit / Return buttons ───────────────────────────────────────────
    ax_pursuit = fig.add_axes([0.05, 0.685, 0.40, 0.063])
    ax_return  = fig.add_axes([0.55, 0.685, 0.40, 0.063])
    for ax in (ax_pursuit, ax_return):
        ax.set_facecolor(PANEL)
    btn_pursuit = W.Button(ax_pursuit,
                           '🔴  START PURSUIT  (drone1 vs evader)',
                           color=PANEL, hovercolor='#3b0000')
    btn_return  = W.Button(ax_return,
                           '🔵  RETURN TO FORMATION',
                           color=PANEL, hovercolor='#001a3b')
    btn_pursuit.label.set_color(RED);   btn_pursuit.label.set_fontsize(9)
    btn_return.label.set_color(BLUE);   btn_return.label.set_fontsize(9)
    for b in (btn_pursuit, btn_return):
        b.label.set_fontweight('bold')

    # ── Divider: capture strategy ─────────────────────────────────────────
    fig.text(0.5, 0.668, '─' * 52 + '  CAPTURE STRATEGY  ' + '─' * 8,
             color=ACCENT, ha='center', fontsize=8)

    # ── Capture strategy buttons (4 wide) ─────────────────────────────────
    cap_btn_axes = []
    cap_btns     = []
    for col, (cmd, label, color) in enumerate(SURROUND_STRATEGIES):
        ax_b = fig.add_axes([0.04 + col * 0.240, 0.575, 0.21, 0.082])
        ax_b.set_facecolor(PANEL)
        btn = W.Button(ax_b, label, color=PANEL, hovercolor=ACCENT)
        btn.label.set_color(color)
        btn.label.set_fontsize(8.5)
        btn.label.set_fontweight('bold')
        cap_btn_axes.append(ax_b)
        cap_btns.append((cmd, btn, color))

    # ── Surround radius progress bar ──────────────────────────────────────
    ax_bar = fig.add_axes([0.05, 0.530, 0.90, 0.030])
    ax_bar.set_facecolor(PANEL)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(0, 1)
    ax_bar.axis('off')
    for sp in ax_bar.spines.values():
        sp.set_edgecolor(ACCENT)
    bar_bg  = ax_bar.barh(0.5, 1.0, color=ACCENT, height=0.8, left=0)[0]   # noqa
    bar_fill = ax_bar.barh(0.5, 0.0, color=TEAL,   height=0.8, left=0)[0]
    bar_txt  = ax_bar.text(0.5, 0.5, 'Surround ring  — idle',
                           color=TEXT, ha='center', va='center',
                           fontsize=8, fontfamily='monospace',
                           transform=ax_bar.transAxes)

    # ── Status display ────────────────────────────────────────────────────
    ax_status = fig.add_axes([0.05, 0.420, 0.90, 0.130])
    ax_status.set_facecolor(PANEL)
    ax_status.axis('off')
    for sp in ax_status.spines.values():
        sp.set_edgecolor(ACCENT)

    status_main = ax_status.text(
        0.5, 0.72, 'TRIANGLE FORMATION',
        color=BLUE, ha='center', va='center',
        fontsize=15, fontweight='bold', fontfamily='monospace',
        transform=ax_status.transAxes,
    )
    status_sub = ax_status.text(
        0.5, 0.35,
        'drone1=Leader  drone2=Follower  drone3=Scout  |  evader=IDLE',
        color=TICK, ha='center', va='center',
        fontsize=8, fontfamily='monospace',
        transform=ax_status.transAxes,
    )
    status_game = ax_status.text(
        0.5, 0.10,
        'Nash P-control  |  Smith predictor active  |  Wind + Itô noise',
        color=ACCENT, ha='center', va='center',
        fontsize=7.5, fontfamily='monospace',
        transform=ax_status.transAxes,
    )

    # ── Divider: auto strategy ────────────────────────────────────────────
    fig.text(0.5, 0.406, '─' * 42 + '  AUTO STRATEGY (MINIMAX)  ' + '─' * 6,
             color=ACCENT, ha='center', fontsize=8)

    # ── AUTO button ───────────────────────────────────────────────────────
    ax_auto = fig.add_axes([0.05, 0.338, 0.90, 0.060])
    ax_auto.set_facecolor(PANEL)
    btn_auto = W.Button(
        ax_auto,
        '⚡  AUTO STRATEGY  —  Minimax Game-Theoretic Selection  (re-evaluates every 3 s)',
        color=PANEL, hovercolor='#1a2a1a',
    )
    btn_auto.label.set_color(GREEN)
    btn_auto.label.set_fontsize(8.5)
    btn_auto.label.set_fontweight('bold')

    # ── Minimax score panel ───────────────────────────────────────────────
    # Bars show coverage fraction (1 − escape fraction): longer = better containment.
    ax_scores = fig.add_axes([0.05, 0.060, 0.90, 0.268])
    ax_scores.set_facecolor(PANEL)
    ax_scores.set_xlim(0, 1)
    ax_scores.set_ylim(-0.5, 3.5)
    ax_scores.axis('off')
    for sp in ax_scores.spines.values():
        sp.set_edgecolor(ACCENT)

    _score_bg_bars = []
    _score_bars    = []
    _score_labels  = []
    _score_pcts    = []
    for row, (label, color) in enumerate(zip(STRATEGY_SHORT, STRATEGY_COLORS)):
        y = 3 - row
        ax_scores.barh(y, 1.0, height=0.55, color=ACCENT, left=0, zorder=1)
        bg = ax_scores.barh(y, 1.0, height=0.55, color=ACCENT, left=0, zorder=1)[0]  # noqa
        bar = ax_scores.barh(y, 0.5, height=0.55, color=color,
                             alpha=0.5, left=0, zorder=2)[0]
        lbl = ax_scores.text(-0.02, y, label, color=color, ha='right', va='center',
                              fontsize=8.5, fontweight='bold', fontfamily='monospace')
        pct = ax_scores.text(0.52, y, '—', color=TEXT, ha='left', va='center',
                              fontsize=8, fontfamily='monospace')
        _score_bars.append(bar)
        _score_labels.append(lbl)
        _score_pcts.append(pct)

    ax_scores.text(0.5, -0.35, 'Coverage  (fraction of escape directions blocked)',
                   color=TICK, ha='center', va='center',
                   fontsize=7.5, fontfamily='monospace', transform=ax_scores.transData)

    auto_active = [False]

    current = {'mode': 'triangle', 'surround_cmd': None}
    prev_captured = [False]

    # ── Highlight helper ──────────────────────────────────────────────────
    def _highlight(active_name):
        for name, ax in preview_axes.items():
            col = _FORM_COLORS[name] if name == active_name else ACCENT
            ax.set_title(_FORM_LABELS.get(name, name), color=col,
                         fontsize=8.5, fontweight='bold', pad=3)
        fig.canvas.draw_idle()

    # ── Formation callbacks ────────────────────────────────────────────────
    def _reset_auto():
        auto_active[0] = False
        node.auto_selected = -1
        for bar in _score_bars:
            bar.set_width(0)
        for lbl in _score_labels:
            lbl.set_color(TICK)
        for pct in _score_pcts:
            pct.set_text('—')

    def make_form_cb(name):
        def cb(_):
            node.send(name)
            _reset_auto()
            current['mode'] = name
            current['surround_cmd'] = None
            status_main.set_text(f'{_FORM_LABELS[name].replace("  ", " ")} FORMATION')
            status_main.set_color(_FORM_COLORS[name])
            status_sub.set_text(
                'drone1=Leader  drone2=Follower  drone3=Scout  |  evader=IDLE'
            )
            status_sub.set_color(TICK)
            status_game.set_text(
                'Nash P-control  |  Smith predictor active  |  Wind + Itô noise'
            )
            bar_fill.set_width(0)
            bar_txt.set_text('Surround ring  — idle')
            _highlight(name)
        return cb

    for name, btn in form_btns.items():
        btn.on_clicked(make_form_cb(name))

    # ── AUTO callback ──────────────────────────────────────────────────────
    def on_auto(_):
        node.send('auto_surround')
        auto_active[0] = True
        current['mode'] = 'auto_surround'
        current['surround_cmd'] = 'auto_surround'
        prev_captured[0] = False
        node.captured = False
        status_main.set_text('⚡  AUTO STRATEGY  (Minimax)')
        status_main.set_color(GREEN)
        status_sub.set_text(
            'System evaluates escape fraction per strategy  ·  switches every 3 s'
        )
        status_sub.set_color(GREEN)
        status_game.set_text(
            'min_s  max_θ  P(escape | s, θ)  ·  Isaacs optimal evasion model'
        )
        bar_fill.set_width(0)
        bar_txt.set_text('Surround ring  — auto initialising…')
        fig.canvas.draw_idle()

    btn_auto.on_clicked(on_auto)

    # ── Pursuit callbacks ──────────────────────────────────────────────────
    def on_pursuit(_):
        node.send('pursuit')
        current['mode'] = 'pursuit'
        current['surround_cmd'] = None
        status_main.set_text('PURSUIT-EVASION  (zero-sum)')
        status_main.set_color(RED)
        status_sub.set_text(
            'drone1=PURSUER  drone2+3=trail formation  |  evader=ACTIVE (jinking)'
        )
        status_sub.set_color(ORANGE)
        status_game.set_text(
            'Leader: Isaacs minimax  |  Followers: Nash relative to drone1'
        )
        bar_fill.set_width(0)
        bar_txt.set_text('Surround ring  — idle')
        _highlight('pursuit' if 'pursuit' in preview_axes else list(preview_axes)[0])
        fig.canvas.draw_idle()

    def on_return(_):
        name = current.get('mode', 'triangle')
        if name in ('pursuit',) or name in {s[0] for s in SURROUND_STRATEGIES}:
            name = 'triangle'
        node.send('return')
        node.send(name)
        current['mode'] = name
        current['surround_cmd'] = None
        status_main.set_text(
            f'RETURNING → {name.upper().replace("_", "-")}'
        )
        status_main.set_color(BLUE)
        status_sub.set_text(
            'drone1=Leader  drone2=Follower  drone3=Scout  |  evader=IDLE'
        )
        status_sub.set_color(TICK)
        status_game.set_text(
            'Nash P-control  |  Smith predictor active  |  Wind + Itô noise'
        )
        bar_fill.set_width(0)
        bar_txt.set_text('Surround ring  — idle')
        _highlight(name)
        fig.canvas.draw_idle()

    btn_pursuit.on_clicked(on_pursuit)
    btn_return.on_clicked(on_return)

    # ── Capture strategy callbacks ─────────────────────────────────────────
    SURROUND_LABELS = {
        'triangle_surround': 'TRIANGLE SURROUND',
        'line_blockade':     'LINE BLOCKADE',
        'v_intercept':       'V-SHAPE INTERCEPT',
        'shrinking':         'SHRINKING RING',
    }
    SURROUND_SUBS = {
        'triangle_surround':
            'Equilateral ring · shrinks inward · gap-finding evasion',
        'line_blockade':
            'Perpendicular line ahead of evader · spread shrinks',
        'v_intercept':
            'V-funnel faces evader heading · wings close in',
        'shrinking':
            'Orbiting noose · rotates + tightens simultaneously',
    }

    def make_surround_cb(cmd, color):
        def cb(_):
            node.send(cmd)
            _reset_auto()
            current['mode'] = cmd
            current['surround_cmd'] = cmd
            prev_captured[0] = False
            node.captured = False
            status_main.set_text(f'◎  {SURROUND_LABELS[cmd]}')
            status_main.set_color(color)
            status_sub.set_text(SURROUND_SUBS[cmd])
            status_sub.set_color(color)
            status_game.set_text(
                'All 3 drones surround  |  Nash slot assignment  |  '
                'Shrinking ring  |  Gap-finding evasion'
            )
            bar_txt.set_text(
                f'{SURROUND_LABELS[cmd]}  ·  ring closing…'
            )
            fig.canvas.draw_idle()
        return cb

    for cmd, btn, color in cap_btns:
        btn.on_clicked(make_surround_cb(cmd, color))

    # ── Live update: ring progress + capture flash + auto scores ──────────
    def _animate(_frame):
        redraw = False

        # Auto strategy score bars
        if auto_active[0] and node.auto_selected >= 0:
            sel = node.auto_selected
            scores = node.auto_scores
            for row, bar in enumerate(_score_bars):
                cov = max(0.0, min(1.0, 1.0 - scores[row]))
                bar.set_width(cov)
                bar.set_alpha(1.0 if row == sel else 0.45)
                pct_txt = f'{cov*100:.0f}% covered'
                if row == sel:
                    pct_txt = f'▶ {STRATEGY_SHORT[row]}: {pct_txt}  ← ACTIVE'
                    _score_labels[row].set_color(STRATEGY_COLORS[row])
                else:
                    pct_txt = f'   {STRATEGY_SHORT[row]}: {pct_txt}'
                    _score_labels[row].set_color(TICK)
                _score_pcts[row].set_text(pct_txt)
            redraw = True

        if current.get('surround_cmd') is None:
            if redraw:
                fig.canvas.draw_idle()
            return

        if node.captured and not prev_captured[0]:
            prev_captured[0] = True
            status_main.set_text('★  CAPTURED!')
            status_main.set_color(ORANGE)
            cmd_key = current['surround_cmd']
            strat_name = (STRATEGY_SHORT[node.auto_selected]
                          if cmd_key == 'auto_surround' and node.auto_selected >= 0
                          else SURROUND_LABELS.get(cmd_key, '?'))
            status_sub.set_text(f'Evader neutralised  —  strategy: {strat_name}')
            status_sub.set_color(ORANGE)
            bar_fill.set_width(1.0)
            bar_txt.set_text('★  CAPTURE COMPLETE')
            redraw = True

        elif not node.captured and node.surround_R is not None:
            R = node.surround_R
            d = node.surround_min_dist or 0.0
            from drone_swarm.formation_controller import (
                SURROUND_RADIUS_INIT, SURROUND_RADIUS_MIN
            )
            frac = 1.0 - max(0.0, (R - SURROUND_RADIUS_MIN) /
                             (SURROUND_RADIUS_INIT - SURROUND_RADIUS_MIN))
            bar_fill.set_width(min(frac, 1.0))
            bar_txt.set_text(f'Ring  R={R:.2f} m   nearest drone={d:.2f} m')
            redraw = True

        if redraw:
            fig.canvas.draw_idle()

    ani = animation.FuncAnimation(    # noqa: F841
        fig, _animate, interval=200, blit=False, cache_frame_data=False,
    )

    _highlight('triangle')
    plt.show()
    rclpy.shutdown()
