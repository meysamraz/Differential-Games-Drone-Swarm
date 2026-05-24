"""
Wind disturbance controller — interactive matplotlib sliders.

Publishes Float64MultiArray to /wind_scale → wind_node reads it.
  data[0] = bias_scale    (0 = calm,  1 = default,  3 = storm)
  data[1] = gust_scale    same range

Run in its own terminal after launching the simulation:
  ros2 run drone_swarm wind_controller
"""

import threading
import matplotlib
matplotlib.use('TkAgg')          # explicit backend — avoids blank-window issues
import matplotlib.pyplot as plt
import matplotlib.widgets as W

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


# ── Visual style ──────────────────────────────────────────────────────────────
BG       = '#0d1117'
PANEL    = '#0f1420'
ACCENT   = '#1e2840'
TEXT     = '#c9d1e0'
TICK     = '#8b9dc3'
RED      = '#ef4444'
BLUE     = '#3b82f6'
GREEN    = '#22c55e'
PURPLE   = '#a78bfa'
YELLOW   = '#fbbf24'


class WindScaleNode(Node):
    def __init__(self):
        super().__init__('wind_controller')
        self.pub = self.create_publisher(Float64MultiArray, '/wind_scale', 10)
        self.bias_scale = 1.0
        self.gust_scale = 1.0
        # Publish at 10 Hz so wind_node always has fresh values
        self.create_timer(0.1, self._publish)

    def _publish(self):
        msg = Float64MultiArray()
        msg.data = [self.bias_scale, self.gust_scale]
        self.pub.publish(msg)

    def set_wind(self, bias: float, gust: float):
        self.bias_scale = max(0.0, bias)
        self.gust_scale = max(0.0, gust)


def main(args=None):
    rclpy.init(args=args)
    node = WindScaleNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7, 5.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Wind Disturbance Controller', color=TEXT, fontsize=13,
                 fontweight='bold', y=0.97)

    # Main display area (shows current wind state as a gauge)
    ax_main = fig.add_axes([0.08, 0.52, 0.84, 0.38])
    ax_main.set_facecolor(PANEL)
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.axis('off')
    for sp in ax_main.spines.values():
        sp.set_edgecolor(ACCENT)

    # Bias bar
    bias_bg  = ax_main.barh(0.72, 1.0, 0.18, left=0,
                             color=ACCENT, align='center')[0]
    bias_bar = ax_main.barh(0.72, 1/3, 0.18, left=0,
                             color=RED, align='center')[0]   # default 1.0 / 3.0 max
    ax_main.text(0.01, 0.72, 'Bias', color=TEXT, va='center', fontsize=10,
                 fontweight='bold', transform=ax_main.transData)

    # Gust bar
    gust_bg  = ax_main.barh(0.38, 1.0, 0.18, left=0,
                             color=ACCENT, align='center')[0]
    gust_bar = ax_main.barh(0.38, 1/3, 0.18, left=0,
                             color=BLUE, align='center')[0]
    ax_main.text(0.01, 0.38, 'Gust', color=TEXT, va='center', fontsize=10,
                 fontweight='bold', transform=ax_main.transData)

    # Scale labels
    bias_label = ax_main.text(0.98, 0.72, '1.00×', color=TEXT, va='center',
                               ha='right', fontsize=11, fontfamily='monospace')
    gust_label = ax_main.text(0.98, 0.38, '1.00×', color=TEXT, va='center',
                               ha='right', fontsize=11, fontfamily='monospace')

    status_txt = ax_main.text(0.5, 0.08, 'DEFAULT', color=GREEN,
                               ha='center', va='center', fontsize=11,
                               fontweight='bold', fontfamily='monospace')

    # Tick marks at 0×, 1×, 2×, 3×
    for x_frac, label in [(0, '0×'), (1/3, '1×'), (2/3, '2×'), (1.0, '3×')]:
        ax_main.axvline(x_frac, color=ACCENT, lw=1.0, alpha=0.6)
        ax_main.text(x_frac, 0.06, label, color=TICK, ha='center',
                     va='center', fontsize=7.5)

    # ── Sliders ───────────────────────────────────────────────────────────
    # [left, bottom, width, height]
    ax_sbias = fig.add_axes([0.15, 0.36, 0.70, 0.04])
    ax_sgust = fig.add_axes([0.15, 0.27, 0.70, 0.04])

    for ax_s in (ax_sbias, ax_sgust):
        ax_s.set_facecolor(PANEL)

    s_bias = W.Slider(ax_sbias, 'Bias ×', 0.0, 3.0,
                      valinit=1.0, color=RED,
                      initcolor=RED, track_color=ACCENT)
    s_gust = W.Slider(ax_sgust, 'Gust ×', 0.0, 3.0,
                      valinit=1.0, color=BLUE,
                      initcolor=BLUE, track_color=ACCENT)

    for s in (s_bias, s_gust):
        s.label.set_color(TEXT)
        s.valtext.set_color(TEXT)

    # ── Buttons ───────────────────────────────────────────────────────────
    btn_specs = [
        # [left, bottom, width, height], label,   bias, gust, color
        ([0.08, 0.12, 0.18, 0.09], 'CALM\n(0×)',     0.0, 0.0, BLUE  ),
        ([0.30, 0.12, 0.18, 0.09], 'DEFAULT\n(1×)',  1.0, 1.0, GREEN ),
        ([0.52, 0.12, 0.18, 0.09], 'STORM\n(2×)',    2.0, 2.0, YELLOW),
        ([0.74, 0.12, 0.18, 0.09], 'HURRICANE\n(3×)',3.0, 3.0, RED   ),
    ]

    buttons = []
    for spec in btn_specs:
        rect, label, b, g, col = spec
        ax_b = fig.add_axes(rect)
        ax_b.set_facecolor(PANEL)
        btn = W.Button(ax_b, label, color=PANEL, hovercolor=ACCENT)
        btn.label.set_color(col)
        btn.label.set_fontsize(9)
        btn.label.set_fontweight('bold')
        buttons.append((btn, b, g, col, label.split('\n')[0]))

    # ── Callbacks ─────────────────────────────────────────────────────────
    def _update_display(bias, gust):
        bias_bar.set_width(bias / 3.0)
        gust_bar.set_width(gust / 3.0)
        bias_label.set_text(f'{bias:.2f}×')
        gust_label.set_text(f'{gust:.2f}×')

        # Color status text
        total = (bias + gust) / 2
        if total < 0.1:
            status_txt.set_text('CALM'); status_txt.set_color(BLUE)
        elif total < 1.2:
            status_txt.set_text('DEFAULT'); status_txt.set_color(GREEN)
        elif total < 2.1:
            status_txt.set_text('STORM'); status_txt.set_color(YELLOW)
        else:
            status_txt.set_text('HURRICANE'); status_txt.set_color(RED)
        fig.canvas.draw_idle()

    def on_bias(val):
        node.set_wind(s_bias.val, s_gust.val)
        _update_display(s_bias.val, s_gust.val)

    def on_gust(val):
        node.set_wind(s_bias.val, s_gust.val)
        _update_display(s_bias.val, s_gust.val)

    s_bias.on_changed(on_bias)
    s_gust.on_changed(on_gust)

    for btn, b_val, g_val, _, _ in buttons:
        def make_cb(bv, gv):
            def cb(_event):
                s_bias.set_val(bv)
                s_gust.set_val(gv)
                node.set_wind(bv, gv)
                _update_display(bv, gv)
            return cb
        btn.on_clicked(make_cb(b_val, g_val))

    plt.show()
    rclpy.shutdown()
