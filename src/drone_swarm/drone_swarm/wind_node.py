"""
Wind disturbance node.

Publishes per-drone wind forces (bias + sinusoidal gusts) to /droneN/wind_force.
Subscribes to /wind_scale for real-time adjustment from wind_controller.py.

/wind_scale message: Float64MultiArray with [bias_scale, gust_scale]
  bias_scale : 0.0 = no bias,  1.0 = default,  3.0 = triple strength
  gust_scale : same range for gust amplitude
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench
from std_msgs.msg import Float64MultiArray

DRONE_NS = ['drone1', 'drone2', 'drone3']

WIND_BIAS  = [(0.045, -0.030), (-0.030, 0.045), (0.015, 0.030)]  # N per drone
GUST_AMP   = [0.08,  0.15,  0.22]   # base gust amplitude (N)
GUST_FREQ  = [0.8,   0.6,   1.0]    # rad/s
GUST_PHASE = [0.0,   1.2,   2.4]    # rad

DT = 0.05   # 20 Hz


class WindNode(Node):

    def __init__(self):
        super().__init__('wind_node')

        self.bias_scale = 1.0
        self.gust_scale = 1.0

        self.pubs = [
            self.create_publisher(Wrench, f'/{ns}/wind_force', 10)
            for ns in DRONE_NS
        ]
        self.create_subscription(
            Float64MultiArray, '/wind_scale', self._scale_cb, 10
        )
        self.t = 0.0
        self.create_timer(DT, self._publish)
        self.get_logger().info('Wind node ready — listening on /wind_scale')

    def _scale_cb(self, msg: Float64MultiArray):
        if len(msg.data) >= 2:
            self.bias_scale = float(msg.data[0])
            self.gust_scale = float(msg.data[1])
            self.get_logger().info(
                f'Wind updated  bias×{self.bias_scale:.2f}  gust×{self.gust_scale:.2f}',
                throttle_duration_sec=0.5,
            )

    def _publish(self):
        for i in range(3):
            gx = GUST_AMP[i] * math.sin(GUST_FREQ[i] * self.t + GUST_PHASE[i])
            gy = GUST_AMP[i] * math.cos(GUST_FREQ[i] * self.t + GUST_PHASE[i] * 0.7)

            msg = Wrench()
            msg.force.x = self.bias_scale * WIND_BIAS[i][0] + self.gust_scale * gx
            msg.force.y = self.bias_scale * WIND_BIAS[i][1] + self.gust_scale * gy
            self.pubs[i].publish(msg)

        self.t += DT


def main(args=None):
    rclpy.init(args=args)
    node = WindNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
