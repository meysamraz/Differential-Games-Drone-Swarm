"""
Drone keyboard teleop – WASD scheme.

  W / S          → altitude UP / DOWN
  A / D          → strafe LEFT / RIGHT
  ↑ / ↓ arrows   → forward / backward
  ← / → arrows   → yaw LEFT / RIGHT
  Space          → stop (hold current altitude)
  Ctrl+C         → quit
"""

import sys
import select
import tty
import termios
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


HELP = """
=== Drone Teleop (WASD) ===
  W / S     : altitude UP / DOWN
  A / D     : strafe LEFT / RIGHT
  ↑ / ↓     : forward / backward
  ← / →     : yaw left / right
  Space     : stop (hold hover)
  Ctrl+C    : quit
===========================
"""

LINEAR_SPEED  = 0.8   # m/s
ANGULAR_SPEED = 0.6   # rad/s

# How long to hold a command after each keypress (covers the ~500 ms
# keyboard auto-repeat delay so one tap produces visible movement).
KEY_HOLD = 0.45       # seconds


def _get_key(timeout: float = 0.05):
    """Return pressed key string, or None on timeout."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready = select.select([sys.stdin], [], [], timeout)[0]
        if not ready:
            return None
        ch = sys.stdin.read(1)
        # Arrow keys → 3-byte ESC sequence: \x1b [ A/B/C/D
        if ch == '\x1b':
            ready = select.select([sys.stdin], [], [], 0.02)[0]
            if ready and sys.stdin.read(1) == '[':
                ready = select.select([sys.stdin], [], [], 0.02)[0]
                if ready:
                    return {'A': 'UP', 'B': 'DOWN',
                            'C': 'RIGHT', 'D': 'LEFT'}.get(sys.stdin.read(1), '')
            return ''   # bare ESC – ignore
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _build_msg(key):
    """Return (Twist, hold) for the given key. hold=False means stop command."""
    msg = Twist()
    if key == 'w':
        msg.linear.z  =  LINEAR_SPEED
    elif key == 's':
        msg.linear.z  = -LINEAR_SPEED
    elif key == 'a':
        msg.linear.y  =  LINEAR_SPEED
    elif key == 'd':
        msg.linear.y  = -LINEAR_SPEED
    elif key == 'UP':
        msg.linear.x  =  LINEAR_SPEED
    elif key == 'DOWN':
        msg.linear.x  = -LINEAR_SPEED
    elif key == 'LEFT':
        msg.angular.z =  ANGULAR_SPEED
    elif key == 'RIGHT':
        msg.angular.z = -ANGULAR_SPEED
    elif key == ' ':
        return msg, False   # explicit stop
    else:
        return None, True   # unknown key – don't update
    return msg, True


class DroneTeleop(Node):
    def __init__(self):
        super().__init__('drone_teleop')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def run(self):
        print(HELP)
        active_cmd = Twist()
        expires_at = 0.0

        try:
            while rclpy.ok():
                key = _get_key(timeout=0.05)
                now = time.monotonic()

                if key == '\x03':       # Ctrl+C
                    break

                if key is not None and key != '':
                    msg, hold = _build_msg(key)

                    if msg is not None:
                        if hold:
                            active_cmd = msg
                            expires_at = now + KEY_HOLD
                        else:
                            # Space – explicit stop
                            active_cmd = Twist()
                            expires_at = 0.0

                        # Status feedback (overwrite same line)
                        label = {'UP': '↑', 'DOWN': '↓',
                                 'LEFT': '←', 'RIGHT': '→'}.get(key, key.upper())
                        sys.stdout.write(
                            f'\r[{label}]  alt={active_cmd.linear.z:+.1f}'
                            f'  fwd={active_cmd.linear.x:+.1f}'
                            f'  str={active_cmd.linear.y:+.1f}'
                            f'  yaw={active_cmd.angular.z:+.1f}    '
                        )
                        sys.stdout.flush()

                # Expire hold
                if now > expires_at:
                    active_cmd = Twist()

                self.pub.publish(active_cmd)

        finally:
            sys.stdout.write('\r\nStopping – drone holds position.\r\n')
            sys.stdout.flush()
            self.pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = DroneTeleop()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
