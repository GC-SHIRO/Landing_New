#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
船舶速度轨迹控制器
==================

船在 episode 中不再被强制设置到参考位置；控制器只发布速度命令，
由 Gazebo 积分当前位置。reset/teleport 是唯一直接改船位置的地方。
"""

import math

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState, ModelStates


class ShipMotionController:
    def __init__(self, ship_name="wamv", init_pos=(10, 5), init_z=0.0,
                 kp=0.2, max_speed=None):
        self.ship_name = ship_name
        self.init_pos = np.array(init_pos, dtype=np.float64)
        self.init_z = float(init_z)
        self.kp = float(kp)
        # A supplied value is a hard safety cap.  Previously _ensure_max_speed
        # could raise an explicitly requested cap while compensating tracking
        # error, which made a "1 m/s" scenario publish faster commands.
        self.max_speed = None if max_speed is None else float(max_speed)
        self._hard_max_speed = max_speed is not None

        self._set_state_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self._model_sub = rospy.Subscriber("/gazebo/model_states", ModelStates,
                                           self._model_states_callback, queue_size=1)

        self._target_vel = np.zeros(2, dtype=np.float64)
        self._mode = "constant"
        self._rng = np.random.RandomState(None)

        self._lin_unit = np.array([1.0, 0.0], dtype=np.float64)
        self._lin_speed = 0.0
        self._lin_max_disp = 15.0

        self._speed_schedule = []
        self._varspeed_last_speed = 0.0

        self._sine_v = 0.8
        self._sine_amp = 3.0
        self._sine_wavelen = 12.0
        self._sine_w = 0.0
        self._sine_v_center = 0.0
        self._sine_cl = np.array([1.0, 0.0], dtype=np.float64)
        self._sine_perp = np.array([0.0, 1.0], dtype=np.float64)

        self._circle_radius = 5.0
        self._circle_period = 30.0
        self._circle_w = 0.0
        self._circle_speed = 0.0
        self._circle_phi0 = 0.0
        self._circle_center = self.init_pos.copy()

        self._combined_curve = "sine"

        self._pos = self.init_pos.copy()
        self._vel = np.zeros(2, dtype=np.float64)
        self._z = self.init_z
        self._vz = 0.0
        self._yaw = 0.0
        self._pose = None
        self._received_model_state = False
        self._last_cmd = np.zeros(2, dtype=np.float64)
        self._ref_pos = self.init_pos.copy()
        self._ref_t = 0.0

        rospy.loginfo(f"[ShipMotion] velocity trajectory servo init: {ship_name}")

    def set_mode_constant(self, vx, vy):
        self._mode = "constant"
        self._target_vel = np.array([float(vx), float(vy)], dtype=np.float64)
        self._configure_linear_direction(vx, vy)
        self._ensure_max_speed()
        rospy.loginfo(
            f"[ShipMotion] constant velocity target=({self._target_vel[0]:.3f},"
            f"{self._target_vel[1]:.3f}), kp={self.kp:.3f}, max_speed={self.max_speed:.3f}"
        )

    def set_mode_linear(self, vx, vy, max_displacement=15.0):
        self._mode = "linear"
        self._target_vel = np.array([float(vx), float(vy)], dtype=np.float64)
        self._configure_linear_direction(vx, vy)
        self._lin_max_disp = float(max_displacement)
        self._ensure_max_speed()
        rospy.loginfo(f"[ShipMotion] linear mode max_disp={self._lin_max_disp:.2f}")

    def set_mode_varspeed(self, direction, v_range, seed=None):
        self._mode = "varspeed"
        vx, vy = direction
        self._configure_linear_direction(vx, vy)
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._speed_schedule = self._gen_schedule(v_range)
        self._varspeed_last_speed = float(v_range[0])
        self._ensure_max_speed(v_range[1])
        rospy.loginfo(f"[ShipMotion] varspeed mode range=({v_range[0]:.2f},{v_range[1]:.2f})")

    def set_mode_sine(self, v, amp, wavelen, vx=1.0, vy=0.0):
        self._mode = "sine"
        self._configure_sine(v, amp, wavelen, vx, vy)
        self._ensure_max_speed(v)
        rospy.loginfo(f"[ShipMotion] sine mode v={v:.2f}, amp={amp:.2f}, wavelen={wavelen:.2f}")

    def set_mode_circle(self, radius, period, vx=1.0, vy=0.0):
        self._mode = "circle"
        self._configure_circle(radius, period, vx, vy)
        self._ensure_max_speed(self._circle_speed)
        rospy.loginfo(f"[ShipMotion] circle mode radius={radius:.2f}, period={period:.2f}")

    def set_mode_combined(self, curve_mode, v_range, seed=None):
        self._mode = "combined"
        self._combined_curve = curve_mode
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._speed_schedule = self._gen_schedule(v_range)
        self._varspeed_last_speed = float(v_range[0])
        self._ensure_max_speed(v_range[1])
        rospy.loginfo(f"[ShipMotion] combined mode curve={curve_mode}, range=({v_range[0]:.2f},{v_range[1]:.2f})")

    def wait_for_odom(self, timeout=15.0):
        if self._received_model_state:
            return True

        start = rospy.Time.now()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self._received_model_state:
                return True
            if (rospy.Time.now() - start).to_sec() > timeout:
                return False
            rate.sleep()
        return False

    def step(self, t):
        t = float(t)
        p_ref, v_ref = self._reference(t)
        err = p_ref - self._pos
        v_cmd = v_ref + self.kp * err
        v_cmd = self._limit_speed(v_cmd)

        self._last_cmd = v_cmd.copy()
        self._publish_velocity(v_cmd)
        return self._pos.copy()

    def teleport_to_origin(self):
        self._pos = self.init_pos.copy()
        self._vel = np.zeros(2, dtype=np.float64)
        self._z = self.init_z
        self._vz = 0.0
        self._yaw = 0.0
        self._last_cmd = np.zeros(2, dtype=np.float64)
        self._ref_pos = self.init_pos.copy()
        self._ref_t = 0.0

        msg = ModelState()
        msg.model_name = self.ship_name
        msg.reference_frame = "world"
        msg.pose.position.x = float(self.init_pos[0])
        msg.pose.position.y = float(self.init_pos[1])
        msg.pose.position.z = self.init_z
        msg.pose.orientation.w = 1.0
        self._pose = msg.pose
        self._set_state_pub.publish(msg)
        rospy.loginfo(f"[ShipMotion] teleport -> ({self.init_pos[0]:.1f},{self.init_pos[1]:.1f})")

    def get_current_pos(self):
        return self._pos.copy()

    def get_landing_target(
        self,
        marker_offset_z,
        marker_offset_x=0.0,
        marker_offset_y=0.0,
    ):
        cos_yaw = math.cos(self._yaw)
        sin_yaw = math.sin(self._yaw)
        world_offset_x = cos_yaw * marker_offset_x - sin_yaw * marker_offset_y
        world_offset_y = sin_yaw * marker_offset_x + cos_yaw * marker_offset_y
        return (
            float(self._pos[0] + world_offset_x),
            float(self._pos[1] + world_offset_y),
            float(self._z + marker_offset_z),
        )

    def get_landing_velocity(self):
        return (
            float(self._vel[0]),
            float(self._vel[1]),
            float(self._vz),
        )

    def get_state(self):
        return {
            "pos": self._pos.copy(),
            "vel": self._vel.copy(),
            "pos_z": float(self._z),
            "vel_z": float(self._vz),
            "cmd_vel": self._last_cmd.copy(),
            "speed": float(np.linalg.norm(self._vel)),
            "cmd_speed": float(np.linalg.norm(self._last_cmd)),
            "yaw": self._yaw,
            "mode": self._mode,
        }

    @property
    def debug_info(self):
        return {
            "pos": self._pos.copy(),
            "vel": self._vel.copy(),
            "cmd_vel": self._last_cmd.copy(),
            "yaw": self._yaw,
            # Backward-compatible keys for existing Step0 print code.
            "odom_pos": self._pos.copy(),
            "odom_vel": self._vel.copy(),
            "left_angle": self._yaw,
            "mode": self._mode,
        }

    def shutdown(self):
        self._publish_velocity(np.zeros(2, dtype=np.float64))
        if self._model_sub is not None:
            self._model_sub.unregister()
        rospy.loginfo("[ShipMotion] shutdown")

    def _publish_zero(self):
        self._last_cmd = np.zeros(2, dtype=np.float64)
        self._publish_velocity(self._last_cmd)

    def _model_states_callback(self, msg):
        try:
            idx = msg.name.index(self.ship_name)
        except ValueError:
            return

        pose = msg.pose[idx]
        twist = msg.twist[idx]
        self._pose = pose
        self._pos[0] = pose.position.x
        self._pos[1] = pose.position.y
        self._vel[0] = twist.linear.x
        self._vel[1] = twist.linear.y
        self._z = pose.position.z
        self._vz = twist.linear.z
        self._yaw = self._yaw_from_orientation(pose.orientation)
        self._received_model_state = True

    def _publish_velocity(self, vel):
        pose = self._pose if self._pose is not None else self._origin_pose()
        msg = ModelState()
        msg.model_name = self.ship_name
        msg.reference_frame = "world"
        msg.pose = pose
        msg.twist.linear.x = float(vel[0])
        msg.twist.linear.y = float(vel[1])
        msg.twist.linear.z = 0.0
        self._set_state_pub.publish(msg)

    def _limit_speed(self, vel):
        speed = float(np.linalg.norm(vel))
        if self.max_speed is None or speed <= self.max_speed or speed < 1e-9:
            return vel
        return vel * (self.max_speed / speed)

    def _reference(self, t):
        if self._mode == "constant":
            return self.init_pos + self._target_vel * t, self._target_vel.copy()

        if self._mode == "linear":
            return self._linear_reference(t)

        if self._mode == "varspeed":
            return self._integrated_reference(t, self._varspeed_velocity)

        if self._mode == "sine":
            return self._sine_reference(t, speed_scale=None)

        if self._mode == "circle":
            return self._circle_reference(t, speed_scale=None)

        if self._mode == "combined":
            return self._integrated_reference(t, self._combined_velocity)

        return self.init_pos.copy(), np.zeros(2, dtype=np.float64)

    def _linear_reference(self, t):
        if self._lin_speed < 1e-9:
            return self.init_pos.copy(), np.zeros(2, dtype=np.float64)

        period = self._lin_max_disp / self._lin_speed
        phase = t % (2.0 * period)
        if phase < period:
            disp = phase * self._lin_speed
            sign = 1.0
        else:
            disp = (2.0 * period - phase) * self._lin_speed
            sign = -1.0
        vel = sign * self._lin_unit * self._lin_speed
        pos = self.init_pos + self._lin_unit * disp
        return pos, vel

    def _sine_reference(self, t, speed_scale=None):
        cl, pp = self._sine_cl, self._sine_perp
        w, amp, vc = self._sine_w, self._sine_amp, self._sine_v_center
        pos = self.init_pos + vc * t * cl + amp * math.sin(w * t) * pp
        vel = vc * cl + amp * w * math.cos(w * t) * pp
        return pos, self._scale_velocity(vel, speed_scale)

    def _circle_reference(self, t, speed_scale=None):
        phi = self._circle_phi0 + self._circle_w * t
        pos = self._circle_center + self._circle_radius * np.array([math.cos(phi), math.sin(phi)])
        vel = self._circle_speed * np.array([-math.sin(phi), math.cos(phi)])
        return pos, self._scale_velocity(vel, speed_scale)

    @staticmethod
    def _scale_velocity(vel, speed_scale):
        if speed_scale is None:
            return vel
        norm = float(np.linalg.norm(vel))
        if norm < 1e-9:
            return vel
        return vel * (float(speed_scale) / norm)

    def _configure_linear_direction(self, vx, vy):
        self._lin_speed = math.hypot(vx, vy)
        if self._lin_speed > 1e-9:
            self._lin_unit = np.array([vx, vy], dtype=np.float64) / self._lin_speed
        else:
            self._lin_unit = np.array([1.0, 0.0], dtype=np.float64)

    def _configure_sine(self, v, amp, wavelen, vx, vy):
        self._sine_v = float(v)
        self._sine_amp = float(amp)
        self._sine_wavelen = float(wavelen)
        d = math.hypot(vx, vy)
        self._sine_cl = (
            np.array([vx / d, vy / d], dtype=np.float64)
            if d > 1e-9 else np.array([1.0, 0.0], dtype=np.float64)
        )
        self._sine_perp = np.array([-self._sine_cl[1], self._sine_cl[0]], dtype=np.float64)
        ratio = (2.0 * math.pi * self._sine_amp / self._sine_wavelen) ** 2
        self._sine_v_center = self._sine_v / math.sqrt(1.0 + ratio / 2.0)
        self._sine_w = 2.0 * math.pi * self._sine_v_center / self._sine_wavelen

    def _configure_circle(self, radius, period, vx, vy):
        self._circle_radius = float(radius)
        self._circle_period = float(period)
        self._circle_w = 2.0 * math.pi / self._circle_period
        self._circle_speed = self._circle_w * self._circle_radius
        d = math.hypot(vx, vy)
        ux, uy = (vx / d, vy / d) if d > 1e-9 else (1.0, 0.0)
        nx, ny = -uy, ux
        self._circle_center = self.init_pos + self._circle_radius * np.array([nx, ny])
        self._circle_phi0 = math.atan2(-self._circle_radius * ny, -self._circle_radius * nx)

    def _gen_schedule(self, v_range, max_t=600.0):
        schedule = []
        t = 0.0
        while t <= max_t:
            schedule.append((t, float(self._rng.uniform(*v_range))))
            t += float(self._rng.uniform(2.0, 5.0))
        return schedule

    def _scheduled_speed(self, t):
        speed = self._varspeed_last_speed
        for start_t, scheduled_speed in self._speed_schedule:
            if t < start_t:
                break
            speed = scheduled_speed
        self._varspeed_last_speed = speed
        return speed

    def _integrated_reference(self, t, velocity_fn):
        if t < self._ref_t:
            self._ref_pos = self.init_pos.copy()
            self._ref_t = 0.0

        dt = max(0.0, t - self._ref_t)
        vel = velocity_fn(t)
        self._ref_pos = self._ref_pos + vel * dt
        self._ref_t = t
        return self._ref_pos.copy(), vel

    def _varspeed_velocity(self, t):
        return self._scheduled_speed(t) * self._lin_unit

    def _combined_velocity(self, t):
        speed = self._scheduled_speed(t)
        if self._combined_curve == "circle":
            _, base_vel = self._circle_reference(t, speed_scale=None)
        else:
            _, base_vel = self._sine_reference(t, speed_scale=None)
        return self._scale_velocity(base_vel, speed)

    def _ensure_max_speed(self, nominal_speed=None):
        if self._hard_max_speed:
            return
        if nominal_speed is None:
            nominal_speed = float(np.linalg.norm(self._target_vel))
        needed = max(0.5, 2.0 * float(nominal_speed))
        self.max_speed = needed if self.max_speed is None else max(self.max_speed, needed)

    def _origin_pose(self):
        msg = ModelState()
        msg.pose.position.x = float(self.init_pos[0])
        msg.pose.position.y = float(self.init_pos[1])
        msg.pose.position.z = self.init_z
        msg.pose.orientation.w = 1.0
        return msg.pose

    @staticmethod
    def _yaw_from_orientation(q):
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)
