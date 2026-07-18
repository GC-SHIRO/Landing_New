#!/usr/bin/env python3
"""Record Gazebo third-person and YOLO debug image topics to MP4.

This mirrors the older two-view recorder style, but is controller-agnostic for
PID, KF+PID, and STAR-TD3 reviewer landing videos.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

for path in [
    "/opt/ros/noetic/lib/python3/dist-packages",
]:
    if path not in sys.path:
        sys.path.insert(0, path)

import cv2
import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import PointStamped, PoseStamped
from mavros_msgs.msg import State
from sensor_msgs.msg import Image


def image_to_bgr(msg):
    enc = msg.encoding.lower()
    channels = {"mono8": 1, "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(enc)
    if channels is None:
        raise ValueError(f"unsupported image encoding: {msg.encoding}")
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.step // channels, channels))[:, :msg.width]
    if enc == "rgb8":
        return arr[:, :, ::-1].copy()
    if enc == "bgr8":
        return arr.copy()
    if enc == "mono8":
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def gazebo_camera_quat(pitch, yaw):
    # Gazebo camera optical axis points +X. This is q_y(pitch) followed by
    # q_z(yaw), so +pitch tilts the optical axis downward.
    return (
        -math.sin(pitch * 0.5) * math.sin(yaw * 0.5),
        math.sin(pitch * 0.5) * math.cos(yaw * 0.5),
        math.sin(yaw * 0.5) * math.cos(pitch * 0.5),
        math.cos(yaw * 0.5) * math.cos(pitch * 0.5),
    )


def clamp(value, lo, hi):
    return min(max(float(value), float(lo)), float(hi))


def target_from_model(model, offset):
    yaw = yaw_from_quat(model.pose.orientation)
    c = math.cos(yaw)
    s = math.sin(yaw)
    ox, oy, oz = offset
    return (
        model.pose.position.x + c * ox - s * oy,
        model.pose.position.y + s * ox + c * oy,
        model.pose.position.z + oz,
        yaw,
    )


class VideoSink:
    def __init__(self, path: Path, fps: float):
        self.path = Path(path)
        self.tmp_path = self.path.with_suffix(".avi")
        self.fps = float(fps)
        self.writer = None
        self.frames = 0
        self.error = ""
        self.lock = threading.Lock()
        self.closed = False
        self.first_frame_wall = None
        self.last_frame_wall = None

    @property
    def effective_fps(self):
        if self.frames < 2 or self.first_frame_wall is None or self.last_frame_wall is None:
            return self.fps
        span = self.last_frame_wall - self.first_frame_wall
        if span <= 0.0:
            return self.fps
        return max(1.0, min(self.fps, (self.frames - 1) / span))

    def write(self, frame):
        with self.lock:
            if self.closed:
                return
            if self.writer is None:
                h, w = frame.shape[:2]
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.writer = cv2.VideoWriter(str(self.tmp_path), cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (w, h))
                if not self.writer.isOpened():
                    raise RuntimeError(f"failed to open video writer: {self.tmp_path}")
            self.writer.write(frame)
            self.frames += 1
            now = time.monotonic()
            if self.first_frame_wall is None:
                self.first_frame_wall = now
            self.last_frame_wall = now

    def release(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.writer is not None:
                self.writer.release()
                self.writer = None
        if self.frames <= 0 or not self.tmp_path.exists():
            return
        try:
            subprocess.run([
                "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
                "-r", f"{self.effective_fps:.6f}",
                "-i", str(self.tmp_path),
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(self.path),
            ], check=True, timeout=300, stdin=subprocess.DEVNULL)
            if self.path.exists() and self.path.stat().st_size > 0:
                self.tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            self.error = str(exc)


class Recorder:
    def __init__(self, args):
        self.args = args
        rospy.init_node("record_landing_gazebo_views", anonymous=True)
        self.lock = threading.Lock()
        self.start_wall = time.time()
        self.stop_requested = False
        self.closed = False
        self.latest = {
            "state": None,
            "pose": None,
            "point": None,
            "point_t": 0.0,
            "truth": None,
        }
        self.out_dir = Path(args.out_dir)
        self.third = VideoSink(self.out_dir / f"{args.controller}_third_person_gazebo.mp4", args.fps)
        self.third_raw = VideoSink(self.out_dir / f"{args.controller}_third_person_gazebo_raw.mp4", args.fps)
        self.camera = VideoSink(self.out_dir / f"{args.controller}_camera_yolo_debug.mp4", args.fps) if args.camera_topic else None
        self.samples_csv = self.out_dir / f"{args.controller}_gazebo_view_samples.csv"
        self.samples_jsonl = self.out_dir / f"{args.controller}_gazebo_view_samples.jsonl"
        self.csv_f = self.samples_csv.open("w", newline="", encoding="utf-8")
        self.json_f = self.samples_jsonl.open("w", encoding="utf-8")
        self.fields = [
            "t", "mode", "armed",
            "iris_x", "iris_y", "iris_z",
            "marker_x", "marker_y", "marker_z",
            "rel_x", "rel_y", "rel_z", "horiz_err",
            "local_x", "local_y", "local_z",
            "visual_x", "visual_y", "visual_z", "visual_age_s",
        ]
        self.csv_w = csv.DictWriter(self.csv_f, fieldnames=self.fields)
        self.csv_w.writeheader()
        self.get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        rospy.wait_for_service("/gazebo/get_model_state", timeout=20)
        self.follow_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1) if args.follow_camera_model else None
        rospy.Subscriber(args.third_topic, Image, self.third_cb, queue_size=2, buff_size=2**24)
        if args.camera_topic:
            rospy.Subscriber(args.camera_topic, Image, self.camera_cb, queue_size=1, buff_size=2**22)
        if args.visual_topic:
            rospy.Subscriber(args.visual_topic, PointStamped, self.point_cb, queue_size=10)
        if args.state_topic:
            rospy.Subscriber(args.state_topic, State, self.state_cb, queue_size=10)
        if args.pose_topic:
            rospy.Subscriber(args.pose_topic, PoseStamped, self.pose_cb, queue_size=10)

    def request_stop(self, signum=None, frame=None):
        self.stop_requested = True
        try:
            rospy.signal_shutdown(f"stop signal {signum}")
        except Exception:
            pass

    def state_cb(self, msg):
        with self.lock:
            self.latest["state"] = msg

    def pose_cb(self, msg):
        with self.lock:
            self.latest["pose"] = msg

    def point_cb(self, msg):
        with self.lock:
            self.latest["point"] = msg
            self.latest["point_t"] = time.time()

    def snapshot(self):
        with self.lock:
            return dict(self.latest)

    def sample_truth(self):
        primary = self.get_state(self.args.primary_model, "")
        target = self.get_state(self.args.target_model, "")
        if not (primary.success and target.success):
            return None
        mx, my, mz, _ = target_from_model(target, self.args.target_offset)
        ix = primary.pose.position.x
        iy = primary.pose.position.y
        iz = primary.pose.position.z
        return {
            "iris_x": ix,
            "iris_y": iy,
            "iris_z": iz,
            "marker_x": mx,
            "marker_y": my,
            "marker_z": mz,
            "rel_x": ix - mx,
            "rel_y": iy - my,
            "rel_z": iz - mz,
            "horiz_err": math.hypot(ix - mx, iy - my),
        }

    def update_follow_camera(self, truth):
        if self.follow_pub is None:
            return
        heading = math.atan2(truth["marker_y"] - truth["iris_y"], truth["marker_x"] - truth["iris_x"])
        cx = 0.5 * (truth["iris_x"] + truth["marker_x"])
        cy = 0.5 * (truth["iris_y"] + truth["marker_y"])
        cz = 0.5 * (truth["iris_z"] + truth["marker_z"])
        span_xy = math.hypot(truth["iris_x"] - truth["marker_x"], truth["iris_y"] - truth["marker_y"])
        span_z = abs(truth["iris_z"] - truth["marker_z"])
        if self.args.look_at_mode == "birdseye":
            st = ModelState()
            st.model_name = self.args.follow_camera_model
            st.reference_frame = "world"
            st.pose.position.x = cx
            st.pose.position.y = cy
            st.pose.position.z = max(truth["iris_z"], truth["marker_z"]) + max(self.args.follow_z, 12.0)
            qx, qy, qz, qw = gazebo_camera_quat(float(self.args.follow_pitch), heading)
            st.pose.orientation.x = qx
            st.pose.orientation.y = qy
            st.pose.orientation.z = qz
            st.pose.orientation.w = qw
            self.follow_pub.publish(st)
            return
        back = self.args.follow_back
        height = self.args.follow_z
        if self.args.look_at_mode == "midpoint3d":
            back = clamp(max(back, span_xy * 2.4 + 7.5, span_z * 0.9 + 8.0), self.args.min_follow_back, self.args.max_follow_back)
            height = clamp(max(height, span_z * 0.45 + 5.2), self.args.min_follow_z, self.args.max_follow_z)
        st = ModelState()
        st.model_name = self.args.follow_camera_model
        st.reference_frame = "world"
        st.pose.position.x = cx - back * math.cos(heading) - self.args.follow_side * math.sin(heading)
        st.pose.position.y = cy - back * math.sin(heading) + self.args.follow_side * math.cos(heading)
        st.pose.position.z = cz + height
        pitch = float(self.args.follow_pitch)
        if self.args.look_at_mode == "midpoint3d":
            horizontal = math.hypot(st.pose.position.x - cx, st.pose.position.y - cy)
            pitch = math.atan2(st.pose.position.z - cz, max(horizontal, 1e-6))
            pitch = clamp(pitch, 0.15, 1.35)
        yaw = heading
        qx, qy, qz, qw = gazebo_camera_quat(pitch, yaw)
        st.pose.orientation.x = qx
        st.pose.orientation.y = qy
        st.pose.orientation.z = qz
        st.pose.orientation.w = qw
        self.follow_pub.publish(st)

    def overlay(self, frame, title):
        if self.args.raw_only:
            return frame
        snap = self.snapshot()
        truth = snap.get("truth")
        now = time.time()
        lines = [title, f"{self.args.controller}  t={now - self.start_wall:.1f}s"]
        if truth:
            lines.append(f"truth rel=({truth['rel_x']:.2f},{truth['rel_y']:.2f},{truth['rel_z']:.2f}) herr={truth['horiz_err']:.2f}")
        pt = snap.get("point")
        age = now - snap.get("point_t", 0.0)
        if pt is not None and age < 2.0:
            lines.append(f"visual=({pt.point.x:.2f},{pt.point.y:.2f},{pt.point.z:.2f}) age={age:.1f}s")
        y = 28
        for text in lines:
            cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
            y += 24
        return frame

    def third_cb(self, msg):
        frame = image_to_bgr(msg)
        self.third_raw.write(frame)
        self.third.write(self.overlay(frame.copy(), "Gazebo third-person camera"))

    def camera_cb(self, msg):
        if self.camera is None:
            return
        frame = image_to_bgr(msg)
        self.camera.write(self.overlay(frame.copy(), "YOLO debug camera"))

    def write_sample(self, truth):
        snap = self.snapshot()
        row = {k: "" for k in self.fields}
        row["t"] = round(time.time() - self.start_wall, 3)
        row.update(truth)
        state = snap.get("state")
        if state:
            row["mode"] = state.mode
            row["armed"] = state.armed
        pose = snap.get("pose")
        if pose:
            row["local_x"] = pose.pose.position.x
            row["local_y"] = pose.pose.position.y
            row["local_z"] = pose.pose.position.z
        pt = snap.get("point")
        if pt:
            row["visual_x"] = pt.point.x
            row["visual_y"] = pt.point.y
            row["visual_z"] = pt.point.z
            row["visual_age_s"] = time.time() - snap.get("point_t", 0.0)
        self.csv_w.writerow(row)
        self.csv_f.flush()
        self.json_f.write(json.dumps(row) + "\n")
        self.json_f.flush()
        with self.lock:
            self.latest["truth"] = dict(truth)

    def run(self):
        try:
            rate = rospy.Rate(float(self.args.sample_hz))
            while not self.stop_requested and not rospy.is_shutdown() and time.time() - self.start_wall <= self.args.run_s:
                try:
                    truth = self.sample_truth()
                    if truth:
                        self.update_follow_camera(truth)
                        self.write_sample(truth)
                except Exception as exc:
                    rospy.logwarn_throttle(2.0, "landing view sample failed: %s", exc)
                try:
                    rate.sleep()
                except rospy.exceptions.ROSInterruptException:
                    break
        finally:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        sinks = [self.third, self.third_raw]
        if self.camera is not None:
            sinks.append(self.camera)
        for sink in sinks:
            sink.release()
        try:
            self.csv_f.close()
        except Exception:
            pass
        try:
            self.json_f.close()
        except Exception:
            pass
        summary = {
            "out_dir": str(self.out_dir),
            "third_person_gazebo": str(self.third.path),
            "third_person_gazebo_raw": str(self.third_raw.path),
            "camera_yolo_debug": str(self.camera.path) if self.camera is not None else "",
            "third_frames": self.third.frames,
            "third_raw_frames": self.third_raw.frames,
            "third_effective_fps": self.third.effective_fps,
            "third_raw_effective_fps": self.third_raw.effective_fps,
            "camera_frames": self.camera.frames if self.camera is not None else 0,
            "third_error": self.third.error,
            "third_raw_error": self.third_raw.error,
            "camera_error": self.camera.error if self.camera is not None else "",
            "samples_csv": str(self.samples_csv),
            "samples_jsonl": str(self.samples_jsonl),
        }
        (self.out_dir / f"{self.args.controller}_gazebo_view_record_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("GAZEBO_VIEW_RECORDER_SUMMARY=" + json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--controller", default="unknown")
    parser.add_argument("--run-s", type=float, default=90.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--third-topic", default="/landing_review/third_person/image_raw")
    parser.add_argument("--camera-topic", default="/marker_yolo_detector/debug_image")
    parser.add_argument("--primary-model", default="iris_1")
    parser.add_argument("--target-model", default="landing_pad")
    parser.add_argument("--target-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--visual-topic", default="/marker_yolo_detector/point")
    parser.add_argument("--state-topic", default="/iris_1/mavros/state")
    parser.add_argument("--pose-topic", default="/iris_1/mavros/local_position/pose")
    parser.add_argument("--follow-camera-model", default="")
    parser.add_argument("--follow-z", type=float, default=9.0)
    parser.add_argument("--follow-back", type=float, default=16.0)
    parser.add_argument("--follow-side", type=float, default=0.0)
    parser.add_argument("--follow-pitch", type=float, default=0.85)
    parser.add_argument("--look-at-mode", choices=["legacy", "midpoint3d", "birdseye"], default="midpoint3d")
    parser.add_argument("--min-follow-back", type=float, default=12.0)
    parser.add_argument("--max-follow-back", type=float, default=24.0)
    parser.add_argument("--min-follow-z", type=float, default=7.0)
    parser.add_argument("--max-follow-z", type=float, default=16.0)
    parser.add_argument("--raw-only", action="store_true")
    args = parser.parse_args()
    rec = Recorder(args)
    signal.signal(signal.SIGTERM, rec.request_stop)
    signal.signal(signal.SIGINT, rec.request_stop)
    rec.run()


if __name__ == "__main__":
    main()
