#!/usr/bin/env python3
"""Run the existing Step1 WAM-V straight-line reciprocating controller."""

import argparse
import os
import sys
import time
from pathlib import Path

import rospy


PROJECT_TD3 = os.environ.get(
    "LANDING_NEW_TD3_ROOT",
    str(Path(__file__).resolve().parents[2] / "TD3-main"),
)
if PROJECT_TD3 not in sys.path:
    sys.path.insert(0, PROJECT_TD3)

from Simulation.ship_motion import ShipMotionController  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=320.0)
    parser.add_argument("--vx", type=float, default=0.15)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--max-displacement", type=float, default=15.0)
    args = parser.parse_args()

    rospy.init_node("step1_recording_ship_motion", anonymous=True)
    controller = ShipMotionController(ship_name="wamv", init_pos=(10, 5))
    controller.set_mode_linear(args.vx, args.vy, args.max_displacement)
    if not controller.wait_for_odom(timeout=20.0):
        raise RuntimeError("WAM-V model state was not received")
    controller.teleport_to_origin()
    rospy.sleep(1.0)

    started_wall = time.monotonic()
    started_sim = rospy.Time.now().to_sec()
    rate = rospy.Rate(20)
    try:
        while not rospy.is_shutdown() and time.monotonic() - started_wall < args.duration:
            controller.step(rospy.Time.now().to_sec() - started_sim)
            rate.sleep()
    finally:
        controller.shutdown()

    state = controller.get_state()
    print(
        "SHIP_MOTION_DONE elapsed=%.3f pos=(%.3f,%.3f)"
        % (
            time.monotonic() - started_wall,
            state["pos"][0],
            state["pos"][1],
        )
    )


if __name__ == "__main__":
    main()
