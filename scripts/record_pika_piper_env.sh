#!/usr/bin/env bash
set -euo pipefail

# Local env-adapted LeRobot recording command (Pika teleop + Piper follower)
# Usage:
#   q
#   bash scripts/record_pika_piper_env.sh

lerobot-record \
  --robot.type=piper_follower \
  --robot.motors.can_name=can0 \
  --robot.cameras='{
    wrist: {type: intelrealsense, serial_number_or_name: "230322271133", width: 640, height: 480, fps: 30},
    left: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30},
    right: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30},
  }' \
  --teleop.type=pika \
  --teleop.sense_port=/dev/ttyUSB81 \
  --teleop.tracker_device=T20 \
  --teleop.piper_description_dir=/home/zhbs/pika_ros/install/piper_description/share/piper_description \
  --teleop.output_mode=joint \
  --teleop.max_joint_step_deg=2.0 \
  --display_data=false \
  --dataset.fps=15 \
  --dataset.repo_id=zhbs/pika_piper_pick_cube_mini_10eps \
  --dataset.num_episodes=10 \
  --dataset.single_task="Pick up the cube" \
  --dataset.push_to_hub=false
