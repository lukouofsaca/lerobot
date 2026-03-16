#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="lerobot"
SNAPSHOT_FILE="/tmp/lerobot_before_ik_fix.txt"
ENV_PYTHON="/home/zhbs/miniconda3/envs/${ENV_NAME}/bin/python"
URDF_PATH="/home/zhbs/pika_ros/install/piper_description/share/piper_description/urdf/piper_description.urdf"

run_smoke_test() {
  env -u LD_LIBRARY_PATH -u PYTHONPATH -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH -u COLCON_PREFIX_PATH \
    "${ENV_PYTHON}" - <<PY
import importlib

mods = ["eigenpy", "pinocchio", "placo"]
for name in mods:
    importlib.import_module(name)
print("imports ok:", ", ".join(mods))

import placo
robot = placo.RobotWrapper("${URDF_PATH}")
print("placo RobotWrapper ok, joints:", len(list(robot.joint_names())))
PY
}

install_combo() {
  local placo_version="$1"
  local pin_version="$2"

  echo "Trying combo: placo==${placo_version}, pin==${pin_version}"
  "${ENV_PYTHON}" -m pip install --no-cache-dir --force-reinstall \
    "placo==${placo_version}" \
    "pin==${pin_version}"
}

echo "[1/5] Exporting conda snapshot to ${SNAPSHOT_FILE}"
conda list -n "${ENV_NAME}" > "${SNAPSHOT_FILE}"

echo "[2/5] Uninstalling potential conflicting IK runtime wheels"
"${ENV_PYTHON}" -m pip uninstall -y \
  placo pin libpinocchio eigenpy coal cmeel cmeel-boost cmeel-urdfdom libcoal || true

echo "[3/5] Installing candidate IK runtime combinations"

if install_combo "0.9.20" "3.8.0" && run_smoke_test; then
  echo "Selected combo: placo==0.9.20, pin==3.8.0"
elif install_combo "0.9.16" "3.4.0" && run_smoke_test; then
  echo "Selected combo: placo==0.9.16, pin==3.4.0"
else
  echo "[ERROR] All tested placo/pin combinations failed"
  exit 1
fi

echo "[4/5] Final smoke test via diagnose script"
env -u LD_LIBRARY_PATH -u PYTHONPATH -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH -u COLCON_PREFIX_PATH \
  "${ENV_PYTHON}" /home/zhbs/sda/zyx/lerobot/examples/pika_to_piper/diagnose_ik_pipeline.py \
  --env-config /home/zhbs/sda/zyx/lerobot/src/lerobot/configs/env_config_pika_piper.json

echo "[5/5] Done"

echo "Done. Snapshot for rollback: ${SNAPSHOT_FILE}"
