#!/usr/bin/env bash
# Train SuperMimic on echo. Assumes the 29-joint yaw-only waist: waist pitch and roll are phase two, after this design walks.
#   train/run_supermimic.sh <name> <motion.pkl> [field=value ...]
# dr=sonic2: encoder, IMU and delay errors, gains, pushes, plus (supermimic_robot.json) mass, armature and friction ranges and the
# torque-speed limit. The X2's 4096 envs and 320k iterations are the paper's; this PC runs the "local" preset for a design check.
set -u
NAME=$1; MOTION=$2; shift 2
ECHO=${ECHO:-C:/Users/PC/projects/echo}
cd "${GWBC:-/c/Users/PC/projects/gwbc-echo}"
mkdir -p logs_sm
export PYTHONPATH=. SM_ROBOT_CFG=${SM_ROBOT_CFG:-$ECHO/train/supermimic_robot.json}
/c/Users/PC/.venvs/sm/Scripts/python.exe -m gear_sonic.supermimic.train --motion "$MOTION" --log-dir logs_sm/$NAME --set \
  num_envs=2048 buffer_steps=500 pi_batch=2048 iterations=8000 lr=0.0002 lr_decay_iters=8000 lr_final_frac=0.03 \
  log_every=50 ckpt_every=500 eval_every=0 compile=false wandb=false wm_warmup=0 dr=sonic2 plant=echo "$@" \
  > logs_sm/$NAME.out 2>&1
status=$?
echo "training exit $status" >> logs_sm/$NAME.out
exit $status
