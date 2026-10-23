# Franka real-robot ROS code

This directory is a compact BM2PC package extracted from the Franka
real-robot research branch.

## Included

- `src/`: Franka-side BM2PC clients, including the obstacle-aware variant, and
  Cartesian pose-control nodes.
- `include/`: controller and trajectory headers.
- `launch/`: launch files for the C++ real-robot nodes.
- `config/`: ROS controller configuration.
- ROS package metadata and the controller plugin description.

Python experiments, training code, collected CSV data, plots, caches, and logs
are intentionally excluded.

The C++ nodes retain their original optional post-run plotting hooks. Those
hooks reference plotting scripts from the full research repository; their
absence does not affect the real-time control loop, but automatic plot
generation is unavailable in this compact extract.

## Configuration

Set the robot and remote inference addresses before launching:

```bash
export ROBOT_ARM_IP=<franka-controller-ip>
export BM2PC_SERVER_IP=<inference-server-ip>
```

The BM2PC launch files fall back to `127.0.0.1` for the inference server. The
robot launch files retain the standard private-network fallback used by the
original experiment and allow it to be overridden through `ROBOT_ARM_IP`.

Main launch files:

- `hrii_bm2pc_franka_C.launch`: BM2PC real-robot experiment.
- `hrii_bm2pc_franka_C_obs.launch`: obstacle-aware BM2PC experiment.
- `hrii_bm2pc_franka_cart_pose.launch`: direct libfranka Cartesian pose controller.

## External dependencies

This package targets ROS Noetic and depends on libfranka/franka_ros, ZeroMQ,
Eigen, matlogger2, MoveIt messages, OptiTrack integration, and the HRII ROS
packages listed in `package.xml` and referenced by the launch files.

## Before publishing

Choose an explicit open-source license and replace the placeholder maintainer
metadata in `package.xml`. Those decisions cannot be inferred safely from the
source code.
