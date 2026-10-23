<div align="center">

# Multimodal MPC via Bayesian Set Optimization for Dynamic Manipulation of Deformable Linear Objects

**Youyuan Long<sup>1</sup>, Gokhan Solak<sup>1</sup>, Arash Ajoudani<sup>1</sup>**

<sup>1</sup>Human-Robot Interfaces and Interaction Lab, Istituto Italiano di Tecnologia (IIT), Genova, Italy

</div>

**BM2PC** is a Bayesian multimodal MPC framework for agile manipulation of deformable linear objects such as ropes, cables and hoses. It formulates multimodal planning as a set optimization over control sequences, and an efficient incremental update keeps several high-quality, behaviorally diverse control modes alive at almost no extra cost. A learned execution dynamics model compensates for the mismatch between commanded and executed robot motion, enabling fast and accurate real-world manipulation without task-specific policy training.

---

## 🎬 Overview Video

https://github.com/user-attachments/assets/705f35a4-955e-4808-a86d-d184d9df0d6e


---

## ✨ Highlights

BM2PC is a sampling-based **Bayesian Multimodal MPC** framework for agile manipulation of ropes, cables and hoses.

- **Bayesian set optimization.** Building on the control-as-inference posterior, multimodal planning is formulated as selecting a *set* of control sequences that are both high-posterior and behaviorally diverse.
- **Almost-free multimodal update.** An incremental greedy update keeps several distinct control modes alive for **~1.55 ms** per step, reaching a mean optimality gap of only **1.14%** w.r.t. exhaustive set optimization while being **~9× faster** in planning.
- **Execution-aware prediction.** A learned GRU-based execution dynamics model predicts the *actually executed* end-effector motion (0.43 cm avg. max error vs. 4.57 cm command–execution mismatch), enabling reliable high-speed deployment on a Franka Emika Panda.
- **No task-specific training.** One parameter set tracks all 18 simulated trajectories (including hand-drawn ones) and outperforms the learning-based baseline SPiD on unseen trajectories; on hardware it reduces RMSE by **44.0%** on the longest task.
- **Efficient multimodal planning.** On obstacle avoidance, BM2PC matches C-MPPI and SVMPC in accuracy while cutting planning time by **29%–66%**.

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/Longyouyuan/BM2PC.git
cd BM2PC
pip install -r requirements.txt   # install a CUDA build of PyTorch for GPU support
```

All simulation experiments run in MuJoCo, with the rope rollouts parallelized on the GPU via NVIDIA Warp.

### 1. Trajectory tracking

```bash
# --traj: sinusoid | egg | lemniscate | spongebob | patrickstar | flower
# --duration: any value in seconds (default 5 s for simple, 15 s for hand-drawn trajectories)
# --visualization / --no-visualization: draw candidate rollouts (default off for simple, on for hand-drawn)
python controller/M2PC_Mujoco_Rope_Tracking.py                                          # PatrickStar, 15 s, vis on
python controller/M2PC_Mujoco_Rope_Tracking.py --traj lemniscate                        # Lemniscate, 5 s, vis off
python controller/M2PC_Mujoco_Rope_Tracking.py --traj flower --duration 18              # Flower, 18 s, vis on
python controller/M2PC_Mujoco_Rope_Tracking.py --traj egg --duration 4 --visualization  # Egg, 4 s, vis on
```

### 2. Obstacle avoidance

```bash
python controller/M2PC_Mujoco_Rope_with_Obstacles.py  # Lemniscate (5 s/cycle) with an obstacle at the crossing, 3 modes
```

### 3. Disturbance rejection

```bash
python controller/M2PC_Mujoco_Rope_Robustness.py  # Lemniscate (5 s/cycle) with 1 N, 50 ms pulses on the tip
```

### Real robot

For deployment on the Franka Emika Panda, see [franka/README.md](franka/README.md).

---

## 📈 Trajectory Tracking (Simulation)

The controller drives the grasped endpoint so that the **rope tip** tracks a target trajectory. Three analytic trajectories (Sinusoid, Egg, Lemniscate) and three hand-drawn ones (SpongeBob, PatrickStar, Flower), each under three execution durations, all with the **same** controller configuration.

### Analytic trajectories

<table>
<tr>
<td align="center" width="33%"><img src="assets/figs/tracking_goal_sinusoid.png" height="230"><br><b>Sinusoid</b> · 5.085 m</td>
<td align="center" width="33%"><img src="assets/figs/tracking_goal_egg.png" height="230"><br><b>Egg</b> · 5.844 m</td>
<td align="center" width="33%"><img src="assets/figs/tracking_goal_lemniscate.png" height="230"><br><b>Lemniscate</b> · 5.540 m</td>
</tr>
<tr>
<td align="center"><img src="assets/gifs/tracking/m2pc_tracking_sin_4s.gif" width="100%"><br><b>Sinusoid · T = 4 s</b></td>
<td align="center"><img src="assets/gifs/tracking/m2pc_tracking_egg_4s.gif" width="100%"><br><b>Egg · T = 4 s</b></td>
<td align="center"><img src="assets/gifs/tracking/m2pc_tracking_eight_4s.gif" width="100%"><br><b>Lemniscate · T = 4 s</b></td>
</tr>
<tr>
<td align="center"><img src="assets/gifs/tracking/m2pc_tracking_sin_5s.gif" width="100%"><br><b>Sinusoid · T = 5 s</b></td>
<td align="center"><img src="assets/gifs/tracking/m2pc_tracking_egg_5s.gif" width="100%"><br><b>Egg · T = 5 s</b></td>
<td align="center"><img src="assets/gifs/tracking/m2pc_tracking_eight_5s.gif" width="100%"><br><b>Lemniscate · T = 5 s</b></td>
</tr>
<tr>
<td align="center"><img src="assets/gifs/tracking/m2pc_tracking_sin_6s.gif" width="100%"><br><b>Sinusoid · T = 6 s</b></td>
<td align="center"><img src="assets/gifs/tracking/m2pc_tracking_egg_6s.gif" width="100%"><br><b>Egg · T = 6 s</b></td>
<td align="center"><img src="assets/gifs/tracking/m2pc_tracking_eight_6s.gif" width="100%"><br><b>Lemniscate · T = 6 s</b></td>
</tr>
</table>

### Hand-drawn trajectories

<table>
<tr>
<td align="center" width="33%"><img src="assets/figs/tracking_goal_spongebob.png" height="230"><br><b>SpongeBob</b> · 18.3 m</td>
<td align="center" width="33%"><img src="assets/figs/tracking_goal_patrickstar.png" height="230"><br><b>PatrickStar</b> · 18.1 m</td>
<td align="center" width="33%"><img src="assets/figs/tracking_goal_flower.png" height="230"><br><b>Flower</b> · 17.5 m</td>
</tr>
<tr>
<td align="center"><img src="assets/gifs/tracking/m2pc_tracking_spongebob_12s.gif" width="100%"><br><b>SpongeBob · T = 12 s</b></td>
<td align="center"><img src="assets/gifs/tracking/m2pc_tracking_PatrickStar_12s.gif" width="100%"><br><b>PatrickStar · T = 12 s</b></td>
<td align="center"><img src="assets/gifs/tracking/m2pc_tracking_flower_12s.gif" width="100%"><br><b>Flower · T = 12 s</b></td>
</tr>
<tr>
<td align="center"><img src="assets/gifs/tracking/m2pc_tracking_spongebob_15s.gif" width="100%"><br><b>SpongeBob · T = 15 s</b></td>
<td align="center"><img src="assets/gifs/tracking/m2pc_tracking_PatrickStar_15s.gif" width="100%"><br><b>PatrickStar · T = 15 s</b></td>
<td align="center"><img src="assets/gifs/tracking/m2pc_tracking_flower_15s.gif" width="100%"><br><b>Flower · T = 15 s</b></td>
</tr>
<tr>
<td align="center"><img src="assets/gifs/tracking/m2pc_tracking_spongebob_18s.gif" width="100%"><br><b>SpongeBob · T = 18 s</b></td>
<td align="center"><img src="assets/gifs/tracking/m2pc_tracking_PatrickStar_18s.gif" width="100%"><br><b>PatrickStar · T = 18 s</b></td>
<td align="center"><img src="assets/gifs/tracking/m2pc_tracking_flower_18s.gif" width="100%"><br><b>Flower · T = 18 s</b></td>
</tr>
</table>

---

## 🚧 Obstacle Avoidance (Simulation)

BM2PC maintains **n = 3** control modes while tracking a Lemniscate. Modes collapse into a single solution far from obstacles and split into distinct left / right / over avoidance behaviors near them, so the controller always has a feasible alternative ready. The 18 scenarios below cover static (**S**), moving (**M**) and mixed static + moving (**SM**) obstacles. **All runs are collision-free.**

<table>
<tr>
<td align="center" width="50%"><img src="assets/gifs/obstacles/bm2pc_01_s1_cross.gif" width="100%"><br><b>S1 · crossing</b><br><sub>clearance 1.68 cm · RMSE 6.65 cm</sub></td>
<td align="center" width="50%"><img src="assets/gifs/obstacles/bm2pc_02_s1_lower_lobe.gif" width="100%"><br><b>S1 · lower lobe</b><br><sub>clearance 2.54 cm · RMSE 5.71 cm</sub></td>
</tr>
<tr>
<td align="center"><img src="assets/gifs/obstacles/bm2pc_03_s1_upper_lobe.gif" width="100%"><br><b>S1 · upper lobe</b><br><sub>clearance 0.65 cm · RMSE 6.26 cm</sub></td>
<td align="center"><img src="assets/gifs/obstacles/bm2pc_04_s2_cross_lobe.gif" width="100%"><br><b>S2 · crossing + lobe</b><br><sub>clearance 1.70 cm · RMSE 8.79 cm</sub></td>
</tr>
<tr>
<td align="center"><img src="assets/gifs/obstacles/bm2pc_05_s2_two_lobes.gif" width="100%"><br><b>S2 · two lobes</b><br><sub>clearance 2.22 cm · RMSE 6.68 cm</sub></td>
<td align="center"><img src="assets/gifs/obstacles/bm2pc_06_s2_lower_pair.gif" width="100%"><br><b>S2 · lower pair</b><br><sub>clearance 1.58 cm · RMSE 8.04 cm</sub></td>
</tr>
<tr>
<td align="center"><img src="assets/gifs/obstacles/bm2pc_07_m1_cross_x.gif" width="100%"><br><b>M1 · crossing, x-oscillation</b><br><sub>clearance 3.01 cm · RMSE 6.32 cm</sub></td>
<td align="center"><img src="assets/gifs/obstacles/bm2pc_08_m1_cross_y.gif" width="100%"><br><b>M1 · crossing, y-oscillation</b><br><sub>clearance 4.71 cm · RMSE 5.03 cm</sub></td>
</tr>
<tr>
<td align="center"><img src="assets/gifs/obstacles/bm2pc_09_m1_sweep_lower.gif" width="100%"><br><b>M1 · lower sweep</b><br><sub>clearance 1.37 cm · RMSE 3.86 cm</sub></td>
<td align="center"><img src="assets/gifs/obstacles/bm2pc_10_m1_circle_cross.gif" width="100%"><br><b>M1 · circling the crossing</b><br><sub>clearance 9.08 cm · RMSE 3.99 cm</sub></td>
</tr>
<tr>
<td align="center"><img src="assets/gifs/obstacles/bm2pc_11_m2_cross_x_lobe_y.gif" width="100%"><br><b>M2 · crossing-x + lobe-y</b><br><sub>clearance 0.81 cm · RMSE 5.90 cm</sub></td>
<td align="center"><img src="assets/gifs/obstacles/bm2pc_12_m2_lobes_y.gif" width="100%"><br><b>M2 · two lobes, anti-phase</b><br><sub>clearance 0.43 cm · RMSE 6.18 cm</sub></td>
</tr>
<tr>
<td align="center"><img src="assets/gifs/obstacles/bm2pc_13_m2_sweep_antiphase.gif" width="100%"><br><b>M2 · anti-phase sweeps</b><br><sub>clearance 1.30 cm · RMSE 5.03 cm</sub></td>
<td align="center"><img src="assets/gifs/obstacles/bm2pc_14_m2_circle_pair.gif" width="100%"><br><b>M2 · circling pair</b><br><sub>clearance 10.84 cm · RMSE 4.05 cm</sub></td>
</tr>
<tr>
<td align="center"><img src="assets/gifs/obstacles/bm2pc_15_sm_cross_static_lobe_y.gif" width="100%"><br><b>SM · static crossing + moving lobe</b><br><sub>clearance 0.93 cm · RMSE 7.89 cm</sub></td>
<td align="center"><img src="assets/gifs/obstacles/bm2pc_16_sm_lobe_static_cross_x.gif" width="100%"><br><b>SM · static lobe + moving crossing</b><br><sub>clearance 2.01 cm · RMSE 5.95 cm</sub></td>
</tr>
<tr>
<td align="center"><img src="assets/gifs/obstacles/bm2pc_17_sm_cross_static_sweep.gif" width="100%"><br><b>SM · static crossing + sweep</b><br><sub>clearance 1.18 cm · RMSE 6.57 cm</sub></td>
<td align="center"><img src="assets/gifs/obstacles/bm2pc_18_sm_cross_static_circle.gif" width="100%"><br><b>SM · static crossing + circling</b><br><sub>clearance 1.19 cm · RMSE 7.15 cm</sub></td>
</tr>
</table>

<sub>Clearance = minimum rope–obstacle distance over the run.</sub>

---

## 💥 Disturbance Rejection (Simulation)

Four force pulses are applied while tracking a Lemniscate (100 g rope). The 18 configurations vary the **application point** (tip, ¼, ½ or ¾ of the rope length measured from the tip), the **direction**, the **magnitude** and the **duration** of the pulses. BM2PC recovers in every case.

<table>
<tr>
<td align="center" width="33%"><img src="assets/gifs/robustness/m2pc_disturbance_01_base_x_1N_50ms_tip.gif" width="100%"><br><b>Tip · +x · 1 N / 50 ms</b><br><sub>peak 68.2 cm · recovery 0.86 s</sub></td>
<td align="center" width="33%"><img src="assets/gifs/robustness/m2pc_disturbance_02_dir_y.gif" width="100%"><br><b>Tip · +y · 1 N / 50 ms</b><br><sub>peak 63.1 cm · recovery 1.16 s</sub></td>
<td align="center" width="33%"><img src="assets/gifs/robustness/m2pc_disturbance_03_dir_xy.gif" width="100%"><br><b>Tip · +x+y · 1 N / 50 ms</b><br><sub>peak 65.6 cm · recovery 0.94 s</sub></td>
</tr>
<tr>
<td align="center"><img src="assets/gifs/robustness/m2pc_disturbance_04_dir_x-y.gif" width="100%"><br><b>Tip · +x−y · 1 N / 50 ms</b><br><sub>peak 66.9 cm · recovery 1.18 s</sub></td>
<td align="center"><img src="assets/gifs/robustness/m2pc_disturbance_05_dir_xyz.gif" width="100%"><br><b>Tip · +x+y+z · 1 N / 50 ms</b><br><sub>peak 61.8 cm · recovery 1.19 s</sub></td>
<td align="center"><img src="assets/gifs/robustness/m2pc_disturbance_06_node_quarter.gif" width="100%"><br><b>¼ length · +x · 1 N / 50 ms</b><br><sub>peak 35.6 cm · recovery 1.14 s</sub></td>
</tr>
<tr>
<td align="center"><img src="assets/gifs/robustness/m2pc_disturbance_07_node_three_quarter.gif" width="100%"><br><b>¾ length · +x · 1 N / 50 ms</b><br><sub>peak 19.6 cm · recovery 1.41 s</sub></td>
<td align="center"><img src="assets/gifs/robustness/m2pc_disturbance_08_tip_0.5N.gif" width="100%"><br><b>Tip · +x · 0.5 N / 50 ms</b><br><sub>peak 38.0 cm · recovery 0.99 s</sub></td>
<td align="center"><img src="assets/gifs/robustness/m2pc_disturbance_09_tip_1.5N_25ms.gif" width="100%"><br><b>Tip · +x · 1.5 N / 25 ms</b><br><sub>peak 57.3 cm · recovery 0.85 s</sub></td>
</tr>
<tr>
<td align="center"><img src="assets/gifs/robustness/m2pc_disturbance_10_tip_25ms.gif" width="100%"><br><b>Tip · +x · 1 N / 25 ms</b><br><sub>peak 41.3 cm · recovery 0.93 s</sub></td>
<td align="center"><img src="assets/gifs/robustness/m2pc_disturbance_11_tip_0.5N_100ms.gif" width="100%"><br><b>Tip · +x · 0.5 N / 100 ms</b><br><sub>peak 58.3 cm · recovery 1.12 s</sub></td>
<td align="center"><img src="assets/gifs/robustness/m2pc_disturbance_12_tip_0.25N_200ms.gif" width="100%"><br><b>Tip · +x · 0.25 N / 200 ms</b><br><sub>peak 46.4 cm · recovery 0.65 s</sub></td>
</tr>
<tr>
<td align="center"><img src="assets/gifs/robustness/m2pc_disturbance_13_middle_1.5N_50ms.gif" width="100%"><br><b>½ length · +x · 1.5 N / 50 ms</b><br><sub>peak 37.4 cm · recovery 1.40 s</sub></td>
<td align="center"><img src="assets/gifs/robustness/m2pc_disturbance_14_middle_1N_100ms.gif" width="100%"><br><b>½ length · +x · 1 N / 100 ms</b><br><sub>peak 44.0 cm · recovery 1.63 s</sub></td>
<td align="center"><img src="assets/gifs/robustness/m2pc_disturbance_15_three_quarter_y_1.5N_50ms.gif" width="100%"><br><b>¾ length · +y · 1.5 N / 50 ms</b><br><sub>peak 22.9 cm · recovery 1.40 s</sub></td>
</tr>
<tr>
<td align="center"><img src="assets/gifs/robustness/m2pc_disturbance_16_three_quarter_1N_75ms.gif" width="100%"><br><b>¾ length · +x · 1 N / 75 ms</b><br><sub>peak 22.9 cm · recovery 1.68 s</sub></td>
<td align="center"><img src="assets/gifs/robustness/m2pc_disturbance_17_three_quarter_y_1N_100ms.gif" width="100%"><br><b>¾ length · +y · 1 N / 100 ms</b><br><sub>peak 25.3 cm · recovery 1.67 s</sub></td>
<td align="center"><img src="assets/gifs/robustness/m2pc_disturbance_18_three_quarter_xyz_2N_25ms.gif" width="100%"><br><b>¾ length · +x+y+z · 2 N / 25 ms</b><br><sub>peak 19.0 cm · recovery 1.46 s</sub></td>
</tr>
</table>

<sub>Peak = mean maximum rope-tip deviation over the four pulses; recovery = mean time until the tracking error returns below the recovery threshold. Undisturbed RMSE: 2.67 cm.</sub>

---

## 🔀 Ablation: Sudden Obstacle

A tall pillar leaves two routes around it. A second pillar suddenly appears on the side the controller is heading to, **τ = 0.75 s** before the crossing. BM2PC keeps an alternative mode on the free side and switches to it; the ablated variants have no such mode ready.

<table>
<tr><td align="center"><img src="assets/gifs/ablation/scenario_b_full.gif" width="50%"><br><b>BM2PC (n = 3, β = 600)</b> · switches mode and passes ✅</td></tr>
<tr><td align="center"><img src="assets/gifs/ablation/scenario_b_n1.gif" width="50%"><br><b>Single mode (n = 1)</b> · no alternative mode</td></tr>
<tr><td align="center"><img src="assets/gifs/ablation/scenario_b_beta0.gif" width="50%"><br><b>No diversity (β = 0)</b> · modes stay similar</td></tr>
</table>

| Metric (τ ∈ [0.65, 0.90] s, 10 seeds) | BM2PC | n = 1 | β = 0 |
|:--|:--:|:--:|:--:|
| Success rate (%) | **64.3** | 45.2 | 47.0 |
| Collision rate (%) | **25.2** | 39.3 | 35.7 |
| Mean max. tracking error (cm) | **56.2** | 80.2 | 76.6 |

---

<!--
## 🤖 Real-World Results

On a Franka Emika Panda with OptiTrack rope tracking, BM2PC runs at 66.67 Hz with the execution dynamics model in the loop.

| Real-world tracking RMSE (cm) | Sinusoid | Lemniscate ×1 | Lemniscate ×3 |
|:--|:--:|:--:|:--:|
| SPiD | 4.300 | 3.451 | 4.414 |
| **BM2PC** | **4.175** | **2.701** | **2.474** |
| BM2PC w/o execution dynamics model | – | – | 23.68 |

In real-world obstacle avoidance with a movable obstacle, all trials were collision-free (RMSE 4.51 ± 1.11 cm), despite ~5% of the motion-capture measurements being corrupted by occlusions. See the overview video above.

---
-->

## 📝 Citation

```bibtex
@article{long2026bm2pc,
  title   = {Multimodal MPC via Bayesian Set Optimization for Dynamic Manipulation of Deformable Linear Objects},
  author  = {Long, Youyuan and Solak, Gokhan and Ajoudani, Arash},
  year    = {2026},
  note    = {Submitted to IEEE}
}
```

## 🙏 Acknowledgements

This work was supported by the Horizon Europe Project TORNADO (GA 101189557).
