# TALOS NIO — Neural-Inertial Odometry Pipeline
### Open-Source 6-DOF Spatial Tracking for Mixed Reality Hardware
**Ground Truth State: March 10, 2026**

---

## 1. Mission

Standard IMUs are mathematically cursed. Integrating acceleration twice yields position — but microscopic thermal and mechanical errors in the silicon compound quadratically over time. Within seconds, unguided inertial tracking pulls the virtual floor into your face.

TALOS solves this not by fighting the noise in open space, but by locking it inside a biomechanical cage. The system doesn't need to be perfect. It needs to be **bounded**.

The drift is not defeated. It is caged.

---

## 2. Platform Context

TALOS NIO is the tracking subsystem of the **Unit TALOS Rev. 1** — a fully open-source Mixed Reality HMD built under the **GLR Project**.

| Component | Specification |
|---|---|
| SoC | FriendlyElec NanoPC-T6 (Rockchip RK3588) |
| Display | Dual AUO AMOLED panels, 90Hz |
| Power | 21700 Li-ion cells, rear battery pack |
| OpenXR Runtime | Monado |
| Graphics Driver | PanVK (Vulkan) |
| OS | Armbian |

TALOS NIO specifically targets the RK3588 NPU for Overlord inference with **zero CPU fallback**. Every design decision in the inference pipeline traces back to this constraint.

---

## 3. TALOS NIO vs TLIO — What Is The Same, What Is Not

TALOS NIO independently converged on the same core neural-inertial fusion architecture as **TLIO** (Tightly-coupled Learned Inertial Odometry, Facebook Research, 2020) — a fast IMU mechanization loop corrected at low frequency by a learned displacement estimator, fused via ESKF. This convergence validates the architecture. It is not coincidence; it is the correct solution to the problem.

**Where TALOS and TLIO share the same foundation:**

| Component | TLIO | TALOS NIO |
|---|---|---|
| Fast loop | IMU mechanization (ESKF predict) | IMU mechanization (ESKF predict) |
| Slow loop | Learned displacement correction | Overlord (SpectralMLP) correction |
| Fusion | ESKF update step | ESKF update step |
| Labels | Ego-centric displacement | Ego-centric displacement |
| Gate | Innovation gating | Mahalanobis Slap Gate |

**Where TALOS goes beyond TLIO:**

TLIO is a pure odometry system. It stops at fusion. TALOS treats the fusion as the floor, not the ceiling.

| Capability | TLIO | TALOS NIO |
|---|---|---|
| Biomechanical cage | No | Yes — floating sphere + rotational limits |
| Dual-IMU veto | No | Yes — LAID bouncer, IMU2 never integrates |
| Neck Pivot Point | No | Yes — dynamic CoR, NPPTracker EMA + Z-lock |
| Yaw anchoring (static) | No | Yes — ZARU compound standstill update |
| Yaw anchoring (dynamic) | No | Yes — LAID differential pseudo-measurement |
| NPU deployment target | No | Yes — RK3588 INT8, zero CPU fallback |
| Training domain | Foot/pocket IMU | Head-mounted Aria (same form factor as TALOS) |
| Hardware constraint | Generic IMU | Designed around TALOS Rev. 1 biomechanics |

TLIO proved the core loop works. TALOS builds the cage around it.

---

## 4. Hardware Topology

```
                    ┌─────────────────────────────┐
                    │        TALOS Rev. 1          │
                    │                              │
   [IMU1] ●────────── Optics Faceplate (front)    │
   [IMU2] ●────────── Battery Pack    (rear)      │
                    │         │                    │
                    │    Lever Arm                 │
                    │    ~15cm Y separation        │
                    │    Same Z axis               │
                    └─────────────────────────────┘
```

### IMU1 — Front (Optics Faceplate)
The **leading sensor**. All position estimates originate from IMU1. Overlord watches IMU1 exclusively. IMU1 is the only sensor that touches the neural pipeline.

### IMU2 — Back (Battery Pack)
The **bouncer**. No AI. No Overlord. Pure classical rigid body mechanics on the Cortex-A76 CPU. IMU2's noise characteristics are irrelevant — it is never integrated over time. It exists to provide the differential signal that enables LAID, dynamic neck pivot approximation, IK, and yaw anchoring during movement.

IMU2 has no state, no history, no model access. It reads one sample, runs a deterministic equation, and either lets the output through or pulls it back to the cage wall.

**IMU1 asked the question. IMU2 confirms or denies. IMU2 never asks anything.**

### The Lever Arm
The 21700 battery cells on the rear are not just power — they are the physical anchor for IMU2. The counterbalance that makes the headset wearable for hours is simultaneously the geometric primitive that makes the entire NIO architecture possible. A longer lever arm produces a stronger differential signal, a more accurate pivot approximation, and a stronger kinematic yaw anchor during rotation.

One hardware decision. Two problems solved.

---

## 5. Dual-Loop Architecture

```
                                          ┌─────────────────────────┐
IMU1 raw ──→ ESKF predict ──────────────→ │   ESKF Fast Loop        │──→ position
                 ↑                        │   15-state, SO(3)       │
                 │                        │   100Hz                 │
         Overlord correction              └─────────────────────────┘
         (subject to Slap Gate)
                 ↑
         SpectralMLP inference
         (RK3588 NPU, frozen weights)
                 ↑
         198-feature FFT vector
         (computed inline, log1p scaled)
                 ↑
IMU1 raw ──→ nymeria_loader.py + incremental_train.py (offline + online)


IMU2 raw ──→ LAID differential math ──→ NPP (CoR solve) ──→ NPPTracker ──→ IK ──→ cage walls
                    │                         ↓
               ZARU detector            Cage center (floating sphere anchor)
                    │
               yaw bias correction (stationary)
```

The fast loop never waits for Overlord. If Overlord is mid-inference, slapped, or unavailable — the ESKF already pushed the frame. Motion-to-photon latency is owned entirely by the fast loop.

---

## 6. Training Data — Nymeria Dataset

All training data comes from **Project Nymeria** (Meta Reality Labs, 2023) — a large-scale egocentric motion dataset captured using Meta Project Aria glasses.

| Property | Value |
|---|---|
| Subjects | ~300 |
| Duration | ~300 hours |
| IMU rate | 800Hz native → resampled to 100Hz |
| Ground truth | MPS closed-loop trajectory (gravity-aligned world frame) |
| Form factor | Head-mounted — same deployment domain as TALOS |
| IMU streams | `imu-right (1202-1)` → IMU1, `imu-left (1202-2)` → IMU2 reference |

**Why Nymeria over all alternatives:**
Every prior training attempt used mismatched hardware — an iPhone strapped to a LiDAR rig (IDOL), or chest/pocket-mounted IMUs (TLIO). Nymeria is head-mounted Aria glasses. The training domain is the deployment domain. No accidental domain mismatch. No domain adaptation required.

### Nymeria Loader — nymeria_loader.py

Per-sequence processing pipeline:

**Step 1 — VRS Read**
Both IMU streams are read from `motion.vrs` via `projectaria_tools`. `imu-right` feeds Overlord. `imu-left` is read for alignment but discarded — it has no role in the training pipeline.

**Step 2 — Resample to 100Hz**
Both streams are resampled to a shared uniform 100Hz grid via linear interpolation. Nymeria native IMU rate is ~800Hz with jitter. Resampling gives clean 10ms spacing.

**Step 3 — GT Interpolation**
MPS closed-loop trajectory keyframes (~10Hz) are interpolated onto the 100Hz IMU grid. Positions via linear interpolation, orientations via SLERP.

**Step 4 — Ego-Centric Labels**
Labels are **local body-frame displacement and rotational delta**, not absolute world pose:
```python
# Translation: global displacement rotated into body frame at window start
delta_p_global = pos[end] - pos[start]
delta_p_local  = R_start.inv().apply(delta_p_global)

# Rotation: relative quaternion delta in body frame
R_delta      = R_start.inv() * R_end
q_delta_wxyz = [W, X, Y, Z]  # converted from scipy [X,Y,Z,W]
```
A stateless model cannot predict absolute world coordinates — it has no memory of where the session started. Ego-centric labels make the problem stateless-solvable.

**Step 5 — Windowing**
- Window size: 64 samples (640ms at 100Hz)
- Stride: 10 samples
- Output shape per window: `imu1_features (64, 6)`, `trans (3,)`, `quat (4,)`

### Output Format
```
dict keys: imu1_features, trans, quat
           (no imu2 — IMU2 is a runtime veto system, not a training signal)
```

---

## 7. Training Pipeline — incremental_train.py

Training is **incremental** — sequences are added one at a time, the model trains on the growing pool, and a physical ESKF evaluation gates whether training is helping or hurting.

### Dual Early Stopping

Two independent kill switches run in parallel:

**Physical veto (ESKF ATE):**
After each round, the trained model is run through the full ESKF pipeline on the Shelby Arroyo validation sequence. If mean ATE degrades for `PATIENCE=8` consecutive rounds, training halts. This catches physical overfitting that neural loss cannot see.

**Neural stagnation:**
If training loss fails to improve by `LOSS_MIN_DELTA=1e-4` for `LOSS_PATIENCE=15` consecutive rounds, training halts. This catches dead models that are accumulating data without learning.

### Warmup Gate
ESKF evaluation is expensive and meaningless on an untrained model. Evaluation is skipped until training loss drops below `WARMUP_LOSS_THRESHOLD=0.005`. The ESKF patience counter does not burn during warmup. The model trains freely until it is actually ready to help the filter.

### Training Configuration
```
Val subject:    Shelby Arroyo — held out entirely, never seen during training
Batch size:     4096
Optimizer:      AdamW  lr=1e-3  weight_decay=1e-2
Scheduler:      OneCycleLR  max_lr=5e-3
Regularization: SpectralDropout(0.4) at input + BatchNorm at every layer
Hardware:       RTX 5070 Ti 16GB
```

### FFT Pipeline (inline, per round)
```python
fft_complex = np.fft.rfft(imu_windows, axis=1)   # (N, 64, 6) → (N, 33, 6) complex
fft_log     = np.log1p(np.abs(fft_complex))        # magnitude + log compression
fft_flat    = fft_log.reshape(N, 198)              # 33 bins × 6 channels = 198 features
```

The model receives `(batch, 198)` float32. No complex numbers, no FFT ops in the inference graph. The RK3588 NPU sees only Linear layers and BatchNorm. Zero CPU fallback.

---

## 8. Overlord — SpectralMLP

Overlord is the neural drift suppressor. It is **not a navigation system**. It predicts ego-centric displacement to correct IMU1's accumulated bias before it enters the ESKF.

### Architecture
```
Input:  (batch, 198)   — precomputed spectral features, log1p scaled

SpectralDropout(0.4)  — blinds network to raw frequency bins at input
Linear(198 → 256) → BatchNorm1d → ReLU
Linear(256 → 128) → BatchNorm1d → ReLU
Linear(128 →  64) → BatchNorm1d → ReLU

head_trans: Linear(64 → 3)   — ego-centric XYZ displacement
head_quat:  Linear(64 → 4)   → normalize → unit quaternion
```

**ReLU, not GELU.** RKNN INT8 NPU does not support GELU. This is a hard deployment constraint.

The model is defined once in `SMLP.py` — single source of truth for architecture, normalization, and dropout behavior across training and inference.

### Why Stateless
Overlord has no recurrent connections, no attention over history, no hidden state. Each 640ms window is evaluated independently. This is a deliberate constraint for RK3588 NPU deployment — stateful models require sequential execution that breaks the NPU's parallelism. Stateless inference means every window is an independent matrix multiply. The NPU runs at full utilization.

### Why Spectral
Human head motion has consistent biomechanical frequency signatures. Walking vibration, saccadic eye movement compensation, and postural sway occupy distinct frequency bands. The FFT exposes these bands explicitly. The MLP learns which frequency signatures correspond to which motion patterns — a fundamentally more generalizable representation than raw time-domain acceleration values.

### Deployment
Overlord runs on the RK3588 NPU. Weights are **frozen at inference time**. No learning, no adaptation, no backpropagation at runtime. The Slap Gate decides whether each prediction is injected into the ESKF or discarded.

---

## 9. LAID — Lever Arm Inertial Disambiguation

### The Problem
A single IMU cannot distinguish between:
- Head rotating left
- Entire body translating left

Both produce identical lateral acceleration on IMU1. This is a fundamental physical limitation — not a noise problem, not a calibration problem. Mathematically unsolvable with one sensor.

### The Solution
Two IMUs on opposite ends of a physical lever disambiguate rotation from translation through differential math. No AI required. No training data required.

### Trust Weighting

| IMU1 vs IMU2 | Interpretation | Action |
|---|---|---|
| Opposite vectors | Rotation — seesaw happening | High confidence → IK applied |
| Identical vectors | Translation — both sensors agree | High confidence → linear motion |
| Partial divergence | Mixed motion | Proportional weighting |

The math self-weights. No artificial decomposition. No "which started first" logic. Order of motion is preserved through sequential ESKF state integration.

### Dynamic Neck Pivot Point (NPP)
The Instantaneous Centre of Rotation is computed each frame:

```
Rigid body identity:  v(p) = v_ref + ω × (p − r_ref)
Set v(NPP) = 0:       ω × Δr = −v_imu1
Minimum-norm solve:   Δr = (ω × v_imu1) / |ω|²
```

Result clamped to anatomical limits: ±8cm lateral, ±20cm depth, ±10cm height.
When `|ω| < threshold` (head nearly still), falls back to anatomical prior.

### NPPTracker — EMA Z-Lock
Raw ICR jitters frame-to-frame due to sensor noise and degenerate geometry at low rotation speeds. `NPPTracker` applies exponential smoothing with separate handling for the vertical axis:

- **X, Y**: EMA with omega-weighted updates (`alpha=0.1`). At low `|ω|` the ICR is unreliable — updates are damped automatically.
- **Z (vertical)**: Locked via deadband (`z_deadband=3cm`). The neck pivot height doesn't bounce during normal head rotation — it only changes on postural shifts (standing ↔ sitting). Z updates slowly (`z_alpha=0.02`) and only when the running mean shifts beyond the deadband.

NPP is not just an IK input. It is the **cage center**. The entire biomechanical constraint system floats on the NPP. Without it, the cage has no anchor and drift is unbounded regardless of Overlord's accuracy.

### LAID Yaw Anchor
Pitch and roll are gravity-anchored. Yaw is not — it is the hardest axis.

During dynamic movement, a pure yaw rotation produces opposite tangential velocity signatures on IMU1 and IMU2. The tangential velocity delta `v_t = ω × r` between the two sensors generates a pseudo-measurement that pulls yaw back into alignment. The lever arm became a yaw sensor.

### Bouncer Logic
IMU2 does not fuse. It does not navigate. It vetoes:

```python
# IMU2 never sees the model. It only answers: is this physically possible?
if laid_trust < threshold:
    v_final = trust * v_mlp + (1 - trust) * v_cage   # pull toward cage wall
    R_obs   = eye(3) * (0.1 / (trust + 1e-6))        # inflate measurement noise
else:
    v_final = v_mlp   # pass through untouched
```

### Fault Handling
```
IMU2 differential broken → reset IMU2
Still broken             → halt IMU2, notify user
Fast loop continues on IMU1 + Overlord alone
Biomechanical cage tightens
ZARU assumes yaw anchoring responsibility
LAID yaw anchor suspended until IMU2 restored
```

---

## 10. Biomechanical Cage

The cage enforces the physical reality that the headset is attached to a human skeleton. It operates in two orthogonal constraint spaces simultaneously — angles and meters do not conflict.

### Rotational Limits (Angular Space)
| Axis | Limit |
|---|---|
| Yaw | ±70° |
| Pitch | ±45° |
| Roll | ±20° |

### Translational Limit (Metric Space)
Floating sphere of ~12cm radius centered on the dynamically computed NPP. The head cannot translate beyond this sphere relative to the torso.

### The Floating Cage
The cage is not fixed in room coordinates. As the user walks, the NPP moves with the body. The sphere floats with it:

```
Cage center = IMU1 position − (neck_radius × current_orientation_vector)
```

Drift cannot accumulate by walking. The cage walks with you.

---

## 11. The Slap — Mahalanobis Innovation Gating

When Overlord produces a prediction that is statistically inconsistent with the current ESKF state, it is **ignored this frame**. Not punished. Not retrained. The gate closes silently.

### Gate Implementation
The Slap is implemented directly inside `update_velocity()` — the gate is not a separate module, it is a precondition on the Kalman update itself:
```python
S        = H @ P @ H.T + R_obs         # innovation covariance
residual = predicted_velocity - velocity
S_inv    = inv(S)
mahal_sq = residual @ S_inv @ residual  # squared Mahalanobis distance

if mahal_sq > threshold²:
    return  # Slap — update silently rejected, ESKF continues on kinematics

K       = P @ H.T @ S_inv              # reuse S_inv from gate check
delta_x = K @ residual
# ... apply state correction
```
Default threshold: 3.0 (squared: 9.0). The `S_inv` computed for the Mahalanobis check is reused for the Kalman gain — zero redundant computation.

### Organic Covariance Inflation
No cooldown timer. No hard ban. When Overlord is slapped, `P` inflates naturally during each predict step. Repeated slaps → `P` grows → `S` grows → Mahalanobis threshold self-loosens. The gate widens organically to accept valid high-energy movements.

Maximum lag from a single slap: one correction cycle at 100Hz = **10ms**. The suspension absorbs the pothole. The fast loop never notices.

---

## 12. ZARU — Zero Angular Rate Update

Yaw is the final boss. Gravity anchors pitch and roll. Yaw has no gravitational reference. Without a magnetometer or camera, the ESKF yaw state wanders freely under gyro bias integration. The user sits still while the virtual world slowly rotates.

### Detection
When gyroscope variance across a rolling window drops below the micro-tremor threshold, the user is classified as stationary. The gyro reading at that moment is **pure bias** — no motion to integrate.

```
ZARU_WINDOW    = 50     # samples (~0.5s at 100Hz)
ZARU_THRESHOLD = 1e-4   # gyro variance (rad/s)² — below = stationary
ZARU_R_DIAG    = 1e-4   # measurement noise for angular rate pseudo-observation
```

### Injection
`H` observes orientation error (indices 6:9). The Kalman gain is a full 15×3 matrix — because `predict()` builds cross-correlation in `P[9:12, 6:9]` through `F[6:9, 9:12] = -R*dt`, the gain automatically pushes correction into gyro bias (9:12). The bias is corrected through **covariance coupling**, not through direct observation.

```python
H = np.zeros((3, 15))
H[0:3, 6:9] = np.eye(3)        # observe orientation error states
z_residual = -(gyro_raw - bg)   # expected = 0, measured = gyro - bg
```

### Compound Standstill Update
ZARU detection is a state classification: "user is stationary." A stationary user has both zero angular rate AND zero velocity. When ZARU fires, it injects both:
1. `update_zaru()` — angular rate pseudo-observation → gyro bias correction
2. `update_velocity(zero)` — zero-velocity anchor → prevents gravity-leakage divergence

Fires once per `ZARU_WINDOW` (every 0.5s) to prevent covariance collapse from over-updating.

### Yaw Coverage Matrix

| Condition | Anchor | Hardware Required |
|---|---|---|
| Stationary | ZARU | IMU1 only — available now |
| Dynamic movement | LAID yaw differential | Dual IMU hardware |
| IMU2 fault | ZARU only | Cage tightens automatically |

---

## 13. ESKF — 15-State Error State Kalman Filter on SO(3)

### State Vector
| Index | States | Description |
|---|---|---|
| 0:3 | Position | XYZ meters |
| 3:6 | Velocity | XYZ m/s |
| 6:9 | Orientation error | so(3) tangent space |
| 9:12 | Gyroscope bias | rad/s |
| 12:15 | Accelerometer bias | m/s² |

### SO(3) — Not Euler
Orientation nominal state is a 3×3 rotation matrix on the SO(3) manifold. Euler angles have gimbal lock at ±90° pitch — a real failure mode when the user looks straight up. SO(3) has no singularities.

Error state lives in `so(3)` tangent space. Corrections applied via exponential map:
```python
# Correct — Rodrigues formula, stays on SO(3) manifold
self.orientation = self.orientation @ Rotation.from_rotvec(delta_x[6:9]).as_matrix()

# Wrong — additive Euler, fails at singularities
self.orientation += delta_x[6:9]   # DO NOT DO THIS
```

Mahalanobis distance for the Slap Gate is computed in position space ℝ³ — avoiding the problem of computing statistical distances on curved rotational manifolds.

### Covariance Propagation
```python
F[6:9, 9:12] = -R * dt   # correct: projects gyro bias body→world
# NOT:
F[6:9, 9:12] = -I * dt   # wrong: only valid when R = identity (no rotation)
```

---

## 14. Current Results

| System | Mean ATE | Final ATE | Dataset |
|---|---|---|---|
| Pure IMU | 45.94m | 169.46m | TUM-VI Room1, 60s |
| TALOS v3 | 3.55m | 3.32m | TUM-VI Room1, 60s |
| TALOS (ego-centric labels) | **1.1m** | — | IDOL building1 |
| TALOS (Nymeria) | training | — | Shelby Arroyo, 1207s |

Nymeria training is active. Results pending.

---

## 15. Active Technical Debt

### ~~Debt 1 — Quaternion Normalization Mismatch~~ ✅ RESOLVED
Single-source `SMLP.py` — both training and inference use the same `forward()`.

### ~~Debt 2 — input_dim Mismatch~~ ✅ RESOLVED
`input_dim=198` canonical. Dead 306/612 values purged.

### ~~Debt 3 — ZARU~~ ✅ RESOLVED
Rolling gyro variance detector + compound standstill update. Yaw bounded to <0.35° over 5s stationary with 0.003 rad/s bias.

### ~~Debt 4 — IDOL Dataset~~ ✅ RESOLVED / REPLACED
IDOL pipeline retired entirely. Replaced by Nymeria — head-mounted, same domain as TALOS hardware.

### ~~Debt 5 — ESKF State Corrections Severed~~ ✅ RESOLVED
All 15 state corrections restored. Slap Gate active. Orientation, gyro bias, accel bias corrections operational.

### ~~Debt 6 — Initial Velocity Frame Error~~ ✅ RESOLVED
MPS provides velocity in device frame. Rotated to world frame on ESKF initialization via `init_vel_world = R @ init_vel_device`.

---

## 16. Roadmap

- [x] SpectralMLP (Overlord) — stateless spectral drift suppression
- [x] ESKF — 15-state, SO(3) orientation, correct covariance propagation
- [x] Ego-centric displacement + rotation labels — stateless-compatible supervision
- [x] Mahalanobis innovation gating (The Slap)
- [x] ZARU — stationary yaw bias correction
- [x] NPPTracker — EMA-smoothed NPP with Z-lock
- [x] Nymeria dataset integration — head-mounted Aria, 300 subjects, 300hrs
- [x] Incremental training loop — ESKF physical veto + loss stagnation dual early stopping
- [x] Warmup gate — ESKF patience protected during model warm-up
- [ ] Dual IMU hardware build
- [ ] LAID full implementation — live bouncer + yaw anchor + dynamic NPP
- [ ] Biomechanical cage enforcement in ESKF state rejection
- [ ] MuJoCo synthetic data pipeline (RTX 5070 Ti)
- [ ] RK3588 NPU deployment — RKNN export, INT8 quantization
- [ ] Visual-Neural-Inertial Odometry layer

---

## 17. Repository Structure

```
TALOS/
├── SMLP.py                   — single-source SpectralMLP (train + inference)
├── nymeria_loader.py          — Nymeria VRS → windowed IMU + ego-centric labels
├── incremental_train.py       — incremental training loop + ESKF evaluation
├── eval_rte.py                — publication metrics (ATE + RTE)
├── run_nymeria_pipeline.py    — full pipeline orchestration
├── scan_dataset.py            — trajectory survey tool
├── test_zaru.py               — ZARU smoke test
├── nymeria/                   — downloaded Nymeria sequences
├── golden/                    — checkpoints + evaluation plots
│   ├── talos.pth              — latest checkpoint
│   ├── talos_best_physical.pth — best ESKF ATE checkpoint
│   ├── eskf_eval_round_N.png  — per-round trajectory plots
│   └── master_telemetry.png   — ATE + loss dashboard
├── pictures/                  — legacy TUM-VI benchmark results
├── TALOS.md                   — this document
└── README.md
```

---

## 18. Target Hardware


| Component | Specification |
|---|---|
| Inference SoC | Rockchip RK3588 (NanoPC-T6) |
| NPU | 6 TOPS INT8, zero CPU fallback target |
| Training GPU | NVIDIA RTX 5070 Ti 16GB |
| IMU1 | Front faceplate (optics position) |
| IMU2 | Rear battery pack (same Z axis, ~15cm Y separation) |

---

*TALOS NIO is part of the GLR Project. Open source. Built for one. Shared for all.*

*The drift is not defeated. It is caged.*
