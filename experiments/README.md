# Experiments

This directory contains scripts for generating predictions, evaluating model performance, and analysing specific aspects of the **EgoForce** hand/arm pose estimation system.

## Pipeline Overview

```
save_predictions.py ──────────────────────► evaluate_predictions.py
        │                                          │
        │                                   evaluate_hand_scale.py
        │                                          │
        │                                   hand_joint_occlusion_graph.py
        │
save_noisy_intrinsic_predictions.py ──► evaluate_noisy_intrinsic_predictions.py
```

The workflow follows a **generate → evaluate → analyse** pattern:

| Stage | Script(s) | Output directory |
|-------|-----------|-----------------|
| **Prediction** | `save_predictions.py`, `save_noisy_intrinsic_predictions.py` | `_DATA/predictions/`, `_DATA/noisy_predictions/` |
| **Evaluation** | `evaluate_predictions.py`, `evaluate_noisy_intrinsic_predictions.py` | `results/OURS/`, `results/intrinsics_robustness/` |
| **Analysis** | `evaluate_hand_scale.py`, `hand_joint_occlusion_graph.py` | `results/hand_scale_eval/`, `results/hand_joint_occlusion_graph/` |

---

## Scripts

### `save_predictions.py` — Main Prediction Generation

Runs the trained model on a dataset, projects predictions to camera space, computes 2D joints, and saves all outputs to a pickle file.

**Supported datasets:** ARCTIC, H2O, HO3D, HOT3D (+ HOT3D_PER, HOT3D_PINHOLE, HOT3D_EQUISOLID, HOT3D_EQUIRECTANGULAR, HOT3D_STEREOGRAPHIC).

**Output format:** `_DATA/predictions/<dataset>_<suffix>_predictions.pkl` — a dict keyed by samplekey, each containing left/right → hand/arm → `{visible, gt_vertices, gt_j3d, gt_j2d, pred_vertices, pred_j3d, pred_j2d}`.

<details>
<summary><strong>CLI Arguments</strong></summary>

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--test-dataset-name` | str | config | Dataset name (`ARCTIC`, `H2O`, `HO3D`, `HOT3D`, …) |
| `--checkpoint-path` | str | config | Path to model weights |
| `--hot3d-conversion` | str | `auto` | HOT3D camera mode (`auto`/`none`/`pinhole`/`equisolid`/`equirectangular`/`stereographic`) |
| `--no-undistort-inp` | flag | — | Use `undistort_inp=False` variant |
| `--no-cit` | flag | — | Disable CIT module (appends `_no_cit` to suffix) |
| `--no-arm-prior` | flag | — | Disable arm prior (appends `_no_arm_prior`) |
| `--no-arm-input` | flag | — | Disable arm input (appends `_no_arm_input`) |
| `--anycalib-624` | flag | — | Use AnyCalib FishEye624 wrapper |
| `--anycalib-pin` | flag | — | Use AnyCalib Pinhole wrapper |
| `--depth-model` | flag | — | Enable depth refinement model |
| `--dgp-model` | flag | — | Enable DGP refinement model |
| `--batch-size` | int | `32` | Batch size |
| `--num-workers` | int | `8` | DataLoader workers |
| `--prefetch-factor` | int | `4` | DataLoader prefetch factor |
| `--persistent-workers` | flag | — | Enable persistent DataLoader workers |

</details>

**Example:**

```bash
python experiments/save_predictions.py \
    --test-dataset-name HOT3D \
    --checkpoint-path checkpoints/model.ckpt \
    --batch-size 64
```

---

### `save_noisy_intrinsic_predictions.py` — Noisy Camera Intrinsic Predictions

Generates predictions under synthetically noisy camera intrinsics for robustness testing. Interpolates between ground-truth HOT3D camera parameters and AnyCalib-inferred parameters at multiple noise levels, and performs camera ray error analysis (angle/depth error vs radial position).

**Noise levels:** 0.5, 1, 2.5, 5, 7.5, 10, 25, 50, 75, 100, 150, 200 (percent).

**Key component:** `InterpolatedNoisyAnyCalib624Dataset` — wraps the HOT3D dataset and interpolates OVR624 parameters as `(1 − α) * GT + α * AnyCalib` where `α = noise_percent / 100`.

<details>
<summary><strong>CLI Arguments</strong></summary>

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--no-cit` | flag | — | Disable CIT module (appends `_no_cit` to suffix) |
| `--ray-grid-size` | int | `21` | Ray sampling grid side length for error analysis |
| `--radial-bins` | int | `10` | Number of radial histogram bins |
| `--force-recompute` | flag | — | Recompute even if cached results exist |
| `--noisy-predictions-dir` | str | `_DATA/noisy_predictions` | Output directory for noisy predictions |

</details>

**Output artifacts (per noise level):**

| File | Description |
|------|-------------|
| `HOT3D_<suffix>_predictions.pkl` | Model predictions under noisy intrinsics |
| `HOT3D_<suffix>_camera_noise_analysis.pkl` | Per-sample camera ray error statistics |
| `*_angle_vs_radius.png` | Angular error vs radial position plot |
| `*_depth_error.png` | Depth error distribution plot |

**Output artifacts (sweep-level):**

| File | Description |
|------|-------------|
| `HOT3D_anycalib_first_frame_intrinsics.json` | Inferred AnyCalib camera parameters |
| `*_angular_vs_noise.png` | Angular error across noise levels |
| `*_depth_error_vs_noise.png` | Depth error across noise levels |

**Example:**

```bash
python experiments/save_noisy_intrinsic_predictions.py --force-recompute
python experiments/save_noisy_intrinsic_predictions.py --no-cit
```

---

### `evaluate_predictions.py` — Main Evaluation

Loads saved predictions, applies optional Kalman temporal filtering, evaluates against ground truth, and outputs per-side and aggregate metrics.

**Metrics computed:** CS_MPJPE, RR_MPJPE, PA_MPJPE, ACC_ERROR (acceleration error), with visibility-conditional splits (occluded/visible).

<details>
<summary><strong>CLI Arguments</strong></summary>

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--test-dataset-name` | str | config | Dataset to evaluate |
| `--log-path` | str | — | Optional subdirectory under `--log-dir` |
| `--log-dir` | str | `_DATA/predictions` | Prediction file search root |
| `--eval-split` | str | `val` | Dataset split |
| `--hot3d-conversion` | str | `auto` | HOT3D camera mode |
| `--no-undistort-inp` | flag | — | Use `undistort_inp=False` variant |
| `--no-cit` | flag | — | Disable CIT module |
| `--no-arm-prior` | flag | — | Disable arm prior |
| `--no-arm-input` | flag | — | Disable arm input |
| `--anycalib-624` / `--anycalib-pin` | flag | — | AnyCalib wrapper variants |
| `--depth-model` / `--dgp-model` | flag | — | Refinement model flags |
| `--disable-kalman-filter` | flag | — | Skip Kalman temporal filtering |
| `--ignore-failure-solves` / `--no-ignore-failure-solves` | flag | `True` | Filter samples with CS > 1000 mm |
| `--results-root` | str | `results/` | Output base directory |
| `--batch-size` | int | — | Evaluation batch size |
| `--num-workers` | int | — | DataLoader workers |

</details>

**Output:** `results/OURS/<dataset>_<suffix>_evaluation_results.txt`

**Example:**

```bash
python experiments/evaluate_predictions.py --test-dataset-name HOT3D
python experiments/evaluate_predictions.py --test-dataset-name ARCTIC --no-cit
python experiments/evaluate_predictions.py --test-dataset-name HOT3D --disable-kalman-filter
```

---

### `evaluate_noisy_intrinsic_predictions.py` — Noisy Intrinsic Evaluation

Evaluates all noisy-intrinsic predictions (generated by `save_noisy_intrinsic_predictions.py`), sweeping noise levels, computing per-level metrics, building a comparative sweep report, and correlating metric degradation with camera error features.

**Fixed configuration:** HOT3D dataset, `undistort_inp=True`, suffix base `undistort_inp_true`.

<details>
<summary><strong>CLI Arguments</strong></summary>

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--no-cit` | flag | — | Disable CIT module (appends `_no_cit` to suffix) |
| `--noisy-predictions-dir` | str | `_DATA/noisy_predictions` | Source directory for noisy predictions |
| `--results-dir` | str | `results/intrinsics_robustness` | Output directory for evaluation artifacts |
| `--baseline-noise-percent` | float | `0.5` | Baseline noise level for percent-increase calculations |
| `--force-recompute` | flag | — | Skip cache and recompute all evaluations |
| `--batch-size` | int | `16` | Evaluation batch size |
| `--num-workers` | int | `16` | DataLoader workers |

</details>

**Output artifacts (per noise level):**

| File | Description |
|------|-------------|
| `HOT3D_<suffix>_evaluation_results.txt` | Evaluation metrics text report |
| `HOT3D_<suffix>_filter_data.pkl` | Translation + visibility data for Kalman state |
| `HOT3D_<suffix>_camera_noise_analysis.pkl` | Camera ray error analysis |

**Output artifacts (sweep-level):**

| File | Description |
|------|-------------|
| `HOT3D_<suffix>_noise_sweep_report.pkl` | Complete sweep summary with correlations |
| `*_metrics_vs_noise.png` | Raw metrics across noise levels |
| `*_metric_increase_vs_noise.png` | Metric degradation (%) relative to baseline |
| `*_camera_error_vs_noise.png` | Camera error features across noise levels |

**Example:**

```bash
python experiments/evaluate_noisy_intrinsic_predictions.py
python experiments/evaluate_noisy_intrinsic_predictions.py --no-cit --force-recompute
```

---

### `evaluate_hand_scale.py` — Hand Scale Analysis

Evaluates hand scale (wrist-to-middle-finger distance) accuracy across distance bins and sequences. Calibrates per-sequence scale factors and reports pre/post-calibration metrics.

**Distance bins (mm):** 10, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 10000.

<details>
<summary><strong>CLI Arguments</strong></summary>

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--predictions-path` | str | **required** | Path to predictions pickle |
| `--results-dir` | str | auto | Output directory for CSVs/plots |
| `--suffix` | str | — | Explicit prediction suffix (overrides ablation flags) |
| `--no-undistort-inp` | flag | — | Use `undistort_inp=False` variant |
| `--no-cit` | flag | — | Disable CIT module |
| `--no-arm-prior` | flag | — | Disable arm prior |
| `--no-arm-input` | flag | — | Disable arm input |
| `--n-calibration-samples` | int | `50` | Samples per sequence for scale factor computation |
| `--plot` | flag | — | Enable plot generation |
| `--eval-batch-size` | int | — | Batch size for metric computation |
| `--fps` | float | `30.0` | Frame rate for acceleration metrics |
| `--acc-threshold-mm` | float | `10.0` | Accuracy-at-threshold (mm) |
| `--kalman-filter` | str | `none` | Temporal filter mode |
| `--ignore-failure-solves` | flag | `True` | Filter CS > 1000 mm outliers |

</details>

**Output artifacts:**

| File | Description |
|------|-------------|
| `<suffix>_distance_bin_aggregate.csv` | Per-distance-bin metrics (left/right/mean) |
| `<suffix>_sequence_scale_factors.csv` | Per-sequence calibrated scale factors |
| `<suffix>_sequence_metrics_before.csv` | Sequence-level metrics before calibration |
| `<suffix>_sequence_metrics_after.csv` | Sequence-level metrics after calibration |
| `*_scale_error_vs_distance_bins.png` | Scale error vs distance plot |
| `*_scale_error_scatter.png` | Per-frame scale error scatter |
| `*_sequence_scale_errors_top40.png` | Top-40 sequence scale errors bar chart |

**Example:**

```bash
python experiments/evaluate_hand_scale.py \
    --predictions-path _DATA/predictions/HOT3D_undistort_inp_true_predictions.pkl \
    --plot
```

---

### `hand_joint_occlusion_graph.py` — Occlusion Analysis

Compares hand predictions with and without arm input by hand-joint visibility bins. Quantifies how much arm information improves accuracy in occluded regions.

**Visibility bins:** 0–25%, 25–50%, 50–75%, 75–100%.

<details>
<summary><strong>CLI Arguments</strong></summary>

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--predictions-dir` | str | `_DATA/predictions` | Prediction file search root |
| `--log-path` | str | — | Optional subdirectory under predictions dir |
| `--results-root` | str | `results/` | Output base directory |
| `--output-dir` | str | — | Explicit output directory override |

</details>

**Output artifacts:**

| File | Description |
|------|-------------|
| `<dataset>_with_vs_without_arm_summary.json` | Aggregate improvement metrics |
| PNG plots | Visibility-binned CS/RR MPJPE comparison charts |

**Example:**

```bash
python experiments/hand_joint_occlusion_graph.py \
    --predictions-dir _DATA/predictions
```

---

## Ablation Flags

Several scripts share common ablation flags that modify the model configuration and append a corresponding token to the output suffix:

| Flag | Suffix token | Effect |
|------|-------------|--------|
| `--no-cit` | `_no_cit` | Disable CIT (Cross-modal Information Transfer) module |
| `--no-arm-prior` | `_no_arm_prior` | Disable arm prior in the model |
| `--no-arm-input` | `_no_arm_input` | Disable arm input features |
| `--no-undistort-inp` | `undistort_inp_false` | Use original (distorted) camera input |

These flags are supported by `save_predictions.py`, `evaluate_predictions.py`, and `evaluate_hand_scale.py`. The noisy intrinsic scripts (`save_noisy_intrinsic_predictions.py`, `evaluate_noisy_intrinsic_predictions.py`) support `--no-cit`.

## Temporal Filtering

`evaluate_predictions.py` and `evaluate_noisy_intrinsic_predictions.py` apply a Kalman constant-velocity filter (`KalmanFilterCV3D`) for temporal smoothing of predicted hand translations:

- **Parameters:** `q_pos=0.001`, `q_vel=1e-05`, `r_meas=0.001`, `freq=30.0`
- Disable with `--disable-kalman-filter` (in `evaluate_predictions.py`)
