#!/usr/bin/env python3
"""
eval_best.py : Load and evaluate the latest best_physical checkpoint

Usage:
    python eval_best.py [--golden /home/TALOS/golden] [--val-seq PATH] [--max-seconds 300]
"""

import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from incremental_train import load_continuous_val_stream, evaluate_eskf, ESKF_DT
from SMLP import SpectralMLP


def find_latest_checkpoint(golden_dir: Path) -> Path:
    """Find the latest run_YYYYMMDD_HHMMSS folder and return best_physical.pth if it exists."""
    golden_dir = Path(golden_dir)
    
    runs = list(golden_dir.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No run folders found in {golden_dir}")
    
    # Sort by folder name (YYYYMMDD_HHMMSS order naturally sorts chronologically)
    latest_run = sorted(runs)[-1]
    ckpt_path = latest_run / "talos_best_physical.pth"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"[Checkpoint] Found: {latest_run.name}/talos_best_physical.pth")
    return ckpt_path


def run_eval(golden_dir: str, val_seq_path: str, max_seconds: int = 300, output_dir: str = None):
    """Load checkpoint and evaluate on validation sequence."""
    
    golden_dir = Path(golden_dir)
    val_seq_path = Path(val_seq_path)
    
    # Find latest checkpoint
    ckpt_path = find_latest_checkpoint(golden_dir)
    
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    
    model = SpectralMLP().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt, strict=False)
    print(f"[Model] Loaded weights from checkpoint")
    
    # Load validation data
    print(f"[Data] Loading validation sequence: {val_seq_path}")
    cache_dir = golden_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache = cache_dir / f"{val_seq_path.parent.name}_val_stream.pkl"
    
    if _cache.exists():
        print(f"[Cache] HIT val_stream")
        import pickle
        df, gravity = pickle.load(open(_cache, 'rb'))
    else:
        print(f"[Cache] MISS -- reading VRS (slow)")
        df, gravity = load_continuous_val_stream(val_seq_path)
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path.cwd() / "eval_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    print(f"\n[Eval] Running evaluate_eskf ({max_seconds}s)...")
    df_walk = df.iloc[313*100:].reset_index(drop=True)  # Skip initial settle period
    
    ate = evaluate_eskf(model, df_walk, gravity, device, 0, output_dir, max_seconds=max_seconds)
    print(f"\n[Result] ATE ({max_seconds}s): {ate:.3f}m")
    
    # Extract trajectory data for secondary analysis
    talos_pos = evaluate_eskf._last_talos_pos
    talos_pos = talos_pos - talos_pos[0]
    N = len(talos_pos)
    gt_pos = df_walk[['px','py','pz']].values[:N].astype('float32')
    gt_pos = gt_pos - gt_pos[0]
    
    # Compute RTE (Relative Trajectory Error)
    total_distance = np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1))
    rte = (ate / total_distance) * 100 if total_distance > 0 else 0.0
    
    print(f"[Result] Total distance: {total_distance:.1f}m")
    print(f"[Result] RTE: {rte:.2f}%")
    print(f"[Output] Plots saved to: {output_dir}")
    
    # Summary stats
    eval_summary = getattr(evaluate_eskf, '_last_summary', {})
    print(f"\n[Summary]")
    print(f"  Neural updates: {eval_summary.get('neural_updates', 'N/A')}")
    print(f"  Slap rate: {eval_summary.get('slap_rate_pct', 'N/A')}%")
    print(f"  Cage clamp rate: {eval_summary.get('cage_clamp_rate_pct', 'N/A')}%")
    print(f"  Yaw error (mean): {eval_summary.get('yaw_err_mean_deg', 'N/A'):.2f}°")
    print(f"  Pure IMU ATE: {eval_summary.get('pure_imu_ate_m', 'N/A'):.3f}m")
    
    return ate, rte


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate latest TALOS checkpoint")
    parser.add_argument('--golden', default=None, 
                        help='Path to golden directory containing run_* folders (default: ~/TALOS/golden)')
    parser.add_argument('--val-seq', default=None,
                        help='Path to validation sequence (default: auto-search Shelby Arroyo)')
    parser.add_argument('--max-seconds', type=int, default=300,
                        help='Maximum seconds of validation sequence to evaluate')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for plots (default: ./eval_output)')
    
    args = parser.parse_args()
    
    # Auto-detect golden directory
    if args.golden is None:
        args.golden = Path.home() / 'TALOS' / 'golden'
    
    # Auto-find Shelby Arroyo validation sequence
    if args.val_seq is None:
        nymeria_path = Path.home() / 'TALOS' / 'nymeria'
        val_seqs = list(nymeria_path.glob('Nymeria_v0.0_*shelby_arroyo*recording_head'))
        if not val_seqs:
            raise FileNotFoundError(f"No Shelby Arroyo sequences found in {nymeria_path}")
        args.val_seq = sorted(val_seqs)[0] / 'recording_head'
    
    ate, rte = run_eval(
        golden_dir=args.golden,
        val_seq_path=args.val_seq,
        max_seconds=args.max_seconds,
        output_dir=args.output_dir
    )
    
    print(f"\n✓ Evaluation complete: ATE={ate:.3f}m, RTE={rte:.2f}%")
