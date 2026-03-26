"""
darwin.py : Evolutionary Algorithm for TALOS NIO Stagnation Recovery

Triggered when ESKF ATE stagnates for PATIENCE rounds. Diagnoses the failure
mode from telemetry and mutates ESKF fusion parameters to escape.

Safety contract:
    - NEVER modifies incremental_train.py globals or source code
    - Returns a plain dict of fusion params passed through evaluate_eskf(fusion_params=...)
    - Delete darwin_config.json to revert everything to hardcoded defaults
"""

import copy
import json
import numpy as np
from pathlib import Path

# ── Genome Definition ──────────────────────────────────────────────────────────
# Each gene: (default, min, max, log_scale)
# log_scale=True → mutations operate in log-space (for values spanning orders of magnitude)

GENOME_SPEC = {
    'SLAP_THRESHOLD':           (4.00,  1.5,   8.0,   False),
    'R_OBS_FIXED_DIAG':         (0.10,  0.01,  1.0,   True),
    'PRED_VEL_GAIN':            (1.00,  0.3,   2.0,   False),
    'CAGE_RADIUS':              (0.30,  0.10,  1.00,  False),
    'MAX_PRED_WORLD_SPEED_MPS': (5.00,  2.0,  10.0,   False),
    'MAX_INNOVATION_NORM_MPS':  (5.00,  2.0,  10.0,   False),
    'USE_DYNAMIC_R_OBS':        (False, None,  None,  None),  # boolean toggle
}

# Mutation strength (sigma as fraction of [min,max] range)
BASE_SIGMA = 0.15


class DarwinEngine:
    """(1+λ) Evolutionary Strategy for ESKF fusion parameters."""

    def __init__(self, population_size: int = 7, seed: int = 42):
        self.lam = population_size
        self.rng = np.random.default_rng(seed)
        self.generation = 0
        self.history = []  # audit trail of all generations

    # ── Diagnostician ──────────────────────────────────────────────────────────

    def diagnose(self, summary_history: list[dict]) -> dict:
        """Analyze recent ESKF telemetry to identify the dominant failure mode.

        Returns a dict of bias hints: gene_name -> direction (+1 increase, -1 decrease)
        with a magnitude weight [0, 1].
        """
        if not summary_history:
            return {}

        # Use last 5 evaluations (or fewer if unavailable)
        recent = summary_history[-5:]
        bias = {}

        # ── Slap rate: are we rejecting too many or too few updates?
        avg_slap = np.mean([s.get('slap_rate_pct', 0) for s in recent])
        if avg_slap > 40:
            bias['SLAP_THRESHOLD'] = (+1, 0.8)  # gate too tight, widen it
        elif avg_slap < 5:
            bias['SLAP_THRESHOLD'] = (-1, 0.5)  # gate too loose, tighten

        # ── Scale collapse / inflation
        avg_ratio = np.mean([s.get('pred_gt_speed_ratio', 1.0) for s in recent])
        if avg_ratio < 0.5:
            bias['PRED_VEL_GAIN'] = (+1, 0.9)   # scale collapse, boost gain
        elif avg_ratio > 1.5:
            bias['PRED_VEL_GAIN'] = (-1, 0.9)   # scale inflation, reduce gain
        elif avg_ratio < 0.8:
            bias['PRED_VEL_GAIN'] = (+1, 0.4)   # mild under-prediction
        elif avg_ratio > 1.2:
            bias['PRED_VEL_GAIN'] = (-1, 0.4)   # mild over-prediction

        # ── Cage clamp rate: is the model causing too much drift?
        avg_cage = np.mean([s.get('cage_clamp_rate_pct', 0) for s in recent])
        if avg_cage > 30:
            bias['R_OBS_FIXED_DIAG'] = (+1, 0.7)      # trust neural less
            bias['PRED_VEL_GAIN'] = bias.get('PRED_VEL_GAIN', (-1, 0.5))
        elif avg_cage < 1:
            bias['CAGE_RADIUS'] = (-1, 0.3)            # cage might be too generous

        # ── Innovation tension: filter and neural heavily disagree
        avg_innov = np.mean([s.get('innovation_norm_p95', 0) for s in recent])
        if avg_innov > 3.0:
            bias['R_OBS_FIXED_DIAG'] = (+1, 0.6)      # widen R_obs, trust neural less

        # ── Safety gate blocking too many updates
        avg_safety = 0
        for s in recent:
            nu = max(s.get('neural_updates', 1), 1)
            avg_safety += s.get('safety_reject_count', 0) / nu
        avg_safety /= len(recent)
        if avg_safety > 0.2:
            bias['MAX_PRED_WORLD_SPEED_MPS'] = (+1, 0.6)
            bias['MAX_INNOVATION_NORM_MPS'] = (+1, 0.6)

        # ── Yaw drift (informational -- we don't have yaw anchor in genome)
        avg_yaw = np.mean([s.get('yaw_err_p95_deg', 0) for s in recent])
        if avg_yaw > 30:
            # Can't fix yaw directly, but loosen cage to let filter breathe
            bias['CAGE_RADIUS'] = (+1, 0.4)

        return bias

    # ── Mutation Engine ────────────────────────────────────────────────────────

    def _get_defaults(self) -> dict:
        return {gene: spec[0] for gene, spec in GENOME_SPEC.items()}

    def spawn_mutants(self, parent: dict, diagnosis: dict, n: int | None = None) -> list[dict]:
        """Create n mutant parameter sets from the parent, biased by diagnosis."""
        n = n or self.lam
        mutants = []

        for _ in range(n):
            child = copy.deepcopy(parent)

            for gene, (default, lo, hi, log_scale) in GENOME_SPEC.items():
                if lo is None:  # boolean gene
                    # Small probability of flipping (10%)
                    if self.rng.random() < 0.10:
                        child[gene] = not child[gene]
                    continue

                # Base sigma scaled by range
                sigma = BASE_SIGMA * (hi - lo)

                # Apply diagnostic bias
                if gene in diagnosis:
                    direction, weight = diagnosis[gene]
                    # Bias the mutation center toward the diagnosed direction
                    bias_offset = direction * weight * sigma * 1.5
                else:
                    bias_offset = 0.0

                if log_scale:
                    # Mutate in log-space for parameters spanning orders of magnitude
                    log_val = np.log10(max(child[gene], 1e-10))
                    log_lo, log_hi = np.log10(lo), np.log10(hi)
                    log_sigma = BASE_SIGMA * (log_hi - log_lo)
                    log_new = log_val + self.rng.normal(0, log_sigma) + (
                        bias_offset / (hi - lo) * (log_hi - log_lo)
                    )
                    child[gene] = float(np.clip(10 ** log_new, lo, hi))
                else:
                    new_val = child[gene] + self.rng.normal(0, sigma) + bias_offset
                    child[gene] = float(np.clip(new_val, lo, hi))

            mutants.append(child)

        return mutants

    # ── Evolution Core ─────────────────────────────────────────────────────────

    def evolve(self, evaluate_fn, parent_params: dict | None,
               summary_history: list[dict], run_dir: Path) -> dict:
        """Full EA cycle: diagnose → spawn → evaluate → select.

        Args:
            evaluate_fn:     callable(params_dict) -> float (ATE). Must accept a
                             fusion_params dict and return the mean ATE.
            parent_params:   Current fusion params (or None for defaults).
            summary_history: List of recent summary_row dicts from evaluate_eskf.
            run_dir:         Directory for logging.

        Returns:
            The winning parameter dict.
        """
        self.generation += 1
        
        # Ensure parent has *all* genes, even if it came from Optuna which only tunes 3
        parent = self._get_defaults()
        if parent_params:
            for k, v in parent_params.items():
                if k in parent:
                    parent[k] = v

        # Step 1: Diagnose
        diagnosis = self.diagnose(summary_history)
        diag_str = {k: f"{'↑' if d > 0 else '↓'} (w={w:.1f})" for k, (d, w) in diagnosis.items()}
        print(f"\n  [Darwin Gen {self.generation}] Diagnosis: {diag_str or 'no clear signal'}")

        # Step 2: Spawn mutants
        mutants = self.spawn_mutants(parent, diagnosis)
        candidates = [parent] + mutants  # (1+λ) strategy

        # Step 3: Evaluate all candidates
        results = []
        for i, params in enumerate(candidates):
            label = "parent" if i == 0 else f"mutant-{i}"
            print(f"  [Darwin] Evaluating {label}...")
            try:
                ate = evaluate_fn(params)
                results.append((ate, params, label))
                print(f"  [Darwin] {label}: ATE = {ate:.4f}m")
            except Exception as e:
                print(f"  [Darwin] {label}: FAILED ({e})")
                results.append((float('inf'), params, label))

        # Step 4: Select winner (lowest ATE)
        results.sort(key=lambda x: x[0])
        best_ate, winner, best_label = results[0]
        print(f"\n  [Darwin Gen {self.generation}] Winner: {best_label} with ATE = {best_ate:.4f}m")

        # Step 5: Log generation
        gen_record = {
            'generation': self.generation,
            'diagnosis': {k: {'direction': d, 'weight': w} for k, (d, w) in diagnosis.items()},
            'parent': _sanitize_params(parent),
            'winner': _sanitize_params(winner),
            'winner_label': best_label,
            'winner_ate': best_ate,
            'all_results': [
                {'label': lbl, 'ate': ate, 'params': _sanitize_params(p)}
                for ate, p, lbl in results
            ],
        }
        self.history.append(gen_record)

        # Persist to disk
        log_path = run_dir / 'darwin_log.json'
        try:
            existing = json.loads(log_path.read_text()) if log_path.exists() else []
        except (json.JSONDecodeError, OSError):
            existing = []
        existing.append(gen_record)
        log_path.write_text(json.dumps(existing, indent=2))
        print(f"  [Darwin] Generation log saved → {log_path.name}")

        # Save winning config for potential manual inspection / revert
        config_path = run_dir / 'darwin_config.json'
        config_path.write_text(json.dumps(_sanitize_params(winner), indent=2))

        return winner


def _sanitize_params(params: dict) -> dict:
    """Ensure all values are JSON-serializable."""
    out = {}
    for k, v in params.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, (np.bool_,)):
            out[k] = bool(v)
        else:
            out[k] = v
    return out
