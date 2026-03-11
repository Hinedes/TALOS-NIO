import numpy as np
from pathlib import Path

# Import the exact ground truth loader your pipeline already uses
from nymeria_loader import load_gt_trajectory

def scan_sequences():
    root = Path('/mnt/c/TALOS/nymeria')
    print("Scanning Nymeria Ground Truth Trajectories...\n")
    
    results = []
    for traj_path in root.rglob('closed_loop_trajectory.csv'):
        seq_name = traj_path.parents[3].name
        
        try:
            # Rely on your existing, proven parser
            gt_ts, gt_pos, gt_quat = load_gt_trajectory(traj_path)
            
            gt_pos = gt_pos - gt_pos[0] # Zero out to origin
            
            max_bounds = np.max(gt_pos, axis=0)
            min_bounds = np.min(gt_pos, axis=0)
            box = max_bounds - min_bounds
            max_displacement = np.max(box)
            
            results.append((seq_name, box, max_displacement))
        except Exception as e:
            print(f"Error reading {seq_name}: {e}")
            
    # Sort by maximum physical displacement
    results.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'Sequence Name':<60} | {'X (m)':<8} | {'Y (m)':<8} | {'Z (m)':<8} | {'Max Disp (m)'}")
    print("-" * 105)
    for seq, box, max_disp in results:
        print(f"{seq[:60]:<60} | {box[0]:<8.2f} | {box[1]:<8.2f} | {box[2]:<8.2f} | {max_disp:.2f}m")

if __name__ == '__main__':
    scan_sequences()