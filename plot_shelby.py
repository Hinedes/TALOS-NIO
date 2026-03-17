import sys
sys.path.insert(0, '/mnt/c/TALOS')
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from incremental_train import load_continuous_val_stream, evaluate_eskf, ESKF_DT
from SMLP import SpectralMLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = SpectralMLP().to(device)
ckpt   = torch.load('/mnt/c/TALOS/golden/talos_best_physical.pth', map_location=device)
model.load_state_dict(ckpt, strict=False)

val_path = Path('/mnt/c/TALOS/nymeria/Nymeria_v0.0_20230608_s0_shelby_arroyo_act0_3ciwl8_recording_head/recording_head')
import pickle
_cache = Path('/mnt/c/TALOS/golden/cache') / f'{val_path.parent.name}_val_stream.pkl'
if _cache.exists():
    print(f'[cache] HIT val_stream')
    df, gravity = pickle.load(open(_cache, 'rb'))
else:
    print(f'[cache] MISS -- reading VRS (slow)')
    df, gravity = load_continuous_val_stream(val_path)

MAX_SECONDS = 300
print(f'Running evaluate_eskf ({MAX_SECONDS}s)...')
df_walk = df.iloc[313*100:].reset_index(drop=True)
ate = evaluate_eskf(model, df_walk, gravity, device, 0, Path('/mnt/c/TALOS/golden'), max_seconds=MAX_SECONDS)
print(f'ATE ({MAX_SECONDS}s): {ate:.3f}m')

talos_pos = evaluate_eskf._last_talos_pos
talos_pos = talos_pos - talos_pos[0]
N = len(talos_pos)
gt_pos = df_walk[['px','py','pz']].values[:N].astype('float32')
gt_pos = gt_pos - gt_pos[0]

dt  = ESKF_DT
t   = np.arange(len(talos_pos)) * dt
ate_over_time = np.linalg.norm(talos_pos - gt_pos, axis=1)
ds  = max(1, len(talos_pos) // 3000)

markers = {'60s': int(60/dt), '150s': int(150/dt), '300s': int(300/dt)-1}

BG='#0d0d0d'; GT='#4488ff'; TALOS='#ff3333'; START='#44ff88'; ANNOT='#aaaaaa'

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_edgecolor('#333333')
    ax.tick_params(colors=ANNOT, labelsize=8)
    ax.xaxis.label.set_color(ANNOT); ax.yaxis.label.set_color(ANNOT)
    ax.title.set_color('#dddddd')
    ax.grid(True, color='#1f1f1f', linewidth=0.7, zorder=0)
    ax.set_title(title, fontsize=10); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

fig = plt.figure(figsize=(20, 7), facecolor=BG)
fig.suptitle(f'TALOS NIO  —  Shelby Arroyo  |  {MAX_SECONDS}s  |  GT (blue) vs TALOS+LAID (red)  |  Best ATE: {ate:.3f}m',
             color='#ffffff', fontsize=13, fontweight='bold', y=1.01)

ax1 = fig.add_subplot(1,3,1)
style_ax(ax1, 'Top-Down View (XY)', 'X (m)', 'Y (m)')
ax1.plot(gt_pos[::ds,0],    gt_pos[::ds,1],    color=GT,    lw=1.4, label='GT',    zorder=2)
ax1.plot(talos_pos[::ds,0], talos_pos[::ds,1], color=TALOS, lw=1.0, alpha=0.85, label='TALOS', zorder=1)
ax1.plot(gt_pos[0,0], gt_pos[0,1], 'o', color=START, ms=7, zorder=4, label='Start')
for label, idx in markers.items():
    if idx < len(gt_pos):
        ax1.annotate(label, xy=(gt_pos[idx,0], gt_pos[idx,1]), color='#ffcc00', fontsize=7, xytext=(5,5), textcoords='offset points')
        ax1.plot(gt_pos[idx,0], gt_pos[idx,1], '+', color='#ffcc00', ms=8, zorder=3)
ax1.set_aspect('equal')
ax1.legend(facecolor='#1a1a1a', edgecolor='#333333', labelcolor='#dddddd', fontsize=8)

ax2 = fig.add_subplot(1,3,2)
style_ax(ax2, 'Side View (XZ)', 'X (m)', 'Z (m)')
ax2.plot(gt_pos[::ds,0],    gt_pos[::ds,2],    color=GT,    lw=1.4, label='GT',    zorder=2)
ax2.plot(talos_pos[::ds,0], talos_pos[::ds,2], color=TALOS, lw=1.0, alpha=0.85, label='TALOS', zorder=1)
ax2.plot(gt_pos[0,0], gt_pos[0,2], 'o', color=START, ms=7, zorder=4)
ax2.set_aspect('equal')
ax2.legend(facecolor='#1a1a1a', edgecolor='#333333', labelcolor='#dddddd', fontsize=8)

ax3 = fig.add_subplot(1,3,3)
style_ax(ax3, 'ATE over Time', 'Time (s)', 'ATE (m)')
ax3.plot(t[::ds], ate_over_time[::ds], color='#ff8800', lw=1.3, label='ATE', zorder=2)
ax3.axhline(2.535, color='#44ff88', lw=0.9, ls='--', label='Best 2.535m', zorder=1)
for label, idx in markers.items():
    if idx < len(ate_over_time):
        ax3.annotate(f'{label}\n{ate_over_time[idx]:.2f}m',
            xy=(t[idx], ate_over_time[idx]), color='#ffcc00', fontsize=7,
            xytext=(8,-15), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='#ffcc00', lw=0.7))
ax3.legend(facecolor='#1a1a1a', edgecolor='#333333', labelcolor='#dddddd', fontsize=8)

plt.tight_layout()
out = '/mnt/c/TALOS/golden/shelby_trajectory.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
print(f'Saved to {out}')
