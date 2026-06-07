import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm

try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'
except:
    pass

def load_poses(filepath):
    poses = []
    with open(filepath) as f:
        for line in f:
            nums = list(map(float, line.strip().split()))
            if len(nums) == 12:
                poses.append([nums[3], nums[7], nums[11]])
            elif len(nums) == 8:
                poses.append([nums[1], nums[2], nums[3]])
            elif len(nums) == 16:
                poses.append([nums[3], nums[7], nums[11]])
    return np.array(poses)

def umeyama_align(src, dst):
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = dst_c.T @ src_c / len(src)
    U, S, Vt = np.linalg.svd(cov)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    scale = np.sum(S) / np.sum(src_c**2 / len(src))
    t = mu_dst - scale * R @ mu_src
    return (scale * (R @ src.T)).T + t

print("读取文件中...")
gt       = load_poses('/home/yuan/kitti_data/dataset/sequences/00/poses.txt')
baseline = load_poses('/home/yuan/slam_results/kitti_baseline/traj.txt')
semantic = load_poses('/home/yuan/slam_results/kitti_semantic/traj.txt')

N = min(len(gt), len(baseline), len(semantic))
gt       = gt[:N]
baseline = baseline[:N]
semantic = semantic[:N]

baseline = umeyama_align(baseline, gt)
semantic  = umeyama_align(semantic,  gt)

gt_x, gt_y     = gt[:, 0],       gt[:, 1]
base_x, base_y = baseline[:, 0], baseline[:, 1]
sem_x,  sem_y  = semantic[:, 0],  semantic[:, 1]
print(f"帧数: {N}")

ate_base = np.sqrt((base_x - gt_x)**2 + (base_y - gt_y)**2)
ate_sem  = np.sqrt((sem_x  - gt_x)**2 + (sem_y  - gt_y)**2)
ATE_base_mean = ate_base.mean()
ATE_sem_mean  = ate_sem.mean()
improve = (ATE_base_mean - ATE_sem_mean) / ATE_base_mean * 100
print(f"ATE 基线={ATE_base_mean:.3f}m  语义={ATE_sem_mean:.3f}m  提升={improve:.1f}%")

frames = np.arange(N)

def find_peak(mask=None):
    err = ate_base.copy()
    if mask is not None:
        err[~mask] = 0
    return np.argmax(err)

peak_A = find_peak()
peak_B = find_peak((frames > N//5) & (frames < 2*N//5))
peak_C = find_peak((frames > N//2) & (frames < 3*N//4))

zoom_defs = [
    ('A', peak_A, 80, 12, f'最大误差区域 A  帧{peak_A}'),
    ('B', peak_B, 80, 12, f'中段误差区域 B  帧{peak_B}'),
    ('C', peak_C, 80, 12, f'后段误差区域 C  帧{peak_C}'),
]

BG='#0d0d0d'; RED='#f87171'; CYAN='#22d3ee'
WHITE='#f1f5f9'; GOLD='#fbbf24'; GRAY='#6b7280'

fig = plt.figure(figsize=(22, 14), facecolor=BG)
gs  = GridSpec(2, 3, figure=fig,
               hspace=0.12, wspace=0.12,
               top=0.93, bottom=0.05, left=0.04, right=0.97,
               height_ratios=[1.55, 1.0])

ax_L = fig.add_axes([0.04, 0.37, 0.44, 0.54])
ax_R = fig.add_axes([0.53, 0.37, 0.44, 0.54])
ax_z = [fig.add_subplot(gs[1, i]) for i in range(3)]

def style(ax):
    ax.set_facecolor('#111827')
    for sp in ax.spines.values(): sp.set_color('#374151')
    ax.tick_params(colors=GRAY, labelsize=9)
    ax.set_xlabel('X (m)', color=GRAY, fontsize=9)
    ax.set_ylabel('Y (m)', color=GRAY, fontsize=9)
    ax.grid(True, color='#1f2937', lw=0.4, alpha=0.6)

xl = [gt_x.min()-30, gt_x.max()+30]
yl = [gt_y.min()-30, gt_y.max()+30]

style(ax_L)
ax_L.plot(base_x, base_y, color=RED, lw=0.8, alpha=0.9, label='基线轨迹')
ax_L.plot(gt_x[0], gt_y[0], 'o', color=WHITE, ms=5, zorder=10)
ax_L.set_title(f'基线 KISS-ICP\nATE均值 = {ATE_base_mean:.3f} m',
               color=RED, fontsize=12, fontweight='bold', pad=6)
ax_L.set_xlim(*xl); ax_L.set_ylim(*yl)
ax_L.legend(loc='upper right', facecolor='#111827', edgecolor='#374151',
            labelcolor=WHITE, fontsize=9)

for lbl, pidx, hw, hh, _ in zoom_defs:
    cx, cy = gt_x[pidx], gt_y[pidx]
    ax_L.add_patch(Rectangle((cx-hw, cy-hh), 2*hw, 2*hh,
                              lw=1.8, edgecolor=GOLD, facecolor='none', ls='--', zorder=8))
    ax_L.text(cx-hw+2, cy+hh-8, lbl, color=GOLD, fontsize=12, fontweight='bold')

style(ax_R)
ax_R.plot(gt_x, gt_y, ':', color='#d1d5db', lw=0.6, alpha=0.4, label='真值 (GT)')
ax_R.plot(sem_x, sem_y, color=CYAN, lw=0.8, alpha=0.9, label='语义加权轨迹')
ax_R.plot(gt_x[0], gt_y[0], 'o', color=WHITE, ms=5, zorder=10)
ax_R.set_title(f'语义加权 KISS-ICP（本研究）\nATE均值 = {ATE_sem_mean:.3f} m  ↓{improve:.1f}%',
               color=CYAN, fontsize=12, fontweight='bold', pad=6)
ax_R.set_xlim(*xl); ax_R.set_ylim(*yl)
ax_R.legend(loc='upper right', facecolor='#111827', edgecolor='#374151',
            labelcolor=WHITE, fontsize=9)

for lbl, pidx, hw, hh, _ in zoom_defs:
    cx, cy = gt_x[pidx], gt_y[pidx]
    ax_R.add_patch(Rectangle((cx-hw, cy-hh), 2*hw, 2*hh,
                              lw=1.8, edgecolor=GOLD, facecolor='none', ls='--', zorder=8))
    ax_R.text(cx-hw+2, cy+hh-8, lbl, color=GOLD, fontsize=12, fontweight='bold')

fig.text(0.5, 0.965, 'KITTI 序列00 轨迹对比（俯视图）+ 局部放大分析',
         ha='center', color=WHITE, fontsize=14, fontweight='bold')
fig.text(0.5, 0.945, '黄色虚线框 A/B/C → 见下方放大分析',
         ha='center', color=GOLD, fontsize=9)

for ax_zi, (lbl, pidx, hw, hh, desc) in zip(ax_z, zoom_defs):
    cx, cy = gt_x[pidx], gt_y[pidx]
    style(ax_zi)
    in_reg = np.sqrt((gt_x - cx)**2 + (gt_y - cy)**2) < max(hw, hh) * 1.5

    ax_zi.plot(gt_x[in_reg], gt_y[in_reg], '--',
               color='#d1d5db', lw=1.5, alpha=0.55, label='GT真值', zorder=5)
    ax_zi.plot(base_x[in_reg], base_y[in_reg], color=RED, lw=3.0, alpha=0.95,
               label='基线 KISS-ICP', zorder=6,
               path_effects=[pe.Stroke(linewidth=5, foreground='#450a0a'), pe.Normal()])
    ax_zi.plot(sem_x[in_reg], sem_y[in_reg], color=CYAN, lw=3.0, alpha=0.95,
               label='语义加权（本研究）', zorder=7,
               path_effects=[pe.Stroke(linewidth=5, foreground='#082f49'), pe.Normal()])

    local_idx = np.where(in_reg)[0]
    if len(local_idx) > 0:
        max_li   = local_idx[np.argmax(ate_base[local_idx])]
        max_base = ate_base[max_li]
        max_sem  = ate_sem[max_li]
        bx, by   = base_x[max_li], base_y[max_li]
        sx, sy   = sem_x[max_li],  sem_y[max_li]

        ax_zi.annotate(f'基线偏差\n{max_base:.2f} m',
                       xy=(bx, by), xytext=(bx + hw*0.15, by + hh*0.5),
                       color=RED, fontsize=8.5, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color=RED, lw=1.2),
                       bbox=dict(boxstyle='round,pad=0.3', fc='#1c0a0a', ec=RED, alpha=0.9),
                       zorder=10)
        ax_zi.annotate(f'语义版偏差\n{max_sem:.2f} m',
                       xy=(sx, sy), xytext=(sx - hw*0.15, sy - hh*0.5),
                       color=CYAN, fontsize=8.5, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color=CYAN, lw=1.2),
                       bbox=dict(boxstyle='round,pad=0.3', fc='#082f49', ec=CYAN, alpha=0.9),
                       zorder=10)

    ax_zi.set_xlim(cx-hw, cx+hw); ax_zi.set_ylim(cy-hh, cy+hh)
    ax_zi.set_title(f'放大区域 {lbl}  |  {desc}',
                    color=GOLD, fontsize=9.5, fontweight='bold', pad=4)
    ax_zi.legend(loc='lower right', facecolor='#0f172a', edgecolor='#374151',
                 labelcolor=WHITE, fontsize=8)

fig.text(0.5, 0.008,
         '红色=基线KISS-ICP  青色=语义加权版本  灰色虚线=GT真值  标注数字为该帧ATE误差（m）',
         ha='center', color='#9ca3af', fontsize=9)

plt.savefig('/home/yuan/slam_figure/figure3_enhanced.png', dpi=180,
            bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close()
print("完成！图已保存为 /home/yuan/slam_figure/figure3_enhanced.png")
