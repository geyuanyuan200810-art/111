import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
matplotlib.rcParams['font.family'] = 'WenQuanYi Zen Hei'
matplotlib.rcParams['axes.unicode_minus'] = False

SEQ    = "/home/yuan/kitti_data/dataset/sequences/00"
TRAJ_B = "/home/yuan/slam_results/kitti_baseline/traj.txt"
TRAJ_S = "/home/yuan/slam_results/kitti_semantic/traj.txt"

DYNAMIC = {10,11,13,15,16,18,20}

vfiles = sorted(glob.glob(f"{SEQ}/velodyne/*.bin"))
lfiles = sorted(glob.glob(f"{SEQ}/labels/*.label"))

def load_traj(path):
    data = np.loadtxt(path)
    poses = []
    for row in data:
        T = np.eye(4)
        T[:3,:4] = row.reshape(3,4)
        poses.append(T)
    return poses

print("加载轨迹...")
poses_b = load_traj(TRAJ_B)
poses_s = load_traj(TRAJ_S)

STEP = 10
N = min(len(poses_b), len(poses_s), len(vfiles), len(lfiles))

map_b_static  = []
map_s_static  = []
map_s_dynamic = []
traj_b_xy = []
traj_s_xy = []

print(f"构建地图（每{STEP}帧采样）...")
for i in range(0, N, STEP):
    pts    = np.fromfile(vfiles[i], dtype=np.float32).reshape(-1,4)
    labels = np.fromfile(lfiles[i], dtype=np.uint32) & 0xFFFF
    dist   = np.sqrt((pts[:,:3]**2).sum(axis=1))
    mask   = (dist > 2.0) & (dist < 50.0) & (pts[:,2] > -1.5)
    pts    = pts[mask]; labels = labels[mask]
    ones   = np.ones((len(pts),1), dtype=np.float32)
    pts_h  = np.hstack([pts[:,:3], ones])

    T_b = poses_b[i]
    w_b = (T_b @ pts_h.T).T[:,:3]
    map_b_static.append(w_b[~np.isin(labels, list(DYNAMIC)), :2])
    traj_b_xy.append(poses_b[i][:2,3])

    T_s = poses_s[i]
    w_s = (T_s @ pts_h.T).T[:,:3]
    map_s_static.append(w_s[~np.isin(labels, list(DYNAMIC)), :2])
    map_s_dynamic.append(w_s[np.isin(labels, list(DYNAMIC)), :2])
    traj_s_xy.append(poses_s[i][:2,3])

    if i % 200 == 0: print(f"  帧 {i}/{N}")

def vstack(lst): return np.vstack(lst) if lst else np.zeros((0,2))
def subsample(arr, n):
    return arr[np.random.choice(len(arr),n,replace=False)] if len(arr)>n else arr

map_b = subsample(vstack(map_b_static), 80000)
map_s = subsample(vstack(map_s_static), 80000)
map_d = subsample(vstack(map_s_dynamic), 10000)
traj_b = np.array(traj_b_xy)
traj_s = np.array(traj_s_xy)

# 三个放大区域（根据KITTI序列00的地图特征手动选取）
zoom_regions = [
    ('A', 242, -9,  60, 40, '主干道区域\n动态车辆密集帧500'),
    ('B', 463, -197, 55, 35, '路口区域\n动态峰值32.69%帧2900'),
    ('C', 348, 272, 55, 35, '上方转弯处\n帧4000附近'),
]

BG='#0f0f0f'; GREEN='#4ade80'; RED='#ef4444'
BLUE='#3b82f6'; WHITE='white'; GOLD='#fbbf24'; GRAY='#888888'

fig = plt.figure(figsize=(22, 15), facecolor=BG)
gs  = GridSpec(2, 3, figure=fig,
               hspace=0.10, wspace=0.10,
               top=0.93, bottom=0.04, left=0.03, right=0.97,
               height_ratios=[1.6, 1.0])

ax_L = fig.add_axes([0.03, 0.37, 0.45, 0.55])
ax_R = fig.add_axes([0.52, 0.37, 0.45, 0.55])
ax_z = [fig.add_subplot(gs[1,i]) for i in range(3)]

def style(ax):
    ax.set_facecolor('#111827')
    for sp in ax.spines.values(): sp.set_color('#374151')
    ax.tick_params(colors=GRAY, labelsize=9)
    ax.set_xlabel('X (m)', color=GRAY, fontsize=9)
    ax.set_ylabel('Y (m)', color=GRAY, fontsize=9)
    ax.grid(True, color='#1f2937', lw=0.4, alpha=0.5)
    ax.set_aspect('equal')

xl = [map_b[:,0].min()-20, map_b[:,0].max()+20]
yl = [map_b[:,1].min()-20, map_b[:,1].max()+20]

# 主图左：基线
style(ax_L)
ax_L.scatter(map_b[:,0], map_b[:,1], s=0.08, c=GREEN, alpha=0.25, linewidths=0)
ax_L.plot(traj_b[:,0], traj_b[:,1], color=RED, lw=1.2, alpha=0.9, label='基线轨迹')
ax_L.plot(traj_b[0,0], traj_b[0,1], 'o', color=WHITE, ms=5, zorder=10)
ax_L.set_title('基线 KISS-ICP\nATE均值 = 2.698 m',
               color=RED, fontsize=12, fontweight='bold', pad=6)
ax_L.set_xlim(*xl); ax_L.set_ylim(*yl)
ax_L.legend(loc='upper right', facecolor='#111827', edgecolor='#374151',
            labelcolor=WHITE, fontsize=9)
ax_L.text(0.02, 0.03, '动态目标点云未处理\n（混入静态地图）',
          transform=ax_L.transAxes, color=GOLD, fontsize=9,
          bbox=dict(boxstyle='round,pad=0.4', fc='#1a1a1a', ec=GOLD, alpha=0.85))

for lbl, cx, cy, hw, hh, _ in zoom_regions:
    ax_L.add_patch(Rectangle((cx-hw,cy-hh),2*hw,2*hh,
                              lw=1.8,edgecolor=GOLD,facecolor='none',ls='--',zorder=8))
    ax_L.text(cx-hw+2, cy+hh-8, lbl, color=GOLD, fontsize=12, fontweight='bold')

# 主图右：语义版
style(ax_R)
ax_R.scatter(map_s[:,0], map_s[:,1], s=0.08, c=GREEN, alpha=0.25, linewidths=0,
             label='静态环境点云')
ax_R.scatter(map_d[:,0], map_d[:,1], s=0.5, c=RED, alpha=0.55, linewidths=0,
             label='动态目标点云（已标注）')
ax_R.plot(traj_s[:,0], traj_s[:,1], color=BLUE, lw=1.2, alpha=0.9, label='语义版轨迹')
ax_R.plot(traj_s[0,0], traj_s[0,1], 'o', color=WHITE, ms=5, zorder=10)
ax_R.set_title('语义权重版本（本研究）\nATE均值 = 2.516 m  ↓6.8%',
               color=BLUE, fontsize=12, fontweight='bold', pad=6)
ax_R.set_xlim(*xl); ax_R.set_ylim(*yl)
ax_R.legend(loc='upper right', facecolor='#111827', edgecolor='#374151',
            labelcolor=WHITE, fontsize=9)
ax_R.text(0.02, 0.03, '红色点=行人/车辆\n已识别并标注\n（不参与定位优化）',
          transform=ax_R.transAxes, color='#86efac', fontsize=9,
          bbox=dict(boxstyle='round,pad=0.4', fc='#1a1a1a', ec='#22c55e', alpha=0.85))

for lbl, cx, cy, hw, hh, _ in zoom_regions:
    ax_R.add_patch(Rectangle((cx-hw,cy-hh),2*hw,2*hh,
                              lw=1.8,edgecolor=GOLD,facecolor='none',ls='--',zorder=8))
    ax_R.text(cx-hw+2, cy+hh-8, lbl, color=GOLD, fontsize=12, fontweight='bold')

fig.text(0.5, 0.965, 'KITTI 序列00 建图效果对比（俯视图）+ 局部放大分析',
         ha='center', color=WHITE, fontsize=14, fontweight='bold')
fig.text(0.5, 0.945, '黄色虚线框 A/B/C → 见下方放大分析',
         ha='center', color=GOLD, fontsize=9)

# 放大子图：每个区域分左（基线）右（语义）两个子图
zoom_axes = []
for i in range(3):
    ax_left  = fig.add_axes([0.03 + i*0.323, 0.04, 0.148, 0.28])
    ax_right = fig.add_axes([0.03 + i*0.323 + 0.152, 0.04, 0.148, 0.28])
    zoom_axes.append((ax_left, ax_right))

# 移除原来的ax_z
for ax_zi in ax_z:
    ax_zi.remove()

for (ax_bl, ax_sm), (lbl, cx, cy, hw, hh, desc) in zip(zoom_axes, zoom_regions):
    def local(arr):
        m = (np.abs(arr[:,0]-cx)<hw*1.4) & (np.abs(arr[:,1]-cy)<hh*1.4)
        return arr[m]
    def local_traj(t):
        m = (np.abs(t[:,0]-cx)<hw*1.4) & (np.abs(t[:,1]-cy)<hh*1.4)
        return t[m]

    lb = local(map_b); ls = local(map_s); ld = local(map_d)
    tb = local_traj(traj_b); ts = local_traj(traj_s)

    for ax_zi in [ax_bl, ax_sm]:
        ax_zi.set_facecolor('#111827')
        for sp in ax_zi.spines.values(): sp.set_color('#374151')
        ax_zi.tick_params(colors=GRAY, labelsize=8)
        ax_zi.set_xlim(cx-hw, cx+hw)
        ax_zi.set_ylim(cy-hh, cy+hh)
        ax_zi.grid(True, color='#1f2937', lw=0.4, alpha=0.5)

    # 左格：基线
    if len(lb): ax_bl.scatter(lb[:,0], lb[:,1], s=1.2, c=GREEN,   alpha=0.35, linewidths=0)
    if len(ld): ax_bl.scatter(ld[:,0], ld[:,1], s=3,   c='#f97316', alpha=0.6,  linewidths=0)
    if len(tb)>1: ax_bl.plot(tb[:,0], tb[:,1], color=RED, lw=2.2, alpha=0.95,
                   path_effects=[pe.Stroke(linewidth=4,foreground='#450a0a'),pe.Normal()])
    ax_bl.set_title(f'{lbl} 基线KISS-ICP', color=RED, fontsize=8.5, fontweight='bold', pad=3)
    ax_bl.text(0.04, 0.93, '动态点混入地图', transform=ax_bl.transAxes,
               color='#f97316', fontsize=7.5,
               bbox=dict(boxstyle='round,pad=0.2', fc='#1a1a1a', ec='#f97316', alpha=0.8))

    # 右格：语义版
    if len(ls): ax_sm.scatter(ls[:,0], ls[:,1], s=1.2, c='#60a5fa', alpha=0.35, linewidths=0)
    if len(ld): ax_sm.scatter(ld[:,0], ld[:,1], s=3,   c=RED,       alpha=0.7,  linewidths=0)
    if len(ts)>1: ax_sm.plot(ts[:,0], ts[:,1], color='#60a5fa', lw=2.2, alpha=0.95,
                   path_effects=[pe.Stroke(linewidth=4,foreground='#082f49'),pe.Normal()])
    ax_sm.set_title(f'{lbl} 语义加权（本研究）', color='#60a5fa', fontsize=8.5, fontweight='bold', pad=3)
    ax_sm.text(0.04, 0.93, '动态点已识别过滤', transform=ax_sm.transAxes,
               color='#4ade80', fontsize=7.5,
               bbox=dict(boxstyle='round,pad=0.2', fc='#1a1a1a', ec='#4ade80', alpha=0.8))

    # 区域标题
    fig.text(0.03 + zoom_regions.index((lbl,cx,cy,hw,hh,desc))*0.323 + 0.148,
             0.335, f'区域{lbl}: {desc}',
             ha='center', color=GOLD, fontsize=8.5, fontweight='bold')

fig.text(0.5, 0.008,
         '绿色=静态环境点云  蓝色=语义版静态点云  红色散点=被识别的动态目标  红/蓝线=两版本轨迹',
         ha='center', color='#9ca3af', fontsize=8.5)

out = '/home/yuan/slam_results/map_compare_enhanced.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
print(f"完成！已保存：{out}")
