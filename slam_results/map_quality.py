"""
建图质量定量评估
指标1：动态点污染率（地图中动态目标点占比）
指标2：静态结构点云密度（建筑/道路点的保留情况）
指标3：局部一致性（同一区域多帧点云的重叠紧密程度）
"""
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'WenQuanYi Zen Hei'
matplotlib.rcParams['axes.unicode_minus'] = False

SEQ    = "/home/yuan/kitti_data/dataset/sequences/00"
TRAJ_B = "/home/yuan/slam_results/kitti_baseline/traj.txt"
TRAJ_S = "/home/yuan/slam_results/kitti_semantic/traj.txt"

DYNAMIC = {10,11,13,15,16,18,20}
STATIC  = {40,44,48,49,50,51,52,60,70,71,72,80,81}

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

# 统计量
total_pts_b      = 0  # 基线地图总点数
dynamic_pts_b    = 0  # 基线地图动态点数
static_pts_b     = 0  # 基线地图静态点数

total_pts_s      = 0
dynamic_pts_s    = 0
static_pts_s     = 0

# 局部一致性：收集某固定区域的点云，计算点间距离标准差
# 选取路口附近区域（X:180-260, Y:-30-30）作为评估区
REGION = (180, 260, -30, 30)
region_pts_b = []
region_pts_s = []

print(f"统计建图质量（每{STEP}帧采样）...")
for i in range(0, N, STEP):
    pts    = np.fromfile(vfiles[i], dtype=np.float32).reshape(-1,4)
    labels = np.fromfile(lfiles[i], dtype=np.uint32) & 0xFFFF

    dist = np.sqrt((pts[:,:3]**2).sum(axis=1))
    mask = (dist > 2.0) & (dist < 50.0) & (pts[:,2] > -1.5)
    pts = pts[mask]; labels = labels[mask]

    ones  = np.ones((len(pts),1), dtype=np.float32)
    pts_h = np.hstack([pts[:,:3], ones])

    # 基线
    T_b   = poses_b[i]
    w_b   = (T_b @ pts_h.T).T[:,:3]
    dyn_b = np.isin(labels, list(DYNAMIC))
    sta_b = np.isin(labels, list(STATIC))
    total_pts_b   += len(w_b)
    dynamic_pts_b += dyn_b.sum()
    static_pts_b  += sta_b.sum()

    # 基线区域点云
    rx = (w_b[:,0] > REGION[0]) & (w_b[:,0] < REGION[1])
    ry = (w_b[:,1] > REGION[2]) & (w_b[:,1] < REGION[3])
    region_pts_b.append(w_b[rx & ry & ~dyn_b, :2])

    # 语义版
    T_s   = poses_s[i]
    w_s   = (T_s @ pts_h.T).T[:,:3]
    dyn_s = np.isin(labels, list(DYNAMIC))
    sta_s = np.isin(labels, list(STATIC))
    total_pts_s   += len(w_s)
    dynamic_pts_s += 0          # 语义版动态点已过滤，不入地图
    static_pts_s  += sta_s.sum()

    rx = (w_s[:,0] > REGION[0]) & (w_s[:,0] < REGION[1])
    ry = (w_s[:,1] > REGION[2]) & (w_s[:,1] < REGION[3])
    region_pts_s.append(w_s[rx & ry & ~dyn_s, :2])

    if i % 200 == 0: print(f"  帧 {i}/{N}")

# ── 计算指标 ──────────────────────────────────────────────────────────
# 指标1：动态点污染率
contam_b = dynamic_pts_b / total_pts_b * 100
contam_s = dynamic_pts_s / total_pts_s * 100

# 指标2：静态点保留率（相对于基线）
retain_b = static_pts_b / total_pts_b * 100
retain_s = static_pts_s / total_pts_s * 100

# 指标3：局部一致性（路口区域点云密集程度）
# 用点云平均最近邻距离衡量，值越小说明同一结构被重复扫到且对齐越好
def mean_nn_dist(pts_list, max_pts=5000):
    if not pts_list: return 0
    all_pts = np.vstack(pts_list)
    if len(all_pts) > max_pts:
        idx = np.random.choice(len(all_pts), max_pts, replace=False)
        all_pts = all_pts[idx]
    if len(all_pts) < 10: return 0
    # 计算每点到最近邻的距离
    from sklearn.neighbors import KDTree
    tree = KDTree(all_pts)
    dist, _ = tree.query(all_pts, k=2)
    return dist[:,1].mean()  # 排除自身

print("计算局部一致性...")
from sklearn.neighbors import KDTree
nn_b = mean_nn_dist(region_pts_b)
nn_s = mean_nn_dist(region_pts_s)

print("\n" + "="*50)
print("建图质量定量评估结果")
print("="*50)
print(f"{'指标':<25} {'基线KISS-ICP':>15} {'语义加权(本研究)':>18} {'改善':>8}")
print("-"*50)
print(f"{'动态点污染率 (%)':.<25} {contam_b:>14.2f}% {contam_s:>17.2f}% {contam_b-contam_s:>+7.2f}%")
print(f"{'静态点保留率 (%)':.<25} {retain_b:>14.2f}% {retain_s:>17.2f}% {retain_s-retain_b:>+7.2f}%")
print(f"{'局部一致性(m,越小越好)':.<25} {nn_b:>15.4f} {nn_s:>18.4f} {nn_b-nn_s:>+8.4f}")
print("="*50)

# ── 绘图 ──────────────────────────────────────────────────────────────
BG='#0d0d0d'; WHITE='#f1f5f9'; GRAY='#9ca3af'
RED='#f87171'; BLUE='#60a5fa'; GREEN='#4ade80'; GOLD='#fbbf24'

fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG)
fig.suptitle('建图质量定量评估（KITTI 序列00，4541帧）',
             color=WHITE, fontsize=13, fontweight='bold', y=1.02)

metrics = [
    ('动态点污染率\n（越低越好）', [contam_b, contam_s], '%', True),
    ('静态点保留率\n（越高越好）', [retain_b, retain_s], '%', False),
    ('局部一致性\n平均最近邻距离（越小越好）', [nn_b, nn_s], 'm', True),
]

for ax, (title, vals, unit, lower_better) in zip(axes, metrics):
    ax.set_facecolor('#111827')
    for sp in ax.spines.values(): sp.set_color('#374151')
    ax.tick_params(colors=GRAY, labelsize=10)

    bars = ax.bar(['基线\nKISS-ICP', '语义加权\n（本研究）'],
                  vals, color=[RED, BLUE], width=0.5,
                  edgecolor='#374151', linewidth=0.5)

    # 数值标注
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.02,
                f'{val:.3f}{unit}',
                ha='center', va='bottom', color=WHITE,
                fontsize=11, fontweight='bold')

    # 改善幅度标注
    if lower_better:
        improve = (vals[0] - vals[1]) / vals[0] * 100
        color = GREEN if improve > 0 else RED
        label = f'↓ {improve:.1f}%'
    else:
        improve = (vals[1] - vals[0]) / vals[0] * 100
        color = GREEN if improve > 0 else RED
        label = f'↑ {improve:.1f}%'

    ax.text(0.98, 0.97, label, transform=ax.transAxes,
            ha='right', va='top', color=color,
            fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', fc='#0f172a',
                      ec=color, alpha=0.9))

    ax.set_title(title, color=GOLD, fontsize=11,
                 fontweight='bold', pad=8)
    ax.set_ylabel(unit, color=GRAY, fontsize=10)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.yaxis.grid(True, color='#1f2937', lw=0.5)
    ax.set_axisbelow(True)

plt.tight_layout()
out = '/home/yuan/slam_results/map_quality.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
print(f"\n图已保存：{out}")
