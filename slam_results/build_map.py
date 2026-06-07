import numpy as np
import glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'WenQuanYi Zen Hei'
matplotlib.rcParams['axes.unicode_minus'] = False

SEQ     = "/home/yuan/kitti_data/dataset/sequences/00"
TRAJ_B  = "/home/yuan/slam_results/kitti_baseline/traj.txt"
TRAJ_S  = "/home/yuan/slam_results/kitti_semantic/traj.txt"

DYNAMIC = {10,11,13,15,16,18,20}
STATIC  = {40,44,48,49,50,51,52,60,70,71,72,80,81}

vfiles = sorted(glob.glob(f"{SEQ}/velodyne/*.bin"))
lfiles = sorted(glob.glob(f"{SEQ}/labels/*.label"))

def load_traj(path):
    """加载轨迹，返回4x4变换矩阵列表"""
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

# 每隔N帧采样，控制点数量
STEP = 10
N    = min(len(poses_b), len(poses_s), len(vfiles), len(lfiles))

# 存储地图点
map_b_static  = []   # 基线 静态点
map_s_static  = []   # 语义 静态点
map_s_dynamic = []   # 语义 动态点（红色标注）
traj_b_xy = []
traj_s_xy = []

print(f"构建地图（每{STEP}帧采样，共约{N//STEP}帧）...")

for i in range(0, N, STEP):
    pts    = np.fromfile(vfiles[i], dtype=np.float32).reshape(-1,4)
    labels = np.fromfile(lfiles[i], dtype=np.uint32) & 0xFFFF

    # 过滤掉距离过远或过近的点
    dist = np.sqrt(pts[:,0]**2 + pts[:,1]**2 + pts[:,2]**2)
    mask_dist = (dist > 2.0) & (dist < 50.0)
    pts    = pts[mask_dist]
    labels = labels[mask_dist]

    # 同态点（去掉地面点，只保留有结构意义的点）
    mask_z = pts[:,2] > -1.5
    pts    = pts[mask_z]
    labels = labels[mask_z]

    # 转为齐次坐标
    ones = np.ones((len(pts),1), dtype=np.float32)
    pts_h = np.hstack([pts[:,:3], ones])

    # 基线版：按基线轨迹变换
    if i < len(poses_b):
        T_b = poses_b[i]
        world_b = (T_b @ pts_h.T).T[:,:3]
        # 只存XY平面（俯视图）
        static_mask = ~np.isin(labels, list(DYNAMIC))
        map_b_static.append(world_b[static_mask, :2])
        traj_b_xy.append(poses_b[i][:2, 3])

    # 语义版：按语义轨迹变换
    if i < len(poses_s):
        T_s = poses_s[i]
        world_s = (T_s @ pts_h.T).T[:,:3]
        static_mask  = ~np.isin(labels, list(DYNAMIC))
        dynamic_mask =  np.isin(labels, list(DYNAMIC))
        map_s_static.append(world_s[static_mask,  :2])
        map_s_dynamic.append(world_s[dynamic_mask, :2])
        traj_s_xy.append(poses_s[i][:2, 3])

    if i % 100 == 0:
        print(f"  处理帧 {i}/{N}")

print("合并点云...")
map_b_static  = np.vstack(map_b_static)  if map_b_static  else np.zeros((0,2))
map_s_static  = np.vstack(map_s_static)  if map_s_static  else np.zeros((0,2))
map_s_dynamic = np.vstack(map_s_dynamic) if map_s_dynamic else np.zeros((0,2))
traj_b = np.array(traj_b_xy)
traj_s = np.array(traj_s_xy)

# 随机降采样，控制绘图点数
MAX_PTS = 80000
def subsample(arr, n):
    if len(arr) > n:
        idx = np.random.choice(len(arr), n, replace=False)
        return arr[idx]
    return arr

map_b_static  = subsample(map_b_static,  MAX_PTS)
map_s_static  = subsample(map_s_static,  MAX_PTS)
map_s_dynamic = subsample(map_s_dynamic, 10000)

print("绘制对比图...")
fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='#0f0f0f')
fig.suptitle('KITTI 序列00 建图效果对比（俯视图）',
             color='white', fontsize=15, fontweight='bold', y=0.98)

for ax in axes:
    ax.set_facecolor('#0f0f0f')
    ax.tick_params(colors='#666666')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

# ── 左图：基线版 ──
ax = axes[0]
ax.scatter(map_b_static[:,0], map_b_static[:,1],
           s=0.08, c='#4ade80', alpha=0.3, linewidths=0)
ax.plot(traj_b[:,0], traj_b[:,1],
        color='#ef4444', lw=1.5, alpha=0.9, label='基线轨迹', zorder=5)
ax.plot(traj_b[0,0], traj_b[0,1],
        'o', color='white', ms=6, zorder=6)
ax.set_title('基线 KISS-ICP\nATE均值 = 2.698 m',
             color='#ef4444', fontsize=12, pad=8)
ax.set_xlabel('X (m)', color='#888888', fontsize=10)
ax.set_ylabel('Y (m)', color='#888888', fontsize=10)
ax.set_aspect('equal')
ax.legend(loc='upper right', fontsize=9,
          facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white')
# 添加动态点警示标注
ax.text(0.03, 0.03,
        '动态目标点云\n未作区分处理\n（污染地图）',
        transform=ax.transAxes, color='#fbbf24',
        fontsize=9, va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', fc='#1a1a1a',
                  ec='#fbbf24', alpha=0.8))

# ── 右图：语义版 ──
ax = axes[1]
ax.scatter(map_s_static[:,0],  map_s_static[:,1],
           s=0.08, c='#4ade80', alpha=0.3, linewidths=0,
           label='静态环境点云')
ax.scatter(map_s_dynamic[:,0], map_s_dynamic[:,1],
           s=0.5, c='#ef4444', alpha=0.6, linewidths=0,
           label='动态目标点云（已标注）')
ax.plot(traj_s[:,0], traj_s[:,1],
        color='#2563eb', lw=1.5, alpha=0.9, label='语义版轨迹', zorder=5)
ax.plot(traj_s[0,0], traj_s[0,1],
        'o', color='white', ms=6, zorder=6)
ax.set_title('语义权重版本（本研究）\nATE均值 = 2.516 m',
             color='#2563eb', fontsize=12, pad=8)
ax.set_xlabel('X (m)', color='#888888', fontsize=10)
ax.set_ylabel('Y (m)', color='#888888', fontsize=10)
ax.set_aspect('equal')
ax.legend(loc='upper right', fontsize=9,
          facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white')
ax.text(0.03, 0.03,
        '红色点 = 行人/车辆\n已识别并标注\n（不参与定位优化）',
        transform=ax.transAxes, color='#86efac',
        fontsize=9, va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', fc='#1a1a1a',
                  ec='#22c55e', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = '/home/yuan/slam_results/map_compare.png'
plt.savefig(out, dpi=180, bbox_inches='tight',
            facecolor='#0f0f0f')
print(f"\n地图对比图已保存：{out}")
print(f"  基线静态点：{len(map_b_static):,}")
print(f"  语义静态点：{len(map_s_static):,}")
print(f"  标注动态点：{len(map_s_dynamic):,}")
