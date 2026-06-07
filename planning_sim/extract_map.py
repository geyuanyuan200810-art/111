"""
从KITTI序列00点云提取2D占用栅格地图
基线版：动态点混入（模拟基线SLAM建图结果）
语义版：动态点过滤（模拟语义SLAM建图结果）
"""
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'
import json

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

# 栅格参数
RESOLUTION = 0.5   # 每格0.5m
X_MIN, X_MAX = -50, 530
Y_MIN, Y_MAX = -320, 310
W = int((X_MAX - X_MIN) / RESOLUTION)
H = int((Y_MAX - Y_MIN) / RESOLUTION)

grid_b = np.zeros((H, W), dtype=np.float32)  # 基线：含鬼影
grid_s = np.zeros((H, W), dtype=np.float32)  # 语义：干净

STEP = 20
N = min(len(poses_b), len(poses_s), len(vfiles), len(lfiles))

print(f"构建占用地图（每{STEP}帧采样，共约{N//STEP}帧）...")
for i in range(0, N, STEP):
    pts    = np.fromfile(vfiles[i], dtype=np.float32).reshape(-1,4)
    labels = np.fromfile(lfiles[i], dtype=np.uint32) & 0xFFFF

    dist = np.sqrt((pts[:,:3]**2).sum(axis=1))
    # 只保留地面以上、一定高度范围内的点（建筑墙面）
    mask = (dist > 1.5) & (dist < 35.0) & (pts[:,2] > -0.5) & (pts[:,2] < 3.0)
    pts    = pts[mask]; labels = labels[mask]

    ones  = np.ones((len(pts),1), dtype=np.float32)
    pts_h = np.hstack([pts[:,:3], ones])
    dyn   = np.isin(labels, list(DYNAMIC))
    sta   = np.isin(labels, list(STATIC))

    # 基线版：静态+动态点都入地图
    T_b   = poses_b[i]
    w_b   = (T_b @ pts_h.T).T[:,:2]
    all_pts_b = w_b  # 基线不过滤

    xi = ((all_pts_b[:,0] - X_MIN) / RESOLUTION).astype(int)
    yi = ((all_pts_b[:,1] - Y_MIN) / RESOLUTION).astype(int)
    valid = (xi>=0)&(xi<W)&(yi>=0)&(yi<H)
    grid_b[yi[valid], xi[valid]] += 1

    # 语义版：只有静态点入地图
    T_s   = poses_s[i]
    w_s   = (T_s @ pts_h.T).T[:,:2]
    sta_pts = w_s[sta]

    if len(sta_pts) > 0:
        xi = ((sta_pts[:,0] - X_MIN) / RESOLUTION).astype(int)
        yi = ((sta_pts[:,1] - Y_MIN) / RESOLUTION).astype(int)
        valid = (xi>=0)&(xi<W)&(yi>=0)&(yi<H)
        grid_s[yi[valid], xi[valid]] += 1

    if i % 200 == 0: print(f"  帧 {i}/{N}")

# 二值化（累积超过阈值视为占用）
thresh = 2
occ_b = (grid_b > thresh).astype(np.uint8)
occ_s = (grid_s > thresh).astype(np.uint8)

print(f"基线地图占用格数: {occ_b.sum():,}")
print(f"语义地图占用格数: {occ_s.sum():,}")
print(f"鬼影格数（基线多出的）: {(occ_b.astype(int)-occ_s.astype(int)).clip(0).sum():,}")

# 保存地图
np.save('/home/yuan/planning_sim/map_baseline.npy', occ_b)
np.save('/home/yuan/planning_sim/map_semantic.npy',  occ_s)

# 保存地图参数（供仿真器读取）
map_params = {
    'resolution': RESOLUTION,
    'x_min': X_MIN, 'x_max': X_MAX,
    'y_min': Y_MIN, 'y_max': Y_MAX,
    'width': W, 'height': H
}
with open('/home/yuan/planning_sim/map_params.json', 'w') as f:
    json.dump(map_params, f)

# 可视化对比
fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor='#0d0d0d')
fig.suptitle('KITTI序列00 二维占用栅格地图对比',
             color='white', fontsize=13, fontweight='bold')

for ax, grid, title in zip(axes,
    [occ_b, occ_s],
    ['基线SLAM地图（含动态鬼影）', '语义SLAM地图（动态点已过滤）']):
    ax.set_facecolor('#111827')
    ax.imshow(grid, origin='lower', cmap='Greens',
              extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], alpha=0.8)
    ax.set_title(title, color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (m)', color='gray'); ax.set_ylabel('Y (m)', color='gray')
    ax.tick_params(colors='gray')

plt.tight_layout()
plt.savefig('/home/yuan/planning_sim/map_compare_2d.png',
            dpi=150, facecolor='#0d0d0d')
print("地图已保存！")
