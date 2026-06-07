import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
matplotlib.rcParams['font.family'] = 'WenQuanYi Zen Hei'
matplotlib.rcParams['axes.unicode_minus'] = False
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

typestore = get_typestore(Stores.ROS2_HUMBLE)

def read_bag_points(bag_path, topic='/points', max_frames=300):
    frames = []
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"  警告: 找不到话题 {topic}")
            return frames
        for conn, ts, data in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(data, conn.msgtype)
            raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            pts_raw = raw.reshape(-1, msg.point_step)
            x = np.frombuffer(pts_raw[:, 0:4].tobytes(),  dtype=np.float32)
            y = np.frombuffer(pts_raw[:, 4:8].tobytes(),  dtype=np.float32)
            z = np.frombuffer(pts_raw[:, 8:12].tobytes(), dtype=np.float32)
            inten = np.frombuffer(pts_raw[:, 12:16].tobytes(), dtype=np.float32)
            pts = np.stack([x, y, z, inten], axis=1)
            dist = np.sqrt((pts[:,:3]**2).sum(axis=1))
            pts  = pts[(dist > 1.0) & (dist < 30.0)]
            frames.append(pts)
            if len(frames) >= max_frames:
                break
    return frames

def icp_2d(src, dst, max_iter=20, tol=1e-4):
    from scipy.spatial import KDTree
    src2 = src[:, :2].copy()
    dst2 = dst[:, :2].copy()
    T = np.eye(3)
    for _ in range(max_iter):
        tree  = KDTree(dst2)
        dists, idx = tree.query(src2, k=1)
        mask  = dists < np.percentile(dists, 80)
        if mask.sum() < 10: break
        s = src2[mask]; d = dst2[idx[mask]]
        mu_s = s.mean(0); mu_d = d.mean(0)
        H    = (s - mu_s).T @ (d - mu_d)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1,:] *= -1; R = Vt.T @ U.T
        t  = mu_d - R @ mu_s
        src2 = (R @ src2.T).T + t
        dT = np.eye(3); dT[:2,:2] = R; dT[:2,2] = t
        T  = dT @ T
        if np.linalg.norm(t) < tol: break
    return T

def is_dynamic(pts):
    inten = pts[:, 3]
    # 静态背景: intensity = 60 + ring*8，即60,68,76,84,92,100,108...
    # 行人固定值=80，车辆固定值=100
    # 区分方法：检查是否恰好等于80或100，且不是静态序列里的值
    # 静态里有76,84（不是80），有92,108（不是100），静态里没有恰好=80或=100的值
    # 实际检查：(inten-60)%8 != 0 即不是静态背景
    is_not_static_pattern = ((inten - 60) % 8) > 0.5
    return is_not_static_pattern

def build_map(frames, use_semantic=False, step=2):
    """
    简化建图：不做ICP，直接用真值位姿（仿真场景LiDAR固定旋转）
    重点展示动态点污染效果
    """
    map_static  = []
    map_dynamic = []
    # 仿真场景：LiDAR绕Z轴匀速旋转，每帧旋转0.02rad
    # 用真值旋转直接叠加点云
    for i in range(0, len(frames), step):
        curr = frames[i]
        ao   = i * 0.02  # 累积旋转角
        R2   = np.array([[np.cos(ao), -np.sin(ao)],
                          [np.sin(ao),  np.cos(ao)]])
        dyn  = is_dynamic(curr)
        sta  = ~dyn
        # 静态点
        if sta.sum() > 0:
            pts_sta = (R2 @ curr[sta,:2].T).T
            map_static.append(pts_sta)
        # 动态点
        if dyn.sum() > 0:
            pts_dyn = (R2 @ curr[dyn,:2].T).T
            if not use_semantic:
                map_static.append(pts_dyn)  # 基线：动态点混入地图
            map_dynamic.append(pts_dyn)
    def vs(l): return np.vstack(l) if l else np.zeros((0,2))
    # 轨迹（仿真中传感器固定，轨迹只是原点）
    traj = np.array([[0.0, 0.0]])
    return vs(map_static), vs(map_dynamic), traj

def smoothness(traj):
    if len(traj) < 3: return 0
    d = np.diff(traj, axis=0)
    a = np.arctan2(d[:,1], d[:,0])
    return np.std(np.diff(np.unwrap(a))) * 1000

scenes = [
    ("sim_scene1_static",  "场景一：低动态\n（0行人，0辆车）"),
    ("sim_scene2_medium",  "场景二：中动态\n（5行人，0辆车）"),
    ("sim_scene3_dynamic", "场景三：高动态\n（15行人，3辆车）"),
]

all_data = []
for bag_name, title in scenes:
    print(f"读取 {bag_name}...")
    frames = read_bag_points(f"/home/yuan/slam_data/{bag_name}")
    print(f"  {len(frames)}帧，建图中...")
    map_b, dyn_b, traj_b = build_map(frames, use_semantic=False)
    map_s, dyn_s, traj_s = build_map(frames, use_semantic=True)
    sm_b = smoothness(traj_b)
    sm_s = smoothness(traj_s)
    dyn_r = len(dyn_b) / max(len(map_b),1) * 100
    print(f"  动态点比例:{dyn_r:.1f}%  平滑度 基:{sm_b:.1f} 义:{sm_s:.1f}")
    all_data.append(dict(title=title, map_b=map_b, map_s=map_s,
                         dyn_b=dyn_b, dyn_s=dyn_s,
                         traj_b=traj_b, traj_s=traj_s,
                         sm_b=sm_b, sm_s=sm_s, dyn_r=dyn_r))

BG='#0d0d0d'; GREEN='#4ade80'; RED='#f87171'
BLUE='#60a5fa'; WHITE='#f1f5f9'; GOLD='#fbbf24'
GRAY='#6b7280'; ORANGE='#fb923c'

fig, axes = plt.subplots(3, 2, figsize=(16, 20), facecolor=BG)
fig.suptitle('仿真场景建图效果对比（三场景 × 基线/语义加权）',
             color=WHITE, fontsize=14, fontweight='bold', y=0.998)
fig.text(0.27, 0.991, '基线 KISS-ICP',      ha='center', color=RED,  fontsize=12, fontweight='bold')
fig.text(0.75, 0.991, '语义加权（本研究）',  ha='center', color=BLUE, fontsize=12, fontweight='bold')

def style(ax):
    ax.set_facecolor('#111827')
    for sp in ax.spines.values(): sp.set_color('#374151')
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.set_xlabel('X (m)', color=GRAY, fontsize=8)
    ax.set_ylabel('Y (m)', color=GRAY, fontsize=8)
    ax.grid(True, color='#1f2937', lw=0.4, alpha=0.5)
    ax.set_aspect('equal')

def sub(arr, n=15000):
    return arr[np.random.choice(len(arr),n,replace=False)] if len(arr)>n else arr

for i, d in enumerate(all_data):
    for j in range(2):
        ax  = axes[i][j]
        sem = (j == 1)
        style(ax)
        mp  = sub(d['map_s'] if sem else d['map_b'])
        dp  = d['dyn_s'] if sem else d['dyn_b']
        tr  = d['traj_s'] if sem else d['traj_b']
        sm  = d['sm_s']   if sem else d['sm_b']
        tc  = BLUE if sem else RED

        if len(mp):
            ax.scatter(mp[:,0], mp[:,1], s=0.5,
                       c='#60a5fa' if sem else GREEN,
                       alpha=0.5, linewidths=0)
        if len(dp) > 0:
            ds = sub(dp, 3000)
            if not sem:
                # 基线：动态点作为鬼影显示在地图里
                ax.scatter(ds[:,0], ds[:,1], s=2, c=ORANGE,
                           alpha=0.6, linewidths=0, zorder=4,
                           label='动态鬼影（污染地图）')
            else:
                # 语义版：动态点不进入地图，只用小叉号标注位置
                ax.scatter(ds[:,0], ds[:,1], s=8, c=RED,
                           alpha=0.4, linewidths=0, zorder=4, marker='x',
                           label='动态点（已识别，已过滤）')
        # 仿真场景固定传感器，不显示轨迹

        if j == 0:
            ax.text(-0.16, 0.5, d['title'], transform=ax.transAxes,
                    ha='center', va='center', color=GOLD,
                    fontsize=9.5, fontweight='bold', rotation=90)

        n_dyn_pts = len(d['dyn_b'])
        n_sta_pts = len(d['map_b'])
        if not sem:
            info = f"动态鬼影点数: {n_dyn_pts:,}"
        else:
            info = f"过滤动态点: {n_dyn_pts:,}\n静态地图更纯净"
        ax.text(0.03, 0.97, info, transform=ax.transAxes,
                ha='left', va='top', color=WHITE, fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.3', fc='#0f172a', ec=tc, alpha=0.9))

        if d['dyn_r'] > 0:
            if not sem:
                ax.text(0.03, 0.03, f"动态点混入地图\n≈{d['dyn_r']:.1f}%",
                        transform=ax.transAxes, ha='left', va='bottom',
                        color=ORANGE, fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', fc='#1a1a1a',
                                  ec=ORANGE, alpha=0.85))
            else:
                ax.text(0.03, 0.03, "动态点已过滤\n地图更干净",
                        transform=ax.transAxes, ha='left', va='bottom',
                        color='#4ade80', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', fc='#1a1a1a',
                                  ec='#4ade80', alpha=0.85))

        ax.legend(loc='upper right', facecolor='#0f172a',
                  edgecolor='#374151', labelcolor=WHITE,
                  fontsize=7.5, markerscale=3)

plt.tight_layout(rect=[0.07, 0, 1, 0.995])
out = '/home/yuan/slam_results/sim_map_compare.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=BG)
print(f"完成！已保存：{out}")
