import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'WenQuanYi Zen Hei'
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取真值轨迹
gt = np.loadtxt('/home/yuan/kitti_data/dataset/sequences/00/poses.txt')

# 读取语义版轨迹
sem = np.loadtxt('/home/yuan/slam_results/kitti_semantic/traj.txt')

# 计算ATE
def calc_ate(est, gt):
    # 提取平移部分
    est_t = est[:, [3,7,11]]
    gt_t  = gt[:,  [3,7,11]]
    n = min(len(est_t), len(gt_t))
    diff = est_t[:n] - gt_t[:n]
    ate_per_frame = np.sqrt((diff**2).sum(axis=1))
    return ate_per_frame, ate_per_frame.mean()

ate_frames, ate_mean = calc_ate(sem, gt)
print(f"语义版 ATE 均值: {ate_mean:.3f} m")
print(f"基线   ATE 均值: 3.524 m (KISS-ICP输出)")
print(f"改善幅度: {(3.524-ate_mean)/3.524*100:.1f}%")

# 绘制ATE曲线
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(ate_frames, color='#2563eb', lw=1, label=f'语义版 ATE (均值={ate_mean:.3f}m)')
ax.axhline(y=3.524, color='#ef4444', lw=1.5, linestyle='--', label='基线 ATE (均值=3.524m)')
ax.set_xlabel('帧编号')
ax.set_ylabel('绝对轨迹误差 ATE (m)')
ax.set_title('KITTI 序列00 — 基线 vs 语义权重版本 ATE 对比')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/yuan/slam_results/ate_compare.png', dpi=150)
print("图表已保存到 ~/slam_results/ate_compare.png")
